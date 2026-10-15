from collections.abc import Callable
from dataclasses import replace
from datetime import date
from decimal import Decimal

from portfolio_client.domain.accounting.corporate_actions import apply_corporate_action
from portfolio_client.domain.accounting.costs import default_fee_schedules, schedule_for
from portfolio_client.domain.accounting.ledger import advance_session, value_account
from portfolio_client.domain.accounting.models import AccountState, Side
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.execution.executor import execute_order
from portfolio_client.domain.execution.models import KnownLiquidity, Order, OrderState
from portfolio_client.domain.execution.rules import ExecutionRules, lot_size_for
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.market.ports import MarketDataPort
from portfolio_client.domain.portfolio.rebalancing import plan_rebalance
from portfolio_client.domain.research.backtest_models import (
    BacktestEvidence,
    BacktestRequest,
    BacktestResult,
    Frequency,
    HoldingsPoint,
    NavPoint,
    Progress,
    Signal,
    Trade,
    UnfilledOrder,
)
from portfolio_client.domain.research.market_window import MarketWindow

__all__ = ["BacktestRequest", "BacktestResult", "Progress", "is_rebalance_session", "run_backtest"]


def is_rebalance_session(day: date, next_day: date, frequency: Frequency) -> bool:
    if frequency == "daily":
        return True
    if frequency == "weekly":
        return day.isocalendar()[:2] != next_day.isocalendar()[:2]
    if frequency == "monthly":
        return (day.year, day.month) != (next_day.year, next_day.month)
    return day.year != next_day.year


def run_backtest(
    request: BacktestRequest, market: MarketDataPort, progress: Callable[[Progress], None]
) -> BacktestResult:
    calendar = market.calendar(request.dataset_id)
    sessions = [day for day in calendar.sessions if request.start <= day <= request.end]
    if (
        not sessions
        or request.start not in calendar.sessions
        or request.end not in calendar.sessions
    ):
        raise DomainError("INVALID_BACKTEST_RANGE", "起止日期必须属于选定交易日历")
    identities = {
        item.security_id: item
        for item in market.securities(request.dataset_id, at_close(request.start))
    }
    selected_ids = sorted(
        set(request.security_ids) | ({request.benchmark_id} if request.benchmark_id else set())
    )
    unknown = set(selected_ids) - identities.keys()
    if unknown:
        raise DomainError(
            "SECURITY_NOT_FOUND", "回测起点存在未知证券", {"security_ids": sorted(unknown)}
        )
    lots = {key: lot_size_for(identities[key]) for key in request.security_ids}
    bars = market.bars(request.dataset_id, selected_ids, at_close(request.end))
    window = MarketWindow(bars, calendar)
    actions = market.actions(request.dataset_id, at_close(request.end))
    schedules = request.fee_schedules or default_fee_schedules()
    state = AccountState(cash=request.capital, withdrawable_cash=request.capital)
    nav: list[NavPoint] = []
    holdings: list[HoldingsPoint] = []
    trades: list[Trade] = []
    signals: list[Signal] = []
    unfilled: list[UnfilledOrder] = []
    pending: tuple[Order, ...] = ()
    liquidity: dict[str, KnownLiquidity] = {}
    prices: dict[str, Decimal] = {}
    warmup_seen, missing_windows = False, 0
    progress(Progress(0, len(sessions), "准备历史窗口"))
    for index, day in enumerate(sessions):
        state = advance_session(state, day)
        known_actions = [action for action in actions if action.announced_at <= at_close(day)]
        for action in known_actions:
            if action.record_date != day:
                state = apply_corporate_action(state, action, day)
        consumed: dict[str, int] = {}
        for order in sorted(pending, key=lambda item: (item.side is Side.BUY, item.security_id)):
            bar = window.bar(order.security_id, day)
            if bar is None:
                unfilled.append(UnfilledOrder(order, "missing_execution_bar"))
                continue
            position = state.positions.get(order.security_id)
            full_liquidation = (
                order.side is Side.SELL
                and position is not None
                and order.quantity == position.quantity
            )
            rules = ExecutionRules(
                lots[order.security_id],
                request.slippage_bps,
                request.participation_rate,
                full_liquidation,
                request.require_price_limits,
            )
            known = replace(
                liquidity[order.security_id], consumed_quantity=consumed.get(order.security_id, 0)
            )
            outcome = execute_order(state, order, bar, known, rules, schedule_for(schedules, day))
            state = outcome.account
            if outcome.match.fill is not None and outcome.fees is not None:
                trades.append(Trade(order.order_id, outcome.match.fill, outcome.fees, bar.open))
                consumed[order.security_id] = (
                    consumed.get(order.security_id, 0) + outcome.match.fill.quantity
                )
            if outcome.match.reason:
                unfilled.append(UnfilledOrder(order, outcome.match.reason))
        pending = ()
        stale: list[str] = []
        for key in request.security_ids:
            bar = window.bar(key, day)
            if bar is not None:
                prices[key] = bar.close
            elif key in state.positions:
                stale.append(key)
        # Entitlements are captured at record-date CLOSE, after that day's trades.
        for action in known_actions:
            if action.record_date == day:
                state = apply_corporate_action(state, action, day)
        nav.append(
            NavPoint(day, value_account(state, prices), state.cash, state.receivables, tuple(stale))
        )
        holdings.append(HoldingsPoint(day, state.positions.copy()))
        history = window.window(request.security_ids, day)
        warmup_seen = warmup_seen or history is not None
        if index + 1 < len(sessions) and is_rebalance_session(
            day, sessions[index + 1], request.frequency
        ):
            if history is None:
                missing_windows += 1
            else:
                pending_quantities: dict[str, int] = {}
                for shares in state.pending_shares.values():
                    pending_quantities[shares.security_id] = (
                        pending_quantities.get(shares.security_id, 0) + shares.quantity
                    )
                plan = plan_rebalance(
                    request.weights,
                    {key: p.quantity for key, p in state.positions.items()},
                    prices,
                    state.available_cash,
                    lots,
                    schedule_for(schedules, day),
                    receivables=state.receivables,
                    pending_holdings=pending_quantities,
                )
                next_day = sessions[index + 1]
                pending = tuple(
                    Order(
                        f"{day.isoformat()}:{item.security_id}:{item.side}",
                        "backtest",
                        item.security_id,
                        item.side,
                        item.quantity,
                        "market",
                        None,
                        at_close(day),
                        next_day,
                        OrderState.SUBMITTED,
                    )
                    for item in plan.orders
                )
                signals.append(Signal(day, next_day, pending))
                for key in request.security_ids:
                    volumes = [bar.volume for bar in history if bar.security_id == key][-20:]
                    liquidity[key] = KnownLiquidity(
                        Decimal(sum(volumes)) / len(volumes), at_close(day)
                    )
        progress(Progress(index + 1, len(sessions), day.isoformat()))
    if not warmup_seen and any(request.weights.values()):
        raise DomainError("INSUFFICIENT_HISTORY", "区间内没有21个完整交易日价格窗口，无法首次建仓")
    warnings = [f"{missing_windows}个信号日缺少完整窗口，未生成订单。"] if missing_windows else []
    if not trades:
        warnings.append("区间没有成交，请检查周期边界、历史窗口、申报单位、价格限制和资金。")
    assumptions = (
        "收盘信号、下一交易日开盘成交；仅开盘撮合，未触发的日单与部分成交剩余量当日失效。",
        "开盘达到涨停拒买、达到跌停拒卖；不使用当日最终高低价判断能否排队成交。",
        "容量为前20日已知平均成交股数乘参与率，非当日实际成交保证；不使用当日最终成交量。",
        "执行和估值使用原始价格；缺失持仓价格沿用最后已知价格并标为stale，不虚构退市卖出。",
        "固定目标权重、100股整数手、方向滑点进入价格；费用按成交日有效版本逐项舍入。",
        "现金分红按登记日收盘权益、税前应收及实际支付日记账；送转上市日期由pay_date提供。",
        "seed记录于证据；撮合无随机过程，相同固定版本输入产生相同结果。",
    )
    if not request.require_price_limits:
        warnings.append("用户关闭了缺失涨跌停价保护，未提供限制的行情可能高估成交。")
    slippage = sum(
        (
            (trade.fill.price - trade.reference_price)
            * trade.fill.quantity
            * (1 if trade.fill.side is Side.BUY else -1)
            for trade in trades
        ),
        Decimal("0"),
    )
    benchmark = None
    if request.benchmark_id is not None:
        benchmark = tuple(
            NavPoint(point.session, bar.close, Decimal("0"), Decimal("0"), ())
            for point in nav
            if (bar := window.bar(request.benchmark_id, point.session)) is not None
        )
    return BacktestResult(
        request=request,
        nav=tuple(nav),
        positions=tuple(holdings),
        trades=tuple(trades),
        signals=tuple(signals),
        unfilled=tuple(unfilled),
        costs=sum((trade.fees.total for trade in trades), Decimal("0")),
        slippage_costs=slippage,
        final_account=state,
        evidence=BacktestEvidence(
            request.dataset_id,
            request.calendar_id,
            "opening-v1",
            tuple(sorted({trade.fees.schedule_version for trade in trades})),
            request.seed,
            request.knowledge_mode,
        ),
        assumptions=assumptions,
        warnings=tuple(warnings),
        benchmark=benchmark,
    )
