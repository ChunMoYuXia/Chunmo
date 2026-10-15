from dataclasses import replace
from datetime import date
from decimal import Decimal

from portfolio_client.domain.accounting.corporate_actions import apply_corporate_action
from portfolio_client.domain.accounting.costs import calculate_fees, money, schedule_for
from portfolio_client.domain.accounting.ledger import (
    advance_session,
    release_cash,
    reserve_cash,
    value_account,
)
from portfolio_client.domain.accounting.models import Side
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.execution.executor import execute_order
from portfolio_client.domain.execution.models import KnownLiquidity, Order, OrderState
from portfolio_client.domain.execution.paper_models import (
    AccountSnapshot,
    OrderRequest,
    PaperAccount,
    PaperOrder,
    PaperTransition,
)
from portfolio_client.domain.execution.rules import ExecutionRules, lot_size_for
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.market.ports import MarketDataPort
from portfolio_client.domain.research.backtest_models import Trade
from portfolio_client.domain.research.market_window import MarketWindow

ASSUMPTIONS = (
    "历史逐日模拟，不连接券商；收盘提交、下一交易日开盘撮合，未成交余量当日失效。",
    "市场单按当前原始收盘价加20%缓冲及费用冻结；成交时重新验资，可部分成交。",
    "容量基于此前20日已知成交量；开盘触及涨跌停保守拒绝相应方向。",
    "现金可取已扣冻结；卖出资金当日可买，下一交易日可取；买入股票T+1。",
    "股利为税前金额；应收与付款分开，送转交付日期由数据提供。",
)


def snapshot(
    account: PaperAccount, orders: list[PaperOrder], market: MarketDataPort
) -> AccountSnapshot:
    state = account.state
    if state.session is None:
        raise DomainError("INVALID_ACCOUNT", "模拟账户缺少当前交易日")
    keys = set(state.positions) | {shares.security_id for shares in state.pending_shares.values()}
    bars = (
        market.bars(account.dataset_id, sorted(keys), at_close(state.session), lookback=1)
        if keys
        else []
    )
    prices = {bar.security_id: bar.close for bar in bars}
    stale = tuple(sorted(bar.security_id for bar in bars if bar.trade_date != state.session))
    reserved: dict[str, int] = {}
    for pending in orders:
        if pending.order.state is OrderState.SUBMITTED and pending.order.side is Side.SELL:
            key = pending.order.security_id
            reserved[key] = reserved.get(key, 0) + pending.order.quantity
    available = {
        key: max(0, position.sellable - reserved.get(key, 0))
        for key, position in state.positions.items()
    }
    return AccountSnapshot(
        account.id,
        account.name,
        account.dataset_id,
        account.calendar_id,
        account.data_version,
        account.source_kind,
        state.session,
        account.status,
        state.cash,
        state.frozen_cash,
        state.available_cash,
        state.available_to_withdraw,
        state.receivables,
        state.positions.copy(),
        reserved,
        available,
        value_account(state, prices),
        stale,
        state.ledger_offset + len(state.events),
        ASSUMPTIONS,
    )


def prepare_order(
    account: PaperAccount,
    command: OrderRequest,
    order_id: str,
    orders: list[PaperOrder],
    market: MarketDataPort,
) -> PaperTransition:
    if account.status != "active":
        raise DomainError("ACCOUNT_PAUSED", "账户已暂停，恢复后才能提交新单")
    day = account.state.session
    if day is None:
        raise DomainError("INVALID_ACCOUNT", "账户当前日期缺失")
    identities = {
        security.security_id: security
        for security in market.securities(account.dataset_id, at_close(day))
    }
    if command.security_id not in identities:
        raise DomainError("SECURITY_NOT_FOUND", "该时点没有对应证券")
    lot = lot_size_for(identities[command.security_id])
    position = account.state.positions.get(command.security_id)
    odd_liquidation = (
        command.side is Side.SELL and position is not None and command.quantity == position.quantity
    )
    if command.quantity % lot and not odd_liquidation:
        raise DomainError("INVALID_LOT", "订单数量必须符合100股整数手或全部卖出零股规则")
    calendar = market.calendar(account.dataset_id)
    history = market.bars(account.dataset_id, [command.security_id], at_close(day), lookback=21)
    window = MarketWindow(history, calendar).window([command.security_id], day)
    if window is None:
        raise DomainError("INSUFFICIENT_HISTORY", "提交订单需要完整21日价格与20日流动性历史")
    order = Order(
        order_id,
        account.id,
        command.security_id,
        command.side,
        command.quantity,
        command.order_type,
        command.limit_price,
        at_close(day),
        calendar.next_session(day),
        OrderState.SUBMITTED,
    )
    reserved_cash = Decimal("0")
    state = account.state
    if order.side is Side.BUY:
        reference = (
            command.limit_price
            if command.limit_price is not None
            else money(window[-1].close * Decimal("1.20"))
        )
        notional = money(reference * order.quantity)
        reserved_cash = (
            notional
            + calculate_fees(notional, Side.BUY, schedule_for(account.fee_schedules, day)).total
        )
        state = reserve_cash(state, reserved_cash)
    else:
        already_reserved = sum(
            pending.order.quantity
            for pending in orders
            if pending.order.state is OrderState.SUBMITTED
            and pending.order.security_id == order.security_id
            and pending.order.side is Side.SELL
        )
        if position is None or order.quantity > position.sellable - already_reserved:
            raise DomainError("INSUFFICIENT_SELLABLE", "卖出超过未冻结的T+1可卖数量")
    sequence = max((item.sequence for item in orders), default=0) + 1
    return PaperTransition(
        replace(account, state=state), (PaperOrder(order, sequence, reserved_cash),)
    )


def cancel_order(account: PaperAccount, pending: PaperOrder) -> PaperTransition:
    if pending.order.state is not OrderState.SUBMITTED:
        raise DomainError("ORDER_NOT_CANCELABLE", "该订单已经终结，无法撤销")
    state = (
        release_cash(account.state, pending.reserved_cash)
        if pending.reserved_cash
        else account.state
    )
    changed = replace(
        pending,
        order=replace(pending.order, state=OrderState.CANCELED),
        reserved_cash=Decimal("0"),
        reason="canceled_by_user",
    )
    return PaperTransition(replace(account, state=state), (changed,))


def advance_paper(
    account: PaperAccount, orders: list[PaperOrder], market: MarketDataPort, day: date
) -> PaperTransition:
    if account.status != "active":
        raise DomainError("ACCOUNT_PAUSED", "账户暂停时不能推进撮合")
    previous_day = account.state.session
    if (
        previous_day is None
        or market.calendar(account.dataset_id).next_session(previous_day) != day
    ):
        raise DomainError("INVALID_SESSION", "每次只能推进到下一实际交易日")
    state = advance_session(account.state, day)
    actions = market.actions(account.dataset_id, at_close(day))
    for action in actions:
        if action.record_date != day:
            state = apply_corporate_action(state, action, day)
    changed, trades = [], []
    consumed: dict[str, int] = {}
    identities = {
        security.security_id: security
        for security in market.securities(account.dataset_id, at_close(previous_day))
    }
    calendar = market.calendar(account.dataset_id)
    for pending in sorted(orders, key=lambda item: (item.order.side is Side.BUY, item.sequence)):
        order = pending.order
        if order.state is not OrderState.SUBMITTED:
            continue
        if pending.reserved_cash:
            state = release_cash(state, pending.reserved_cash)
        bars = market.bars(account.dataset_id, [order.security_id], at_close(day), lookback=22)
        window = MarketWindow(bars, calendar)
        bar = window.bar(order.security_id, day)
        history = window.window([order.security_id], previous_day)
        fill_quantity = 0
        reason: str | None = "missing_execution_bar"
        if bar is not None and history is not None:
            volumes = [bar.volume for bar in history][-20:]
            known = KnownLiquidity(
                Decimal(sum(volumes)) / len(volumes),
                at_close(previous_day),
                consumed.get(order.security_id, 0),
            )
            position = state.positions.get(order.security_id)
            full_liquidation = (
                order.side is Side.SELL
                and position is not None
                and order.quantity == position.quantity
            )
            rules = ExecutionRules(
                lot_size_for(identities[order.security_id]),
                account.slippage_bps,
                account.participation_rate,
                full_liquidation,
                account.require_price_limits,
            )
            outcome = execute_order(
                state, order, bar, known, rules, schedule_for(account.fee_schedules, day)
            )
            state = outcome.account
            reason = outcome.match.reason
            if outcome.match.fill is not None and outcome.fees is not None:
                fill_quantity = outcome.match.fill.quantity
                consumed[order.security_id] = consumed.get(order.security_id, 0) + fill_quantity
                trades.append(Trade(order.order_id, outcome.match.fill, outcome.fees, bar.open))
        final_state = OrderState.FILLED if fill_quantity == order.quantity else OrderState.EXPIRED
        changed.append(
            replace(
                pending,
                order=replace(order, state=final_state),
                reserved_cash=Decimal("0"),
                filled_quantity=fill_quantity,
                reason=reason,
            )
        )
    for action in actions:
        if action.record_date == day:
            state = apply_corporate_action(state, action, day)
    return PaperTransition(replace(account, state=state), tuple(changed), tuple(trades))
