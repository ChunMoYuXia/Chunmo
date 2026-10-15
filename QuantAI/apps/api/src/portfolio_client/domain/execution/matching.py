from datetime import UTC, datetime, time
from decimal import Decimal

from portfolio_client.domain.accounting.costs import BPS_DIVISOR, money
from portfolio_client.domain.accounting.models import Fill, Side
from portfolio_client.domain.execution.models import KnownLiquidity, MatchResult, Order, OrderState
from portfolio_client.domain.execution.rules import ExecutionRules
from portfolio_client.domain.market.clock import CN_TIMEZONE, as_utc
from portfolio_client.domain.market.models import DailyBar


def match_order(
    order: Order, bar: DailyBar, known_liquidity: KnownLiquidity, rules: ExecutionRules
) -> MatchResult:
    def no_fill(reason: str) -> MatchResult:
        return MatchResult("unfilled", None, reason)

    if order.state is not OrderState.SUBMITTED:
        return no_fill("order_not_open")
    opening_time = datetime.combine(bar.trade_date, time(9, 30), CN_TIMEZONE).astimezone(UTC)
    if (
        order.security_id != bar.security_id
        or order.eligible_session != bar.trade_date
        or order.submitted_at >= opening_time
    ):
        return no_fill("not_eligible")
    if (
        not (order.side is Side.SELL and rules.allow_odd_lot_sell)
        and order.quantity % rules.lot_size
    ):
        return no_fill("invalid_lot")
    if bar.suspended:
        return no_fill("suspended")
    if rules.require_price_limits and (bar.limit_up is None or bar.limit_down is None):
        return no_fill("unknown_price_limits")
    if order.side is Side.BUY and bar.limit_up is not None and bar.open >= bar.limit_up:
        return no_fill("limit_up_at_open")
    if order.side is Side.SELL and bar.limit_down is not None and bar.open <= bar.limit_down:
        return no_fill("limit_down_at_open")
    if as_utc(known_liquidity.as_of) >= opening_time:
        return no_fill("future_liquidity")
    if (
        not known_liquidity.average_daily_volume.is_finite()
        or known_liquidity.average_daily_volume <= 0
    ):
        return no_fill("unknown_liquidity")
    capacity = max(
        0,
        int(known_liquidity.average_daily_volume * rules.participation_rate)
        - known_liquidity.consumed_quantity,
    )
    quantity = min(order.quantity, capacity // rules.lot_size * rules.lot_size)
    if rules.allow_odd_lot_sell and order.side is Side.SELL and order.quantity <= capacity:
        quantity = order.quantity
    if quantity <= 0:
        return no_fill("liquidity_below_lot")
    direction = Decimal("1") if order.side is Side.BUY else Decimal("-1")
    price = money(bar.open * (1 + direction * rules.slippage_bps / BPS_DIVISOR))
    if bar.limit_up is not None:
        price = min(price, bar.limit_up)
    if bar.limit_down is not None:
        price = max(price, bar.limit_down)
    if price <= 0:
        return no_fill("invalid_price")
    if (
        order.order_type == "limit"
        and order.limit_price is not None
        and (
            (order.side is Side.BUY and price > order.limit_price)
            or (order.side is Side.SELL and price < order.limit_price)
        )
    ):
        return no_fill("limit_not_executable_at_open")
    fill = Fill(order.security_id, order.side, quantity, price, bar.trade_date)
    return MatchResult(
        "filled" if quantity == order.quantity else "partially_filled",
        fill,
        None if quantity == order.quantity else "liquidity_capacity",
    )
