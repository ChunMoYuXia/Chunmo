from dataclasses import dataclass, replace

from portfolio_client.domain.accounting.costs import calculate_fees, money
from portfolio_client.domain.accounting.ledger import apply_fill
from portfolio_client.domain.accounting.models import AccountState, FeeBreakdown, FeeSchedule, Side
from portfolio_client.domain.execution.matching import match_order
from portfolio_client.domain.execution.models import KnownLiquidity, MatchResult, Order
from portfolio_client.domain.execution.rules import ExecutionRules
from portfolio_client.domain.market.models import DailyBar


@dataclass(frozen=True)
class ExecutionOutcome:
    account: AccountState
    match: MatchResult
    fees: FeeBreakdown | None


def execute_order(
    state: AccountState,
    order: Order,
    bar: DailyBar,
    liquidity: KnownLiquidity,
    rules: ExecutionRules,
    schedule: FeeSchedule,
) -> ExecutionOutcome:
    result = match_order(order, bar, liquidity, rules)
    fill = result.fill
    if fill is None:
        return ExecutionOutcome(state, result, None)
    if fill.side is Side.BUY:
        low, high = 0, fill.quantity // rules.lot_size
        while low < high:
            mid = (low + high + 1) // 2
            notional = money(mid * rules.lot_size * fill.price)
            if (
                notional + calculate_fees(notional, Side.BUY, schedule).total
                <= state.available_cash
            ):
                low = mid
            else:
                high = mid - 1
        quantity = low * rules.lot_size
        if quantity == 0:
            return ExecutionOutcome(state, MatchResult("unfilled", None, "insufficient_cash"), None)
        if quantity < fill.quantity:
            fill = replace(fill, quantity=quantity)
            result = MatchResult("partially_filled", fill, "cash_capacity")
    fees = calculate_fees(money(fill.quantity * fill.price), fill.side, schedule)
    return ExecutionOutcome(apply_fill(state, fill, fees), result, fees)
