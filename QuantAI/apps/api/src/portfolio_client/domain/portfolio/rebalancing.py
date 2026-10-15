from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from portfolio_client.domain.accounting.costs import calculate_fees, money
from portfolio_client.domain.accounting.models import FeeBreakdown, FeeSchedule, Side
from portfolio_client.domain.errors import DomainError


@dataclass(frozen=True)
class PlannedTrade:
    security_id: str
    side: Side
    quantity: int
    reference_price: Decimal
    fees: FeeBreakdown


@dataclass(frozen=True)
class RebalancePlan:
    orders: tuple[PlannedTrade, ...]
    cash_after: Decimal
    estimated_fees: Decimal
    equity_before: Decimal
    assumptions: tuple[str, ...] = (
        "使用参考价格预留费用；实际成交价变化需再次验资。",
        "未提供sellable时假设导入持仓已满足T+1。",
    )


def plan_rebalance(
    target_weights: dict[str, Decimal],
    holdings: dict[str, int],
    prices: dict[str, Decimal],
    cash: Decimal,
    lot_sizes: dict[str, int],
    fee_schedule: FeeSchedule,
    *,
    sellable: dict[str, int] | None = None,
    receivables: Decimal = Decimal("0"),
    pending_holdings: dict[str, int] | None = None,
) -> RebalancePlan:
    pending_holdings = pending_holdings or {}
    if (
        any(not value.is_finite() or value < 0 or value > 1 for value in target_weights.values())
        or sum(target_weights.values()) > 1
    ):
        raise DomainError("INVALID_WEIGHTS", "目标权重必须非负且总和不超过1")
    if (
        not cash.is_finite()
        or cash < 0
        or any(type(q) is not int or q < 0 for q in holdings.values())
        or not receivables.is_finite()
        or receivables < 0
        or any(type(q) is not int or q < 0 for q in pending_holdings.values())
    ):
        raise DomainError("INVALID_ACCOUNT", "调仓资金及持仓不合法")
    securities = sorted(set(target_weights) | set(holdings) | set(pending_holdings))
    for key in securities:
        if key not in prices or not prices[key].is_finite() or prices[key] <= 0:
            raise DomainError("PRICE_NOT_FOUND", "调仓标的缺少原始参考价格", {"security_id": key})
        if key not in lot_sizes or type(lot_sizes[key]) is not int or lot_sizes[key] <= 0:
            raise DomainError("INVALID_LOT", "缺少有效买卖单位", {"security_id": key})
    equity = (
        cash
        + receivables
        + sum(
            (holdings.get(key, 0) + pending_holdings.get(key, 0)) * prices[key]
            for key in securities
        )
    )
    targets = {
        key: int(
            (
                equity * target_weights.get(key, Decimal("0")) / prices[key] / lot_sizes[key]
            ).to_integral_value(rounding=ROUND_DOWN)
        )
        * lot_sizes[key]
        for key in securities
    }
    targets = {
        key: max(0, quantity - pending_holdings.get(key, 0)) for key, quantity in targets.items()
    }
    orders: list[PlannedTrade] = []
    # Sells first release cash; complete liquidation may include an imported odd lot.
    for key in securities:
        quantity = max(0, holdings.get(key, 0) - targets[key])
        if not quantity:
            continue
        if quantity > (sellable if sellable is not None else holdings).get(key, 0):
            raise DomainError("INSUFFICIENT_SELLABLE", "调仓卖出超过可卖持仓", {"security_id": key})
        if quantity != holdings[key]:
            quantity = quantity // lot_sizes[key] * lot_sizes[key]
        if not quantity:
            continue
        notional = money(quantity * prices[key])
        fees = calculate_fees(notional, Side.SELL, fee_schedule)
        cash += notional - fees.total
        if cash < 0:
            raise DomainError("INSUFFICIENT_CASH", "卖出费用超过现金及所得")
        orders.append(PlannedTrade(key, Side.SELL, quantity, prices[key], fees))
    reserve = equity * (1 - sum(target_weights.values()))
    for key in securities:
        wanted_lots = max(0, targets[key] - holdings.get(key, 0)) // lot_sizes[key]
        low, high = 0, wanted_lots
        while low < high:
            mid = (low + high + 1) // 2
            notional = money(mid * lot_sizes[key] * prices[key])
            if notional + calculate_fees(notional, Side.BUY, fee_schedule).total <= cash - reserve:
                low = mid
            else:
                high = mid - 1
        if low:
            quantity = low * lot_sizes[key]
            notional = money(quantity * prices[key])
            fees = calculate_fees(notional, Side.BUY, fee_schedule)
            cash -= notional + fees.total
            orders.append(PlannedTrade(key, Side.BUY, quantity, prices[key], fees))
    return RebalancePlan(
        tuple(orders), cash, sum((order.fees.total for order in orders), Decimal("0")), equity
    )
