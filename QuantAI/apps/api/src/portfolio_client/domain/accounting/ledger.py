from dataclasses import replace
from datetime import date
from decimal import Decimal

from portfolio_client.domain.accounting.costs import money
from portfolio_client.domain.accounting.models import (
    AccountState,
    FeeBreakdown,
    Fill,
    LedgerEvent,
    Position,
    Side,
)
from portfolio_client.domain.errors import DomainError


def append_event(
    state: AccountState,
    kind: str,
    *,
    security_id: str | None = None,
    delta_cash: Decimal = Decimal("0"),
    delta_frozen: Decimal = Decimal("0"),
    delta_receivables: Decimal = Decimal("0"),
    delta_quantity: int = 0,
    fees: FeeBreakdown | None = None,
    realized_pnl: Decimal | None = None,
    description: str = "",
) -> AccountState:
    event = LedgerEvent(
        state.ledger_offset + len(state.events) + 1,
        state.session,
        kind,
        security_id,
        delta_cash,
        delta_frozen,
        delta_receivables,
        delta_quantity,
        fees,
        realized_pnl,
        description,
    )
    return replace(state, events=(*state.events, event))


def advance_session(state: AccountState, session: date) -> AccountState:
    if state.session is not None and session < state.session:
        raise DomainError("SESSION_REWIND", "账本不可倒退到过去交易日")
    if state.session == session:
        return state
    positions = {
        key: replace(position, sellable=position.quantity)
        for key, position in state.positions.items()
    }
    return replace(state, positions=positions, session=session, withdrawable_cash=state.cash)


def reserve_cash(state: AccountState, amount: Decimal) -> AccountState:
    if not amount.is_finite() or amount < 0 or amount > state.available_cash:
        raise DomainError("INSUFFICIENT_CASH", "可用资金不足以冻结订单预留款")
    return append_event(
        replace(state, frozen_cash=state.frozen_cash + amount), "reserve", delta_frozen=amount
    )


def release_cash(state: AccountState, amount: Decimal) -> AccountState:
    if not amount.is_finite() or amount < 0 or amount > state.frozen_cash:
        raise DomainError("INVALID_RESERVATION", "释放金额超过已冻结资金")
    return append_event(
        replace(state, frozen_cash=state.frozen_cash - amount), "release", delta_frozen=-amount
    )


def apply_fill(state: AccountState, fill: Fill, fees: FeeBreakdown) -> AccountState:
    state = advance_session(state, fill.session)
    if any(not fee.is_finite() or fee < 0 for fee in (fees.commission, fees.transfer, fees.stamp)):
        raise DomainError("INVALID_FEES", "成交费用不合法")
    notional = money(fill.price * fill.quantity)
    positions = state.positions.copy()
    old = positions.get(fill.security_id, Position(0, 0, Decimal("0")))
    realized = None
    if fill.side is Side.BUY:
        expense = notional + fees.total
        if expense > state.available_cash:
            raise DomainError("INSUFFICIENT_CASH", "成交金额及费用超过可用资金")
        delta_cash, delta_quantity = -expense, fill.quantity
        positions[fill.security_id] = Position(
            old.quantity + fill.quantity, old.sellable, old.cost_basis + expense
        )
    else:
        if fill.quantity > old.sellable:
            raise DomainError("INSUFFICIENT_SELLABLE", "卖出数量超过T+1可卖持仓")
        delta_cash, delta_quantity = notional - fees.total, -fill.quantity
        removed_cost = old.cost_basis * fill.quantity / old.quantity
        realized = delta_cash - removed_cost
        remaining = old.quantity - fill.quantity
        if remaining:
            positions[fill.security_id] = Position(
                remaining, old.sellable - fill.quantity, old.cost_basis - removed_cost
            )
        else:
            positions.pop(fill.security_id)
    cash = state.cash + delta_cash
    if cash < state.frozen_cash:
        raise DomainError("INSUFFICIENT_CASH", "成交费用使现金低于冻结金额")
    updated = replace(
        state, cash=cash, positions=positions, withdrawable_cash=min(cash, state.withdrawable_cash)
    )
    return append_event(
        updated,
        "fill",
        security_id=fill.security_id,
        delta_cash=delta_cash,
        delta_quantity=delta_quantity,
        fees=fees,
        realized_pnl=realized,
    )


def value_account(state: AccountState, prices: dict[str, Decimal]) -> Decimal:
    quantities = {key: position.quantity for key, position in state.positions.items()}
    for pending in state.pending_shares.values():
        quantities[pending.security_id] = quantities.get(pending.security_id, 0) + pending.quantity
    total = state.cash + state.receivables
    for security_id, quantity in quantities.items():
        price = prices.get(security_id)
        if price is None or not price.is_finite() or price <= 0:
            raise DomainError(
                "PRICE_NOT_FOUND", "持仓缺少有效原始估值价格", {"security_id": security_id}
            )
        total += quantity * price
    return total
