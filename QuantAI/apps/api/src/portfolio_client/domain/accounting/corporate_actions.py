from dataclasses import replace
from datetime import date
from decimal import ROUND_DOWN, Decimal

from portfolio_client.domain.accounting.costs import money
from portfolio_client.domain.accounting.ledger import advance_session, append_event
from portfolio_client.domain.accounting.models import (
    AccountState,
    DividendReceivable,
    PendingShares,
    Position,
)
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import CorporateActionRecord


def apply_corporate_action(
    state: AccountState, action: CorporateActionRecord, session: date
) -> AccountState:
    if not action.record_date <= action.ex_date <= action.pay_date:
        raise DomainError("INVALID_CORPORATE_ACTION", "登记、除权、支付日期顺序无效")
    if (
        not action.cash_per_share.is_finite()
        or action.cash_per_share < 0
        or not action.bonus_ratio.is_finite()
        or action.bonus_ratio < 0
    ):
        raise DomainError("INVALID_CORPORATE_ACTION", "公司行为金额及比例不合法")
    state = advance_session(state, session)
    if session == action.record_date and action.event_id not in state.entitlements:
        entitlements = {
            **state.entitlements,
            action.event_id: state.positions.get(
                action.security_id, Position(0, 0, Decimal("0"))
            ).quantity,
        }
        state = replace(state, entitlements=entitlements)
    marker = f"ex:{action.event_id}"
    if session == action.ex_date and marker not in state.processed_actions:
        quantity = state.entitlements.get(action.event_id, 0)
        state = replace(state, processed_actions=(*state.processed_actions, marker))
        if action.kind == "dividend":
            amount = money(action.cash_per_share * quantity)
            if amount:
                state = replace(
                    state,
                    receivables=state.receivables + amount,
                    dividends={
                        **state.dividends,
                        action.event_id: DividendReceivable(
                            action.security_id, amount, action.pay_date
                        ),
                    },
                )
                state = append_event(
                    state,
                    "dividend_receivable",
                    security_id=action.security_id,
                    delta_receivables=amount,
                    description="税前现金股利，M0不模拟持有期税补扣",
                )
        elif action.kind == "bonus":
            bonus = int((quantity * action.bonus_ratio).to_integral_value(rounding=ROUND_DOWN))
            if bonus:
                positions = state.positions.copy()
                current = positions.get(action.security_id, Position(0, 0, Decimal("0")))
                assigned_cost = current.cost_basis * bonus / (current.quantity + bonus)
                if current.quantity:
                    positions[action.security_id] = replace(
                        current, cost_basis=current.cost_basis - assigned_cost
                    )
                state = replace(
                    state,
                    positions=positions,
                    pending_shares={
                        **state.pending_shares,
                        action.event_id: PendingShares(
                            action.security_id, bonus, assigned_cost, action.pay_date
                        ),
                    },
                )
                state = append_event(
                    state,
                    "bonus_receivable",
                    security_id=action.security_id,
                    description=f"应收送转{bonus}股；碎股向下取整，上市日按导入pay_date",
                )
    if session >= action.pay_date:
        dividend = state.dividends.get(action.event_id)
        if dividend is not None:
            dividends = state.dividends.copy()
            dividends.pop(action.event_id)
            state = append_event(
                replace(
                    state,
                    dividends=dividends,
                    cash=state.cash + dividend.amount,
                    withdrawable_cash=state.withdrawable_cash + dividend.amount,
                    receivables=state.receivables - dividend.amount,
                ),
                "dividend_payment",
                security_id=action.security_id,
                delta_cash=dividend.amount,
                delta_receivables=-dividend.amount,
            )
        pending = state.pending_shares.get(action.event_id)
        if pending is not None:
            pending_shares = state.pending_shares.copy()
            pending_shares.pop(action.event_id)
            positions = state.positions.copy()
            old = positions.get(action.security_id, Position(0, 0, Decimal("0")))
            positions[action.security_id] = Position(
                old.quantity + pending.quantity,
                old.sellable + pending.quantity,
                old.cost_basis + pending.cost_basis,
            )
            state = append_event(
                replace(state, positions=positions, pending_shares=pending_shares),
                "bonus_delivery",
                security_id=action.security_id,
                delta_quantity=pending.quantity,
            )
    return state
