from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal as D

from portfolio_client.domain.market.models import CorporateActionRecord


def action(kind="dividend"):
    return CorporateActionRecord(
        "div-1",
        "XSHE:000001",
        kind,
        date(2024, 1, 2),
        date(2024, 1, 3),
        date(2024, 1, 5),
        datetime(2024, 1, 1, tzinfo=UTC),
        D(".5"),
        D(".1"),
    )


def held():
    from portfolio_client.domain.accounting.models import AccountState, Position

    return AccountState(
        cash=D("0"),
        withdrawable_cash=D("0"),
        positions={"XSHE:000001": Position(100, 100, D("1000"))},
    )


def test_dividend_entitlement_survives_sale_and_is_cash_only_on_payment():
    from portfolio_client.domain.accounting.corporate_actions import apply_corporate_action
    from portfolio_client.domain.accounting.ledger import value_account

    record = apply_corporate_action(held(), action(), date(2024, 1, 2))
    sold_after_record = replace(record, positions={})
    ex = apply_corporate_action(sold_after_record, action(), date(2024, 1, 3))
    assert ex.receivables == D("50.00") and ex.available_cash == 0
    assert value_account(ex, {}) == D("50.00")
    assert apply_corporate_action(ex, action(), date(2024, 1, 3)) == ex
    paid = apply_corporate_action(ex, action(), date(2024, 1, 5))
    assert paid.cash == D("50.00") and paid.receivables == 0
    assert apply_corporate_action(paid, action(), date(2024, 1, 5)) == paid


def test_buyer_after_record_date_does_not_receive_dividend():
    from portfolio_client.domain.accounting.corporate_actions import apply_corporate_action

    assert apply_corporate_action(held(), action(), date(2024, 1, 3)).receivables == 0


def test_bonus_shares_valued_from_ex_date_and_delivered_on_declared_payment_date():
    from portfolio_client.domain.accounting.corporate_actions import apply_corporate_action
    from portfolio_client.domain.accounting.ledger import value_account

    record = apply_corporate_action(held(), action("bonus"), date(2024, 1, 2))
    ex = apply_corporate_action(record, action("bonus"), date(2024, 1, 3))
    assert ex.positions["XSHE:000001"].quantity == 100
    assert value_account(ex, {"XSHE:000001": D("10")}) == D("1100")
    paid = apply_corporate_action(ex, action("bonus"), date(2024, 1, 5))
    assert paid.positions["XSHE:000001"].quantity == 110
    assert paid.positions["XSHE:000001"].cost_basis == D("1000")
