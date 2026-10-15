from datetime import date
from decimal import Decimal as D

import pytest

from portfolio_client.domain.errors import DomainError
from tests.helpers import schedule


def buy(state, day=date(2024, 1, 2)):
    from portfolio_client.domain.accounting.costs import calculate_fees
    from portfolio_client.domain.accounting.ledger import apply_fill
    from portfolio_client.domain.accounting.models import Fill, Side

    return apply_fill(
        state,
        Fill("XSHE:000001", Side.BUY, 100, D("10"), day),
        calculate_fees(D("1000"), Side.BUY, schedule()),
    )


def test_cash_cost_and_t_plus_one_match_hand_calculated_ledger():
    from portfolio_client.domain.accounting.costs import calculate_fees
    from portfolio_client.domain.accounting.ledger import advance_session, apply_fill
    from portfolio_client.domain.accounting.models import AccountState, Fill, Side

    start = AccountState(cash=D("2000"), withdrawable_cash=D("2000"))
    bought = buy(start)
    assert start.cash == D("2000") and not start.positions
    assert bought.cash == D("994.99")
    assert bought.positions["XSHE:000001"].cost_basis == D("1005.01")
    assert bought.positions["XSHE:000001"].sellable == 0
    fees = calculate_fees(D("1000"), Side.SELL, schedule())
    with pytest.raises(DomainError) as error:
        apply_fill(bought, Fill("XSHE:000001", Side.SELL, 100, D("10"), date(2024, 1, 2)), fees)
    assert error.value.code == "INSUFFICIENT_SELLABLE"
    sold = apply_fill(
        advance_session(bought, date(2024, 1, 3)),
        Fill("XSHE:000001", Side.SELL, 100, D("10"), date(2024, 1, 3)),
        fees,
    )
    assert sold.cash == D("1989.48")
    assert sold.withdrawable_cash == D("994.99")
    assert sold.events[-1].realized_pnl == D("-10.52")
    assert advance_session(sold, date(2024, 1, 4)).withdrawable_cash == D("1989.48")


def test_frozen_cash_is_not_available_for_an_unrelated_fill():
    from portfolio_client.domain.accounting.ledger import release_cash, reserve_cash
    from portfolio_client.domain.accounting.models import AccountState

    start = AccountState(cash=D("2000"), withdrawable_cash=D("2000"))
    reserved = reserve_cash(start, D("1500"))
    assert reserved.available_cash == D("500")
    with pytest.raises(DomainError) as error:
        buy(reserved)
    assert error.value.code == "INSUFFICIENT_CASH"
    assert buy(release_cash(reserved, D("1500"))).cash == D("994.99")


def test_missing_price_is_not_a_fictitious_liquidation():
    from portfolio_client.domain.accounting.ledger import value_account
    from portfolio_client.domain.accounting.models import AccountState

    held = buy(AccountState(cash=D("2000"), withdrawable_cash=D("2000")))
    with pytest.raises(DomainError) as error:
        value_account(held, {})
    assert error.value.code == "PRICE_NOT_FOUND"
    assert held.positions["XSHE:000001"].quantity == 100


def test_explicit_same_session_does_not_unlock_newly_bought_stock():
    from portfolio_client.domain.accounting.ledger import advance_session
    from portfolio_client.domain.accounting.models import AccountState

    held = buy(AccountState(cash=D("2000"), withdrawable_cash=D("2000")))
    assert advance_session(held, date(2024, 1, 2)).positions["XSHE:000001"].sellable == 0
