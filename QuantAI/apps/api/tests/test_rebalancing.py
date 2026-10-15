from decimal import Decimal as D

import pytest

from portfolio_client.domain.errors import DomainError
from tests.helpers import schedule


def test_whole_lot_plan_reserves_minimum_commission_before_buy():
    from portfolio_client.domain.portfolio.rebalancing import plan_rebalance

    result = plan_rebalance(
        {"XSHE:000001": D("1")},
        {},
        {"XSHE:000001": D("10")},
        D("2000"),
        {"XSHE:000001": 100},
        schedule(),
    )
    assert result.orders[0].quantity == 100
    assert result.estimated_fees == D("5.01")
    assert result.cash_after == D("994.99")


def test_missing_price_never_creates_an_order():
    from portfolio_client.domain.portfolio.rebalancing import plan_rebalance

    with pytest.raises(DomainError) as error:
        plan_rebalance(
            {"XSHE:000001": D("1")}, {}, {}, D("10000"), {"XSHE:000001": 100}, schedule()
        )
    assert error.value.code == "PRICE_NOT_FOUND"


def test_rebalance_rejects_selling_more_than_available_t_plus_one_quantity():
    from portfolio_client.domain.portfolio.rebalancing import plan_rebalance

    with pytest.raises(DomainError) as error:
        plan_rebalance(
            {"XSHE:000001": D("0")},
            {"XSHE:000001": 100},
            {"XSHE:000001": D("10")},
            D("100"),
            {"XSHE:000001": 100},
            schedule(),
            sellable={"XSHE:000001": 0},
        )
    assert error.value.code == "INSUFFICIENT_SELLABLE"


def test_receivables_and_undelivered_shares_count_in_target_nav_but_cannot_fund_buys():
    from portfolio_client.domain.portfolio.rebalancing import plan_rebalance

    result = plan_rebalance(
        {"XSHE:000001": D("1")},
        {"XSHE:000001": 100},
        {"XSHE:000001": D("10")},
        D("0"),
        {"XSHE:000001": 100},
        schedule(),
        receivables=D("50"),
        pending_holdings={"XSHE:000001": 10},
    )
    assert result.equity_before == D("1150")
    assert result.orders == () and result.cash_after == 0
