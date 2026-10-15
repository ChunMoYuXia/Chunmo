from datetime import date
from decimal import Decimal as D

import pytest

from portfolio_client.domain.errors import DomainError
from tests.helpers import schedule


def test_stock_fees_buy_and_sell():
    from portfolio_client.domain.accounting.costs import calculate_fees
    from portfolio_client.domain.accounting.models import Side

    buy = calculate_fees(D("100000"), Side.BUY, schedule())
    sell = calculate_fees(D("100000"), Side.SELL, schedule())
    assert (buy.commission, buy.transfer, buy.stamp) == (D("25.00"), D("1.00"), D("0.00"))
    assert sell.stamp == D("50.00")
    assert sell.total == D("76.00")


def test_minimum_commission_and_independent_item_rounding():
    from portfolio_client.domain.accounting.costs import calculate_fees
    from portfolio_client.domain.accounting.models import Side

    fees = calculate_fees(D("1000"), Side.BUY, schedule())
    assert fees.total == D("5.01")


def test_effective_fee_change_and_unknown_period_are_explicit():
    from portfolio_client.domain.accounting.costs import default_fee_schedules, schedule_for

    assert schedule_for(default_fee_schedules(), date(2023, 8, 27)).stamp_bps == D("10")
    assert schedule_for(default_fee_schedules(), date(2023, 8, 28)).stamp_bps == D("5")
    with pytest.raises(DomainError, match="费率"):
        schedule_for(default_fee_schedules(), date(2020, 1, 1))


def test_unsupported_money_magnitude_has_domain_error_instead_of_decimal_crash():
    from portfolio_client.domain.accounting.costs import calculate_fees
    from portfolio_client.domain.accounting.models import Side

    with pytest.raises(DomainError) as error:
        calculate_fees(D("1e100"), Side.BUY, schedule())
    assert error.value.code == "INVALID_MONEY_PRECISION"
