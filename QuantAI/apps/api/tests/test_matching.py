from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal as D

import pytest

from tests.helpers import bars_for_prices


def test_legacy_misidentified_beijing_security_cannot_bypass_execution_guard():
    from portfolio_client.domain.errors import DomainError
    from portfolio_client.domain.execution.rules import lot_size_for
    from portfolio_client.domain.market.models import Security

    security = Security("XSHG:920005", "920005", "XSHG", "Imported", None, "i", "main")
    with pytest.raises(DomainError) as error:
        lot_size_for(security)
    assert error.value.code == "UNSUPPORTED_MARKET_RULES"


def inputs():
    from portfolio_client.domain.accounting.models import Side
    from portfolio_client.domain.execution.models import KnownLiquidity, Order, OrderState
    from portfolio_client.domain.execution.rules import ExecutionRules

    order = Order(
        "o1",
        "a1",
        "XSHE:000001",
        Side.BUY,
        1000,
        "market",
        None,
        datetime(2026, 1, 1, 7, tzinfo=UTC),
        date(2026, 1, 2),
        OrderState.SUBMITTED,
    )
    bar = replace(bars_for_prices([D("10"), D("10")])[1], limit_up=D("11"), limit_down=D("9"))
    return (
        order,
        bar,
        KnownLiquidity(D("2000"), order.submitted_at),
        ExecutionRules(slippage_bps=D("0")),
    )


def test_opening_capacity_depends_on_previously_known_adv_not_final_day_volume():
    from portfolio_client.domain.execution.matching import match_order

    order, bar, known, rules = inputs()
    thin = match_order(order, replace(bar, volume=100), known, rules)
    huge = match_order(order, replace(bar, volume=100_000_000), known, rules)
    assert thin.fill.quantity == huge.fill.quantity == 100
    assert thin.status == "partially_filled"


def test_suspension_and_open_at_up_limit_block_buy_without_peeking_at_day_range():
    from portfolio_client.domain.execution.matching import match_order

    order, bar, known, rules = inputs()
    assert match_order(order, replace(bar, suspended=True), known, rules).reason == "suspended"
    result = match_order(order, replace(bar, open=D("11"), high=D("11"), low=D("10")), known, rules)
    assert result.fill is None and result.reason == "limit_up_at_open"


def test_limit_order_does_not_invent_intraday_queue_fill():
    from portfolio_client.domain.execution.matching import match_order

    order, bar, known, rules = inputs()
    result = match_order(
        replace(order, order_type="limit", limit_price=D("9.5")),
        replace(bar, low=D("9")),
        known,
        rules,
    )
    assert result.fill is None and result.reason == "limit_not_executable_at_open"


def test_future_liquidity_or_ineligible_session_is_rejected():
    from portfolio_client.domain.execution.matching import match_order
    from portfolio_client.domain.execution.models import KnownLiquidity

    order, bar, known, rules = inputs()
    assert (
        match_order(order, bar, KnownLiquidity(D("2000"), bar.available_at), rules).reason
        == "future_liquidity"
    )
    assert (
        match_order(replace(order, eligible_session=date(2026, 1, 3)), bar, known, rules).reason
        == "not_eligible"
    )
