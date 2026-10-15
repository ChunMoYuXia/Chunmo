from dataclasses import replace
from datetime import date, timedelta
from decimal import Decimal as D

import pytest

from portfolio_client.domain.errors import DomainError
from tests.market_helpers import MemoryMarket


def request(market, frequency="daily", start=None, zero_fees=False):
    from portfolio_client.domain.accounting.models import FeeSchedule
    from portfolio_client.domain.research.backtest import BacktestRequest

    options = (
        {
            "fee_schedules": (
                FeeSchedule("zero", date(2022, 1, 1), None, D("0"), D("0"), D("0"), D("0")),
            ),
            "fee_version": "zero",
        }
        if zero_fees
        else {}
    )
    return BacktestRequest(
        dataset_id="test",
        calendar_id="test-calendar",
        security_ids=["XSHE:000001"],
        weights={"XSHE:000001": D("1")},
        frequency=frequency,
        start=start or market.sessions[0],
        end=market.sessions[-1],
        capital=D("10000"),
        slippage_bps=D("0"),
        seed=42,
        **options,
    )


def test_year_end_signal_executes_on_next_actual_session():
    from portfolio_client.domain.research.backtest import run_backtest

    sessions = [date(2025, 12, 10) + timedelta(days=i) for i in range(22)] + [date(2026, 1, 5)]
    market = MemoryMarket(sessions)
    assert market.calendar("test").next_session(date(2025, 12, 31)) == date(2026, 1, 5)
    result = run_backtest(request(market, "yearly"), market, lambda _: None)
    assert result.signals[0].session == date(2025, 12, 31)
    assert result.trades[0].fill.session == date(2026, 1, 5)
    assert result.trades[0].fill.quantity == 900


def test_no_cost_single_asset_matches_independent_price_return_under_exact_lots():
    from portfolio_client.domain.research.backtest import run_backtest

    sessions = [date(2024, 1, 2) + timedelta(days=i) for i in range(22)]
    market = MemoryMarket(sessions, [D("10")] * 21 + [D("11")])
    spec = request(market, start=sessions[20], zero_fees=True).model_copy(
        update={"capital": D("1000")}
    )
    result = run_backtest(spec, market, lambda _: None)
    assert [point.nav for point in result.nav] == [D("1000"), D("1100")]
    assert result.costs == 0


def test_future_prices_do_not_rewrite_past_signal_and_same_input_is_reproducible():
    from portfolio_client.domain.research.backtest import run_backtest

    sessions = [date(2024, 1, 2) + timedelta(days=i) for i in range(23)]
    market = MemoryMarket(sessions)
    spec = request(market)
    original = run_backtest(spec, market, lambda _: None)
    assert run_backtest(spec, market, lambda _: None) == original
    market.history[-1] = replace(market.history[-1], open=D("12"), high=D("12"), close=D("12"))
    changed = run_backtest(spec, market, lambda _: None)
    assert changed.signals[0] == original.signals[0]
    assert changed.nav[:-1] == original.nav[:-1]


def test_incomplete_warmup_is_an_error_not_an_all_cash_success():
    from portfolio_client.domain.research.backtest import run_backtest

    market = MemoryMarket([date(2024, 1, 2) + timedelta(days=i) for i in range(10)])
    with pytest.raises(DomainError) as error:
        run_backtest(request(market), market, lambda _: None)
    assert error.value.code == "INSUFFICIENT_HISTORY"


def test_slippage_is_reported_separately_and_charged_only_in_fill_price():
    from portfolio_client.domain.research.backtest import run_backtest

    sessions = [date(2024, 1, 2) + timedelta(days=i) for i in range(22)]
    market = MemoryMarket(sessions)
    spec = request(market, start=sessions[20]).model_copy(
        update={"capital": D("2000"), "slippage_bps": D("10")}
    )
    result = run_backtest(spec, market, lambda _: None)
    assert result.trades[0].fill.price == D("10.01")
    assert result.costs == D("5.01")
    assert result.slippage_costs == D("1.00")
    assert result.final_account.cash == D("993.99")
    assert result.nav[-1].nav == D("1993.99")


@pytest.mark.parametrize(
    "frequency,day,next_day,expected",
    [
        ("weekly", date(2025, 12, 31), date(2026, 1, 2), False),
        ("weekly", date(2026, 1, 2), date(2026, 1, 5), True),
        ("monthly", date(2025, 12, 31), date(2026, 1, 2), True),
        ("yearly", date(2025, 12, 30), date(2025, 12, 31), False),
    ],
)
def test_period_boundary_uses_next_trading_session(frequency, day, next_day, expected):
    from portfolio_client.domain.research.backtest import is_rebalance_session

    assert is_rebalance_session(day, next_day, frequency) is expected
