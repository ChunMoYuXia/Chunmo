from datetime import date, timedelta
from decimal import Decimal as D

import numpy as np
import pytest

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import Security
from portfolio_client.domain.research.backtest_models import NavPoint


def points(values):
    return [
        NavPoint(date(2024, 1, 2) + timedelta(days=index), D(str(value)), D("0"), D("0"), ())
        for index, value in enumerate(values)
    ]


def test_drawdown_and_empty_history():
    from portfolio_client.domain.research.metrics import max_drawdown

    assert max_drawdown([100.0, 120.0, 90.0, 110.0]) == pytest.approx(0.25)
    assert max_drawdown([]) is None


def test_performance_metrics_match_independent_four_point_baseline():
    from portfolio_client.domain.research.metrics import compute_metrics

    result = compute_metrics(points([100, 120, 90, 110]), None, 244, 0.03)
    assert result.total_return.value == pytest.approx(0.1)
    assert result.max_drawdown.value == pytest.approx(0.25)
    # Independent statistics.stdev + math.sqrt baseline, checked before implementation.
    assert result.annualized_volatility.value == pytest.approx(4.162150639079132)
    assert result.sharpe.value == pytest.approx(3.358323142713201)
    assert result.total_return.unit == "ratio"
    assert result.total_return.formula_version == "metrics-v1"
    assert result.total_return.as_of.date() == date(2024, 1, 5)


def test_constant_prices_have_defined_zero_volatility_but_no_sharpe():
    from portfolio_client.domain.research.metrics import compute_metrics

    result = compute_metrics(points([100, 100, 100]), None, 244, 0)
    assert result.annualized_volatility.value == 0
    assert result.sharpe.value is None and result.sharpe.reason == "zero_volatility"
    empty = compute_metrics([], None, 244, 0)
    assert empty.total_return.value is None and empty.total_return.reason == "empty_history"


@pytest.mark.parametrize("values", [[100, 0], [100, float("nan")], [100, -1]])
def test_invalid_nav_is_rejected(values):
    from portfolio_client.domain.research.metrics import max_drawdown

    with pytest.raises(DomainError) as error:
        max_drawdown(values)
    assert error.value.code == "INVALID_NAV"


def test_missing_benchmark_date_does_not_silently_change_comparison_period():
    from portfolio_client.domain.research.metrics import compute_metrics

    nav = points([100, 120, 90, 110])
    benchmark = points([100, 110, 120, 130])
    complete = compute_metrics(nav, benchmark, 244, 0)
    assert complete.benchmark_return.value == pytest.approx(0.3)
    assert complete.excess_return.value == pytest.approx(-0.2)
    incomplete = compute_metrics(nav, [benchmark[0], *benchmark[2:]], 244, 0)
    assert incomplete.benchmark_return.value is None
    assert incomplete.benchmark_return.reason == "benchmark_missing_sessions"


def test_tail_loss_uses_higher_quantile_and_reports_positive_loss():
    from portfolio_client.domain.research.metrics import historical_tail_risk

    result = historical_tail_risk(np.array([-0.04, -0.02, 0.0, 0.01, 0.03]), 0.75)
    assert result.var.value == pytest.approx(0.02)
    assert result.cvar.value == pytest.approx(0.03)
    assert result.quantile_method == "higher"
    assert result.warnings
    assert historical_tail_risk(np.array([])).var.value is None


def test_exposures_include_cash_and_preserve_unknown_industry():
    from portfolio_client.domain.research.exposures import exposures

    identities = [Security("XSHE:000001", "000001", "XSHE", "测试", None, "cs")]
    result = exposures({"XSHE:000001": 100}, {"XSHE:000001": D("10")}, identities, cash=D("1000"))
    assert result.total_value == D("2000")
    assert result.asset_weights == {"XSHE:000001": 0.5}
    assert result.industry_weights == {"未分类": 0.5}
    assert result.cash_weight == 0.5 and result.max_asset_weight == 0.5
    assert result.stock_hhi == 0.25
    assert result.warnings
    assert exposures({}, {}, [], cash=D("1000")).cash_weight == 1
