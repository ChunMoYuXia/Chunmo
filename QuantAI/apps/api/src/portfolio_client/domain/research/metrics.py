import math
from datetime import datetime

import numpy as np
from numpy.typing import NDArray

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.research.backtest_models import NavPoint
from portfolio_client.domain.research.metric_models import Metric, PerformanceMetrics, TailRisk


def metric(
    value: float | None, unit: str, source: str, as_of: datetime | None, reason: str | None = None
) -> Metric:
    if value is not None and not math.isfinite(value):
        value, reason = None, "non_finite_result"
    return Metric(value, unit, "metrics-v1", source, as_of, reason)


def _positive_values(values: list[float]) -> NDArray[np.float64]:
    data = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(data)) or np.any(data <= 0):
        raise DomainError("INVALID_NAV", "净值必须为有限正数")
    return data


def max_drawdown(nav: list[float]) -> float | None:
    if not nav:
        return None
    values = _positive_values(nav)
    return float(np.max(1 - values / np.maximum.accumulate(values)))


def _validate_points(points: list[NavPoint]) -> NDArray[np.float64]:
    days = [point.session for point in points]
    if days != sorted(set(days)):
        raise DomainError("INVALID_NAV_DATES", "净值日期必须有序且唯一")
    return _positive_values([float(point.nav) for point in points])


def compute_metrics(
    nav: list[NavPoint],
    benchmark: list[NavPoint] | None,
    annualization: int = 244,
    risk_free_annual: float = 0,
    *,
    benchmark_basis: str = "price",
) -> PerformanceMetrics:
    if (
        not 1 <= annualization <= 1000
        or not math.isfinite(risk_free_annual)
        or risk_free_annual <= -1
    ):
        raise DomainError("INVALID_METRIC_PARAMETERS", "年化因子或无风险年利率不合法")
    values = _validate_points(nav)
    cutoff = at_close(nav[-1].session) if nav else None
    source = "daily_nav"

    def measure(value: float | None, unit: str = "ratio", reason: str | None = None) -> Metric:
        return metric(value, unit, source, cutoff, reason)

    reason = "empty_history" if not nav else "insufficient_history"
    observations = max(0, len(nav) - 1)
    total, annualized, volatility, sharpe = None, None, None, None
    sharpe_reason: str | None = reason
    if observations:
        total = float(nav[-1].nav / nav[0].nav - 1)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            returns = values[1:] / values[:-1] - 1
        if not np.all(np.isfinite(returns)):
            raise DomainError("INVALID_NAV", "净值尺度产生非有限收益")
        try:
            annualized = math.expm1(
                math.log(float(nav[-1].nav / nav[0].nav)) * annualization / observations
            )
        except (OverflowError, ValueError):
            annualized = None
        if observations >= 2:
            standard_deviation = float(np.std(returns, ddof=1))
            volatility = standard_deviation * math.sqrt(annualization)
            if standard_deviation == 0:
                sharpe_reason = "zero_volatility"
            else:
                daily_rf = math.expm1(math.log1p(risk_free_annual) / annualization)
                sharpe = float(
                    (np.mean(returns) - daily_rf) / standard_deviation * math.sqrt(annualization)
                )
                sharpe_reason = None
    benchmark_return, excess_return = None, None
    benchmark_reason: str | None = "benchmark_not_provided"
    if benchmark is not None:
        _validate_points(benchmark)
        indexed = {point.session: point for point in benchmark}
        if observations and all(point.session in indexed for point in nav):
            benchmark_return = float(indexed[nav[-1].session].nav / indexed[nav[0].session].nav - 1)
            excess_return = total - benchmark_return if total is not None else None
            benchmark_reason = None
        else:
            benchmark_reason = "benchmark_missing_sessions" if nav else "empty_history"
    warnings = ("收益样本少于20个，年化和尾部估计稳定性有限。",) if observations < 20 else ()
    return PerformanceMetrics(
        total_return=measure(total, reason=None if total is not None else reason),
        annualized_return=measure(
            annualized,
            "annualized_ratio",
            None if annualized is not None else ("non_finite_result" if observations else reason),
        ),
        annualized_volatility=measure(
            volatility, "annualized_ratio", None if volatility is not None else reason
        ),
        max_drawdown=measure(
            max_drawdown([float(value) for value in values]), reason=None if nav else reason
        ),
        sharpe=measure(sharpe, "dimensionless", sharpe_reason),
        benchmark_return=metric(
            benchmark_return, "ratio", "date_aligned_benchmark", cutoff, benchmark_reason
        ),
        excess_return=metric(
            excess_return, "ratio", "portfolio_minus_benchmark_return", cutoff, benchmark_reason
        ),
        annualization=annualization,
        risk_free_annual=risk_free_annual,
        observations=observations,
        benchmark_basis=benchmark_basis if benchmark is not None else None,
        warnings=warnings,
    )


def historical_tail_risk(
    returns: NDArray[np.float64],
    confidence: float = 0.95,
    *,
    source: str = "historical_returns",
    as_of: datetime | None = None,
    empty_reason: str = "insufficient_history",
) -> TailRisk:
    values = np.asarray(returns, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise DomainError("INVALID_RETURNS", "尾部风险需要一维有限收益序列")
    if not math.isfinite(confidence) or not 0.5 <= confidence < 1:
        raise DomainError("INVALID_CONFIDENCE", "置信度必须在[0.5,1)内")
    var, cvar = None, None
    reason: str | None = empty_reason
    if len(values) >= 2:
        losses = -values
        threshold = float(np.quantile(losses, confidence, method="higher"))
        var = max(0.0, threshold)
        cvar = max(0.0, float(np.mean(losses[losses >= threshold])))
        reason = None
    return TailRisk(
        metric(var, "one_day_loss_ratio", source, as_of, reason),
        metric(cvar, "one_day_loss_ratio", source, as_of, reason),
        confidence,
        len(values),
        "higher",
        ("历史收益样本少于20个，尾部估计稳定性有限。",) if len(values) < 20 else (),
    )
