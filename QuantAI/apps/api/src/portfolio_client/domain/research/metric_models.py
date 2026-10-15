from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class Metric:
    value: float | None
    unit: str
    formula_version: str
    source: str
    as_of: datetime | None
    reason: str | None = None


@dataclass(frozen=True)
class PerformanceMetrics:
    total_return: Metric
    annualized_return: Metric
    annualized_volatility: Metric
    max_drawdown: Metric
    sharpe: Metric
    benchmark_return: Metric
    excess_return: Metric
    annualization: int
    risk_free_annual: float
    observations: int
    benchmark_basis: str | None
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class TailRisk:
    var: Metric
    cvar: Metric
    confidence: float
    sample_count: int
    quantile_method: str
    warnings: tuple[str, ...]
