from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol

from portfolio_client.domain.market.models import DatasetSnapshot, SourceKind
from portfolio_client.domain.research.backtest_models import (
    BacktestRequest,
    BacktestResult,
    Progress,
)
from portfolio_client.domain.research.metric_models import Metric, PerformanceMetrics

JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]


@dataclass(frozen=True)
class JobFailure:
    code: str
    message: str
    trace_id: str


@dataclass(frozen=True)
class JobRecord:
    id: str
    kind: Literal["backtest"]
    state: JobState
    payload: BacktestRequest
    data_version: int
    content_hash: str
    progress: Progress
    trace_id: str
    created_at: datetime
    updated_at: datetime
    result_id: str | None = None
    error: JobFailure | None = None


@dataclass(frozen=True)
class ResearchArtifact:
    id: str
    snapshot: DatasetSnapshot
    result: BacktestResult
    metrics: PerformanceMetrics


@dataclass(frozen=True)
class BacktestSummary:
    id: str
    dataset_id: str
    data_version: int
    source_kind: SourceKind
    created_at: datetime
    trade_count: int
    costs: Decimal
    total_return: Metric
    max_drawdown: Metric


class JobRepositoryPort(Protocol):
    def submit(
        self, request: BacktestRequest, snapshot: DatasetSnapshot, key: str, trace_id: str
    ) -> JobRecord: ...
    def get(self, job_id: str) -> JobRecord: ...
    def list_jobs(self, limit: int) -> list[JobRecord]: ...
    def cancel(self, job_id: str) -> JobRecord: ...
    def claim(self) -> JobRecord | None: ...
    def progress(self, job_id: str, progress: Progress) -> JobRecord: ...
    def complete(self, job_id: str, artifact: ResearchArtifact) -> JobRecord: ...
    def fail(self, job_id: str, error: JobFailure) -> JobRecord: ...
    def recover(self) -> None: ...
    def artifact(self, result_id: str) -> ResearchArtifact: ...
    def summaries(self) -> list[BacktestSummary]: ...
