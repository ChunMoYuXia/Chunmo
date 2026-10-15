from uuid import uuid4

from portfolio_client.application.research import ResearchRun, ResearchService
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.backtest_models import BacktestRequest, Progress
from portfolio_client.domain.research.jobs import (
    JobFailure,
    JobRecord,
    JobRepositoryPort,
    ResearchArtifact,
)
from portfolio_client.domain.research.metrics import compute_metrics


class JobService:
    def __init__(self, repository: JobRepositoryPort, research: ResearchService) -> None:
        self.repository, self.research = repository, research

    def submit(
        self, kind: str, payload: BacktestRequest, key: str, trace_id: str | None = None
    ) -> JobRecord:
        if kind != "backtest":
            raise DomainError("INVALID_JOB_KIND", "M0研究队列仅接受回测任务")
        if not 1 <= len(key) <= 128:
            raise DomainError("INVALID_IDEMPOTENCY_KEY", "研究幂等键长度必须为1–128")
        snapshot = self.research.validate(payload)
        return self.repository.submit(payload, snapshot, key, trace_id or uuid4().hex)

    def get(self, job_id: str) -> JobRecord:
        return self.repository.get(job_id)

    def cancel(self, job_id: str) -> JobRecord:
        return self.repository.cancel(job_id)

    def claim(self) -> JobRecord | None:
        return self.repository.claim()

    def progress(self, job_id: str, progress: Progress) -> JobRecord:
        return self.repository.progress(job_id, progress)

    def complete(self, job_id: str, run: ResearchRun) -> JobRecord:
        artifact = ResearchArtifact(
            job_id,
            run.snapshot,
            run.result,
            compute_metrics(
                list(run.result.nav),
                list(run.result.benchmark) if run.result.benchmark is not None else None,
                run.result.request.annualization,
                run.result.request.risk_free_annual,
                benchmark_basis="price",
            ),
        )
        return self.repository.complete(job_id, artifact)

    def fail(self, job_id: str, code: str, message: str) -> JobRecord:
        job = self.get(job_id)
        return self.repository.fail(job_id, JobFailure(code, message, job.trace_id))

    def recover(self) -> None:
        self.repository.recover()

    def result(self, job_id: str) -> ResearchArtifact:
        job = self.get(job_id)
        if job.state != "succeeded" or job.result_id is None:
            raise DomainError("JOB_NOT_READY", "任务尚无可读取的成功结果", {"state": job.state})
        return self.repository.artifact(job.result_id)
