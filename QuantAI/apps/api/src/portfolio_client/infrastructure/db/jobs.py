import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import uuid4

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.execution.audit import canonical_json
from portfolio_client.domain.market.models import DatasetSnapshot
from portfolio_client.domain.research.backtest_models import BacktestRequest, Progress
from portfolio_client.domain.research.jobs import (
    BacktestSummary,
    JobFailure,
    JobRecord,
    ResearchArtifact,
)
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.models import (
    IdempotencyRecord,
    ResearchJobRecord,
    ResearchResultRecord,
)

JOB = TypeAdapter(JobRecord)
ARTIFACT = TypeAdapter(ResearchArtifact)
SUMMARY = TypeAdapter(BacktestSummary)


def _json[T](adapter: TypeAdapter[T], value: T) -> dict[str, Any]:
    return cast(dict[str, Any], adapter.dump_python(value, mode="json"))


def _get(session: Session, job_id: str) -> ResearchJobRecord:
    record = session.get(ResearchJobRecord, job_id)
    if record is None:
        raise DomainError("JOB_NOT_FOUND", "研究任务不存在")
    return record


def _save(record: ResearchJobRecord, job: JobRecord) -> JobRecord:
    updated = replace(job, updated_at=datetime.now(UTC))
    record.payload, record.state = _json(JOB, updated), updated.state
    return updated


class SQLiteJobs:
    def __init__(self, database: Database) -> None:
        self.database = database

    def submit(
        self, request: BacktestRequest, snapshot: DatasetSnapshot, key: str, trace_id: str
    ) -> JobRecord:
        digest = hashlib.sha256(
            canonical_json(request.model_dump(mode="json")).encode()
        ).hexdigest()
        with self.database.transaction() as session:
            prior = session.get(IdempotencyRecord, ("__jobs__", "backtest", key))
            if prior is not None:
                if digest != prior.request_hash:
                    raise DomainError("IDEMPOTENCY_CONFLICT", "该研究幂等键对应不同请求")
                return JOB.validate_json(prior.response_json)
            active = session.scalars(
                select(ResearchJobRecord.id)
                .where(ResearchJobRecord.state.in_(("queued", "running")))
                .limit(16)
            ).all()
            if len(active) >= 16:
                raise DomainError("JOB_CAPACITY", "研究队列已满，请等待任务结束或取消")
            now = datetime.now(UTC)
            job = JobRecord(
                uuid4().hex,
                "backtest",
                "queued",
                request,
                snapshot.version,
                snapshot.content_hash,
                Progress(0, 0, "排队中"),
                trace_id,
                now,
                now,
            )
            session.add(
                ResearchJobRecord(
                    id=job.id, state=job.state, payload=_json(JOB, job), created_at=now.isoformat()
                )
            )
            session.add(
                IdempotencyRecord(
                    scope="__jobs__",
                    operation="backtest",
                    key=key,
                    request_hash=digest,
                    response_json=JOB.dump_json(job).decode(),
                )
            )
            return job

    def get(self, job_id: str) -> JobRecord:
        with self.database.read() as session:
            return JOB.validate_python(_get(session, job_id).payload)

    def list_jobs(self, limit: int) -> list[JobRecord]:
        with self.database.read() as session:
            return [
                JOB.validate_python(record.payload)
                for record in session.scalars(
                    select(ResearchJobRecord)
                    .order_by(ResearchJobRecord.created_at.desc())
                    .limit(limit)
                )
            ]

    def claim(self) -> JobRecord | None:
        with self.database.transaction() as session:
            record = session.scalar(
                select(ResearchJobRecord)
                .where(ResearchJobRecord.state == "queued")
                .order_by(ResearchJobRecord.created_at, ResearchJobRecord.id)
                .limit(1)
            )
            return (
                _save(record, replace(JOB.validate_python(record.payload), state="running"))
                if record is not None
                else None
            )

    def cancel(self, job_id: str) -> JobRecord:
        with self.database.transaction() as session:
            record = _get(session, job_id)
            job = JOB.validate_python(record.payload)
            return (
                _save(record, replace(job, state="cancelled"))
                if job.state in {"queued", "running"}
                else job
            )

    def progress(self, job_id: str, progress: Progress) -> JobRecord:
        if not 0 <= progress.completed <= progress.total:
            raise DomainError("INVALID_PROGRESS", "任务进度范围无效")
        with self.database.transaction() as session:
            record = _get(session, job_id)
            job = JOB.validate_python(record.payload)
            if job.state != "running":
                raise DomainError(
                    "JOB_CANCELLED" if job.state == "cancelled" else "JOB_INTERRUPTED", "任务已停止"
                )
            return _save(record, replace(job, progress=progress))

    def complete(self, job_id: str, artifact: ResearchArtifact) -> JobRecord:
        with self.database.transaction() as session:
            record = _get(session, job_id)
            job = JOB.validate_python(record.payload)
            if job.state != "running":
                return job
            summary = BacktestSummary(
                artifact.id,
                artifact.snapshot.dataset_id,
                artifact.snapshot.version,
                artifact.snapshot.source_kind,
                job.created_at,
                len(artifact.result.trades),
                artifact.result.costs,
                artifact.metrics.total_return,
                artifact.metrics.max_drawdown,
            )
            session.add(
                ResearchResultRecord(
                    id=artifact.id,
                    dataset_id=artifact.snapshot.dataset_id,
                    payload=_json(ARTIFACT, artifact),
                    summary=_json(SUMMARY, summary),
                    created_at=job.created_at.isoformat(),
                )
            )
            return _save(record, replace(job, state="succeeded", result_id=artifact.id))

    def fail(self, job_id: str, error: JobFailure) -> JobRecord:
        with self.database.transaction() as session:
            record = _get(session, job_id)
            job = JOB.validate_python(record.payload)
            return (
                _save(record, replace(job, state="failed", error=error))
                if job.state == "running"
                else job
            )

    def recover(self) -> None:
        with self.database.transaction() as session:
            for record in session.scalars(
                select(ResearchJobRecord).where(ResearchJobRecord.state == "running")
            ):
                job = JOB.validate_python(record.payload)
                _save(
                    record,
                    replace(
                        job,
                        state="failed",
                        error=JobFailure(
                            "JOB_INTERRUPTED",
                            "上次进程退出，运行中任务已中断；请求与进度保留。",
                            job.trace_id,
                        ),
                    ),
                )

    def artifact(self, result_id: str) -> ResearchArtifact:
        with self.database.read() as session:
            record = session.get(ResearchResultRecord, result_id)
            if record is None:
                raise DomainError("BACKTEST_NOT_FOUND", "回测结果不存在")
            return ARTIFACT.validate_python(record.payload)

    def summaries(self) -> list[BacktestSummary]:
        with self.database.read() as session:
            return [
                SUMMARY.validate_python(record.summary)
                for record in session.scalars(
                    select(ResearchResultRecord).order_by(ResearchResultRecord.created_at.desc())
                )
            ]
