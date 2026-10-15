import hashlib
from uuid import uuid4

from sqlalchemy import func, select

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.execution.audit import canonical_json
from portfolio_client.domain.research.ai_models import (
    AIEvent,
    AIEventPayload,
    AIRun,
    ExplanationRequest,
)
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.models import AIRunRecord


def request_hash(request: ExplanationRequest) -> str:
    return hashlib.sha256(canonical_json(request.model_dump(mode="json")).encode()).hexdigest()


def checked(record: AIRunRecord, request: ExplanationRequest) -> AIRun:
    if record.request_hash != request_hash(request):
        raise DomainError("IDEMPOTENCY_CONFLICT", "同一幂等键对应了不同AI请求")
    return AIRun.model_validate(record.payload)


class SQLiteAIRuns:
    def __init__(self, database: Database) -> None:
        self.database = database

    def find(self, key: str, request: ExplanationRequest) -> AIRun | None:
        with self.database.read() as session:
            record = session.scalar(select(AIRunRecord).where(AIRunRecord.idempotency_key == key))
            return checked(record, request) if record else None

    def create(
        self, key: str, request: ExplanationRequest, model: str, trace_id: str
    ) -> tuple[AIRun, bool]:
        with self.database.transaction() as session:
            record = session.scalar(select(AIRunRecord).where(AIRunRecord.idempotency_key == key))
            if record:
                return checked(record, request), False
            count = (
                session.scalar(
                    select(func.count())
                    .select_from(AIRunRecord)
                    .where(AIRunRecord.state == "running")
                )
                or 0
            )
            if count >= 4:
                raise DomainError("AI_CAPACITY", "已有过多AI请求，请等待或取消")
            run = AIRun(
                id=uuid4().hex,
                state="running",
                model=model,
                request=request,
                events=(),
                trace_id=trace_id,
            )
            session.add(
                AIRunRecord(
                    id=run.id,
                    idempotency_key=key,
                    request_hash=request_hash(request),
                    state=run.state,
                    payload=run.model_dump(mode="json"),
                )
            )
            return run, True

    def get(self, run_id: str) -> AIRun:
        with self.database.read() as session:
            record = session.get(AIRunRecord, run_id)
            if not record:
                raise DomainError("AI_RUN_NOT_FOUND", "AI运行记录不存在或已被移除")
            return AIRun.model_validate(record.payload)

    def append(self, run_id: str, kind: str, payload: AIEventPayload) -> AIRun:
        with self.database.transaction() as session:
            record = session.get(AIRunRecord, run_id)
            if not record:
                raise DomainError("AI_RUN_NOT_FOUND", "AI运行记录不存在")
            run = AIRun.model_validate(record.payload)
            if run.state != "running":
                return run
            event = AIEvent.model_validate(
                {
                    "run_id": run_id,
                    "seq": len(run.events) + 1,
                    "type": kind,
                    "payload": payload.model_dump(),
                }
            )
            terminal = kind if kind in {"succeeded", "failed", "cancelled"} else "running"
            updated = AIRun.model_validate(
                {**run.model_dump(), "state": terminal, "events": (*run.events, event)}
            )
            record.state, record.payload = updated.state, updated.model_dump(mode="json")
            return updated

    def recover(self) -> None:
        with self.database.read() as session:
            ids = list(
                session.scalars(select(AIRunRecord.id).where(AIRunRecord.state == "running"))
            )
        for run_id in ids:
            self.append(
                run_id,
                "failed",
                AIEventPayload(
                    message="本地服务已重启；保留已有事件，不自动重新付费", code="AI_INTERRUPTED"
                ),
            )
