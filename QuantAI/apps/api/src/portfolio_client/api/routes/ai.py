import asyncio
from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import APIRouter, Header, Request
from pydantic import BaseModel, Field
from starlette.responses import StreamingResponse

from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.ai import AIService
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.ai_models import AIRun, ExplanationRequest


class AIConfig(BaseModel):
    configured: bool
    model: str | None
    max_output_tokens: int = 2048
    total_timeout_seconds: int = 60
    usage_note: str = "显式发起才调用模型；中断可能没有最终用量，费用以服务商账单为准"


class EventCursor(BaseModel):
    after: int = Field(default=0, ge=0)


def create_ai_router(ai: AIService) -> APIRouter:
    router = APIRouter(prefix="/v1/ai", tags=["optional-ai"])

    @router.get("/config", response_model=Envelope[AIConfig])
    def config(request: Request) -> Envelope[AIConfig]:
        return Envelope(
            data=AIConfig(configured=ai.provider is not None, model=ai.model),
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.post("/runs", response_model=Envelope[AIRun], status_code=202)
    async def start(
        payload: ExplanationRequest,
        request: Request,
        key: Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)],
    ) -> Envelope[AIRun]:
        return Envelope(
            data=await ai.start(payload, key, request.state.trace_id),
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.get("/runs/{run_id}", response_model=Envelope[AIRun])
    def get(run_id: str, request: Request) -> Envelope[AIRun]:
        return Envelope(
            data=ai.repository.get(run_id), meta=Metadata(trace_id=request.state.trace_id)
        )

    @router.post("/runs/{run_id}/cancel", response_model=Envelope[AIRun])
    async def cancel(run_id: str, request: Request) -> Envelope[AIRun]:
        return Envelope(data=ai.cancel(run_id), meta=Metadata(trace_id=request.state.trace_id))

    @router.post("/runs/{run_id}/events", response_class=StreamingResponse)
    async def events(run_id: str, payload: EventCursor, request: Request) -> StreamingResponse:
        initial = ai.repository.get(run_id)
        if payload.after > len(initial.events):
            raise DomainError("AI_EVENT_CURSOR_INVALID", "事件游标超出已保存记录")

        async def stream() -> AsyncIterator[str]:
            cursor = payload.after
            while True:
                record = ai.repository.get(run_id)
                for event in record.events[cursor:]:
                    yield f"id: {event.seq}\ndata: {event.model_dump_json()}\n\n"
                    cursor = event.seq
                if record.state != "running" or await request.is_disconnected():
                    break
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(), media_type="text/event-stream", headers={"X-Accel-Buffering": "no"}
        )

    return router
