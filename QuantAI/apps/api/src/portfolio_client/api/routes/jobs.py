from typing import Annotated

from fastapi import APIRouter, Header, Query, Request
from pydantic import BaseModel

from portfolio_client.api.market_schemas import market_metadata
from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.jobs import JobService
from portfolio_client.application.market import MarketContext
from portfolio_client.domain.market.clock import at_close
from portfolio_client.domain.research.backtest_models import BacktestRequest
from portfolio_client.domain.research.jobs import (
    BacktestSummary,
    JobRecord,
    JobState,
    ResearchArtifact,
)

Key = Annotated[str, Header(alias="Idempotency-Key", min_length=1, max_length=128)]


class JobSubmission(BaseModel):
    task_id: str
    state: JobState


def create_jobs_router(jobs: JobService) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["research-jobs"])

    def artifact_response(
        artifact: ResearchArtifact, request: Request
    ) -> Envelope[ResearchArtifact]:
        context = MarketContext(
            artifact.snapshot,
            at_close(artifact.result.request.end),
            artifact.result.request.knowledge_mode,
        )
        return Envelope(data=artifact, meta=market_metadata(request, context))

    @router.post("/backtests", response_model=Envelope[JobSubmission], status_code=202)
    def submit(payload: BacktestRequest, request: Request, key: Key) -> Envelope[JobSubmission]:
        job = jobs.submit("backtest", payload, key, request.state.trace_id)
        return Envelope(
            data=JobSubmission(task_id=job.id, state=job.state),
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.get("/backtests", response_model=Envelope[list[BacktestSummary]])
    def results(request: Request) -> Envelope[list[BacktestSummary]]:
        return Envelope(
            data=jobs.repository.summaries(), meta=Metadata(trace_id=request.state.trace_id)
        )

    @router.get("/backtests/{result_id}", response_model=Envelope[ResearchArtifact])
    def result(result_id: str, request: Request) -> Envelope[ResearchArtifact]:
        return artifact_response(jobs.repository.artifact(result_id), request)

    @router.get("/jobs", response_model=Envelope[list[JobRecord]])
    def all_jobs(
        request: Request, limit: int = Query(100, ge=1, le=1000)
    ) -> Envelope[list[JobRecord]]:
        return Envelope(
            data=jobs.repository.list_jobs(limit), meta=Metadata(trace_id=request.state.trace_id)
        )

    @router.get("/jobs/{job_id}", response_model=Envelope[JobRecord])
    def get(job_id: str, request: Request) -> Envelope[JobRecord]:
        return Envelope(data=jobs.get(job_id), meta=Metadata(trace_id=request.state.trace_id))

    @router.post("/jobs/{job_id}/cancel", response_model=Envelope[JobRecord])
    def cancel(job_id: str, request: Request) -> Envelope[JobRecord]:
        return Envelope(data=jobs.cancel(job_id), meta=Metadata(trace_id=request.state.trace_id))

    @router.get("/jobs/{job_id}/result", response_model=Envelope[ResearchArtifact])
    def job_result(job_id: str, request: Request) -> Envelope[ResearchArtifact]:
        return artifact_response(jobs.result(job_id), request)

    return router
