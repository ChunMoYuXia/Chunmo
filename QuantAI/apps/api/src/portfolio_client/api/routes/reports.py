from typing import Literal

from fastapi import APIRouter, Request

from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.exports import ExportDocument
from portfolio_client.application.reports import ReportService
from portfolio_client.domain.research.reports import ResearchReport


def create_reports_router(reports: ReportService) -> APIRouter:
    router = APIRouter(prefix="/v1/reports", tags=["reports"])

    @router.get("/{artifact_id}", response_model=Envelope[ResearchReport])
    def report(artifact_id: str, request: Request) -> Envelope[ResearchReport]:
        return Envelope(
            data=reports.get(artifact_id), meta=Metadata(trace_id=request.state.trace_id)
        )

    @router.get("/{artifact_id}/export", response_model=Envelope[ExportDocument])
    def export(
        artifact_id: str,
        request: Request,
        format: Literal["markdown", "metrics", "nav", "trades"] = "markdown",
    ) -> Envelope[ExportDocument]:
        return Envelope(
            data=reports.export(artifact_id, format), meta=Metadata(trace_id=request.state.trace_id)
        )

    return router
