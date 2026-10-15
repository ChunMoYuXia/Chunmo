import json
from datetime import date, datetime
from decimal import Decimal
from typing import Annotated, Literal

from fastapi import APIRouter, File, Form, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.datasets import DatasetService
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import (
    DatasetSnapshot,
    ImportPreview,
    SourceFile,
    SourceKind,
)

MAX_UPLOAD_BYTES = 50 * 1024 * 1024


class DatasetSummary(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    dataset_id: str
    version: int
    calendar_id: str
    source_kind: SourceKind
    name: str
    content_hash: str
    row_count: int
    security_count: int
    first_session: date
    last_session: date
    ingested_at: datetime
    warnings: tuple[str, ...]
    source_files: tuple[SourceFile, ...]
    generator_seed: int | None
    generator_version: str | None
    parser_version: str | None
    column_mapping: tuple[tuple[str, str], ...]
    volume_unit: str


class PreviewRow(BaseModel):
    security_id: str
    code: str
    name: str
    exchange: str
    industry: str | None
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    amount: Decimal | None


class PreviewIssue(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    row: int
    code: str
    message: str


class PreviewSummary(BaseModel):
    id: str
    name: str
    row_count: int
    source_kind: SourceKind
    content_hash: str
    rows: list[PreviewRow]
    errors: list[PreviewIssue]
    warnings: tuple[str, ...]
    calendar_sessions: int
    volume_unit: str
    can_commit: bool


class SyntheticRequest(BaseModel):
    seed: int = Field(default=42, ge=0, le=2**32 - 1)
    securities: int = Field(default=12, ge=1, le=200)
    sessions: int = Field(default=756, ge=1, le=2520)


def _dataset(data: DatasetSnapshot, request: Request) -> Envelope[DatasetSummary]:
    return Envelope(
        data=DatasetSummary.model_validate(data), meta=Metadata(trace_id=request.state.trace_id)
    )


def _preview(data: ImportPreview, request: Request) -> Envelope[PreviewSummary]:
    rows = [
        PreviewRow(
            security_id=row.security.security_id,
            code=row.security.code,
            name=row.security.name,
            exchange=row.security.exchange,
            industry=row.security.industry,
            trade_date=row.bar.trade_date,
            open=row.bar.open,
            high=row.bar.high,
            low=row.bar.low,
            close=row.bar.close,
            volume=row.bar.volume,
            amount=row.bar.amount,
        )
        for row in data.rows[:50]
    ]
    summary = PreviewSummary(
        id=data.id,
        name=data.name,
        row_count=len(data.rows),
        source_kind=data.source_kind,
        content_hash=data.content_hash,
        rows=rows,
        errors=[PreviewIssue.model_validate(issue) for issue in data.errors],
        warnings=data.warnings,
        calendar_sessions=len(data.calendar_dates),
        volume_unit=data.volume_unit,
        can_commit=bool(data.rows and not data.errors and data.calendar_dates),
    )
    return Envelope(data=summary, meta=Metadata(trace_id=request.state.trace_id))


async def _read_file(upload: UploadFile) -> bytes:
    if not upload.filename or not upload.filename.lower().endswith(".csv"):
        raise DomainError("INVALID_FILE_TYPE", "请上传CSV文件")
    content = await upload.read(MAX_UPLOAD_BYTES + 1)
    if len(content) > MAX_UPLOAD_BYTES:
        raise DomainError("FILE_TOO_LARGE", "单个CSV文件不得超过50MB")
    return content


def create_market_router(datasets: DatasetService) -> APIRouter:
    router = APIRouter(prefix="/v1/datasets", tags=["datasets"])

    @router.get("", response_model=Envelope[list[DatasetSummary]])
    def list_datasets(request: Request) -> Envelope[list[DatasetSummary]]:
        return Envelope(
            data=[DatasetSummary.model_validate(item) for item in datasets.list_datasets()],
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.post("/synthetic", response_model=Envelope[DatasetSummary], status_code=201)
    def synthetic(payload: SyntheticRequest, request: Request) -> Envelope[DatasetSummary]:
        return _dataset(
            datasets.create_synthetic(payload.seed, payload.securities, payload.sessions), request
        )

    @router.post("/import/preview", response_model=Envelope[PreviewSummary])
    async def preview(
        request: Request,
        file: Annotated[UploadFile, File()],
        calendar_file: Annotated[UploadFile | None, File()] = None,
        mapping: Annotated[str, Form()] = "{}",
        volume_unit: Annotated[Literal["shares", "lots"], Form()] = "shares",
    ) -> Envelope[PreviewSummary]:
        content = await _read_file(file)
        calendar = await _read_file(calendar_file) if calendar_file else None
        try:
            parsed_mapping = json.loads(mapping)
        except json.JSONDecodeError:
            raise DomainError("INVALID_MAPPING", "字段映射格式无效") from None
        if not isinstance(parsed_mapping, dict) or not all(
            isinstance(key, str) and isinstance(value, str) for key, value in parsed_mapping.items()
        ):
            raise DomainError("INVALID_MAPPING", "字段映射须为字段名称对照表")
        result = await run_in_threadpool(
            datasets.preview,
            content,
            parsed_mapping,
            calendar,
            volume_unit,
            file.filename or "导入行情",
        )
        return _preview(result, request)

    @router.post(
        "/import/{preview_id}/commit", response_model=Envelope[DatasetSummary], status_code=201
    )
    def commit(preview_id: str, request: Request) -> Envelope[DatasetSummary]:
        return _dataset(datasets.commit_import(preview_id), request)

    @router.get("/{dataset_id}", response_model=Envelope[DatasetSummary])
    def get_dataset(dataset_id: str, request: Request) -> Envelope[DatasetSummary]:
        return _dataset(datasets.get_dataset(dataset_id), request)

    @router.get("/{dataset_id}/sources/{role}")
    def source(dataset_id: str, role: Literal["quotes", "calendar"]) -> Response:
        return Response(
            content=datasets.read_source(dataset_id, role),
            media_type="application/octet-stream",
            headers={"Content-Disposition": f'attachment; filename="{role}.csv"'},
        )

    return router
