from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel

from portfolio_client.domain.market.models import KnowledgeMode, SourceKind


class Metadata(BaseModel):
    trace_id: str
    dataset_id: str | None = None
    data_version: int | None = None
    as_of: datetime | None = None
    knowledge_mode: KnowledgeMode | None = None
    source_kind: SourceKind | None = None


class Envelope[DataT](BaseModel):
    data: DataT
    meta: Metadata


class ErrorBody(BaseModel):
    code: str
    message: str
    details: dict[str, Any]
    trace_id: str


class ErrorEnvelope(BaseModel):
    error: ErrorBody


class LocalConfig(BaseModel):
    mode: Literal["local"] = "local"
    version: str = "0.1.0"
    real_trading_enabled: bool = False
