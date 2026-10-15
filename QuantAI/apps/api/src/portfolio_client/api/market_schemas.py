from datetime import date, datetime
from decimal import Decimal
from typing import Literal, cast

from fastapi import Request
from pydantic import BaseModel, ConfigDict

from portfolio_client.api.schemas import Metadata
from portfolio_client.application.market import MarketContext
from portfolio_client.domain.market.models import Security
from portfolio_client.domain.research.features import FeatureSnapshot


class SecurityDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    security_id: str
    code: str
    exchange: str
    name: str
    industry: str | None
    initials: str
    board: str
    currency: str


class BarDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    security_id: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    amount: Decimal | None
    available_at: datetime
    suspended: bool
    limit_up: Decimal | None
    limit_down: Decimal | None
    adj_factor: Decimal | None


class FeatureDTO(BaseModel):
    security_id: str
    code: str
    exchange: str
    name: str
    industry: str | None
    close: Decimal | None
    momentum_20d: float | None
    volatility_20d: float | None
    average_amount_20d: Decimal | None
    sample_count: int
    last_session: date | None
    price_basis: Literal["raw", "adjusted", "incomplete"]
    warnings: tuple[str, ...]
    missing_sessions: tuple[date, ...]
    stale: bool


def feature_dto(feature: FeatureSnapshot) -> FeatureDTO:
    # The application projection always attaches the time-visible identity.
    security = cast(Security, feature.security)
    return FeatureDTO(
        security_id=feature.security_id,
        code=security.code,
        exchange=security.exchange,
        name=security.name,
        industry=security.industry,
        close=feature.close,
        momentum_20d=feature.momentum_20d,
        volatility_20d=feature.volatility_20d,
        average_amount_20d=feature.average_amount_20d,
        sample_count=feature.sample_count,
        last_session=feature.last_session,
        price_basis=feature.price_basis,
        warnings=feature.warnings,
        missing_sessions=feature.missing_sessions,
        stale=feature.stale,
    )


def market_metadata(request: Request, context: MarketContext) -> Metadata:
    return Metadata(
        trace_id=request.state.trace_id,
        dataset_id=context.snapshot.dataset_id,
        data_version=context.snapshot.version,
        as_of=context.as_of,
        knowledge_mode=context.knowledge_mode,
        source_kind=context.snapshot.source_kind,
    )
