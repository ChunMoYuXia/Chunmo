from datetime import date
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from portfolio_client.api.market_schemas import market_metadata
from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.risk import RiskService
from portfolio_client.domain.research.backtest_models import NavPoint
from portfolio_client.domain.research.exposures import ExposureResult
from portfolio_client.domain.research.metric_models import Metric, PerformanceMetrics, TailRisk
from portfolio_client.domain.research.metrics import compute_metrics


class PortfolioRiskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    portfolio_id: str
    mode: Literal["holdings", "target"] = "holdings"
    confidence: float = Field(default=0.95, ge=0.5, lt=1, allow_inf_nan=False)
    lookback: int = Field(default=20, ge=20, le=756)


class PortfolioRiskDTO(BaseModel):
    exposure: ExposureResult
    tail_risk: TailRisk
    annualized_volatility: Metric
    source: str
    price_basis: str | None
    warnings: tuple[str, ...]


class NavObservation(BaseModel):
    session: date
    value: Decimal = Field(gt=0, allow_inf_nan=False)


class MetricsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nav: list[NavObservation] = Field(max_length=10000)
    benchmark: list[NavObservation] | None = Field(default=None, max_length=10000)
    annualization: int = Field(default=244, ge=1, le=1000)
    risk_free_annual: float = Field(default=0, gt=-1, allow_inf_nan=False)
    benchmark_basis: Literal["price", "total_return"] = "price"


def create_risk_router(risk: RiskService) -> APIRouter:
    router = APIRouter(prefix="/v1/risk", tags=["risk"])

    @router.post("/portfolio", response_model=Envelope[PortfolioRiskDTO])
    def portfolio(payload: PortfolioRiskRequest, request: Request) -> Envelope[PortfolioRiskDTO]:
        run = risk.portfolio(**payload.model_dump())
        data = PortfolioRiskDTO(
            exposure=run.exposure,
            tail_risk=run.tail_risk,
            annualized_volatility=run.annualized_volatility,
            source=run.source,
            price_basis=run.price_basis,
            warnings=run.warnings,
        )
        return Envelope(data=data, meta=market_metadata(request, run.context))

    @router.post("/metrics", response_model=Envelope[PerformanceMetrics])
    def metrics(payload: MetricsRequest, request: Request) -> Envelope[PerformanceMetrics]:
        def convert(items: list[NavObservation]) -> list[NavPoint]:
            return [
                NavPoint(item.session, item.value, Decimal("0"), Decimal("0"), ()) for item in items
            ]

        result = compute_metrics(
            convert(payload.nav),
            convert(payload.benchmark) if payload.benchmark is not None else None,
            payload.annualization,
            payload.risk_free_annual,
            benchmark_basis=payload.benchmark_basis,
        )
        return Envelope(data=result, meta=Metadata(trace_id=request.state.trace_id))

    return router
