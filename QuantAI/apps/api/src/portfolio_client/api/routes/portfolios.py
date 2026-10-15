from datetime import datetime
from decimal import Decimal
from typing import Self

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from portfolio_client.api.market_schemas import market_metadata
from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.portfolios import PortfolioService
from portfolio_client.domain.accounting.models import FeeSchedule
from portfolio_client.domain.market.clock import as_utc
from portfolio_client.domain.market.models import KnowledgeMode
from portfolio_client.domain.portfolio.models import (
    OptimizationMethod,
    PortfolioSpec,
    Quantity,
    SavedPortfolio,
    Weight,
)
from portfolio_client.domain.portfolio.rebalancing import PlannedTrade


class PortfolioDTO(BaseModel):
    id: str
    spec: PortfolioSpec
    created_at: datetime
    cash_weight: str


def portfolio_dto(value: SavedPortfolio) -> PortfolioDTO:
    return PortfolioDTO(
        id=value.id,
        spec=value.spec,
        created_at=value.created_at,
        cash_weight=str(1 - sum(value.spec.target_weights.values())),
    )


class OptimizeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    dataset_id: str
    as_of: datetime
    universe: list[str] = Field(min_length=1, max_length=500)
    method: OptimizationMethod
    max_weight: Weight = Decimal("1")
    lookback: int = Field(default=20, ge=20, le=756)
    knowledge_mode: KnowledgeMode = "public_pit"

    @field_validator("as_of")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        return as_utc(value)

    @model_validator(mode="after")
    def unique_assets(self) -> Self:
        if len(set(self.universe)) != len(self.universe):
            raise ValueError("组合不能包含重复标的")
        return self


class OptimizationDTO(BaseModel):
    target_weights: dict[str, str]
    method: OptimizationMethod
    observations: int
    covariance_shrinkage: float | None
    price_basis: str
    warnings: tuple[str, ...]


class RebalanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sellable: dict[str, Quantity] | None = None
    fee_schedule: FeeSchedule | None = None


class RebalanceDTO(BaseModel):
    orders: tuple[PlannedTrade, ...]
    cash_after: Decimal
    estimated_fees: Decimal
    equity_before: Decimal
    assumptions: tuple[str, ...]
    fee_schedule: FeeSchedule


def create_portfolio_router(portfolios: PortfolioService) -> APIRouter:
    router = APIRouter(prefix="/v1/portfolios", tags=["portfolios"])

    @router.post("/{portfolio_id}/rebalance", response_model=Envelope[RebalanceDTO])
    def rebalance(
        portfolio_id: str, payload: RebalanceRequest, request: Request
    ) -> Envelope[RebalanceDTO]:
        run = portfolios.rebalance(portfolio_id, payload.sellable, payload.fee_schedule)
        plan = run.plan
        data = RebalanceDTO(
            orders=plan.orders,
            cash_after=plan.cash_after,
            estimated_fees=plan.estimated_fees,
            equity_before=plan.equity_before,
            assumptions=plan.assumptions,
            fee_schedule=run.fee_schedule,
        )
        return Envelope(data=data, meta=market_metadata(request, run.context))

    @router.post("/optimize", response_model=Envelope[OptimizationDTO])
    def optimize(payload: OptimizeRequest, request: Request) -> Envelope[OptimizationDTO]:
        run = portfolios.optimize(**payload.model_dump())
        data = OptimizationDTO(
            target_weights={key: str(weight) for key, weight in run.target_weights.items()},
            method=run.solution.method,
            observations=run.solution.observations,
            covariance_shrinkage=run.solution.covariance_shrinkage,
            price_basis=run.price_basis,
            warnings=run.warnings,
        )
        return Envelope(data=data, meta=market_metadata(request, run.context))

    @router.post("", response_model=Envelope[PortfolioDTO], status_code=201)
    def save(payload: PortfolioSpec, request: Request) -> Envelope[PortfolioDTO]:
        return Envelope(
            data=portfolio_dto(portfolios.save(payload)),
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.get("", response_model=Envelope[list[PortfolioDTO]])
    def all_portfolios(request: Request) -> Envelope[list[PortfolioDTO]]:
        return Envelope(
            data=[portfolio_dto(value) for value in portfolios.repository.list()],
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.get("/{portfolio_id}", response_model=Envelope[PortfolioDTO])
    def get(portfolio_id: str, request: Request) -> Envelope[PortfolioDTO]:
        return Envelope(
            data=portfolio_dto(portfolios.repository.get(portfolio_id)),
            meta=Metadata(trace_id=request.state.trace_id),
        )

    return router
