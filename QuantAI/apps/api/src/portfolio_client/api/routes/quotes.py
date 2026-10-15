from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Query, Request

from portfolio_client.api.market_schemas import (
    BarDTO,
    FeatureDTO,
    SecurityDTO,
    feature_dto,
    market_metadata,
)
from portfolio_client.api.schemas import Envelope
from portfolio_client.application.exports import ExportDocument, export_market_csv
from portfolio_client.application.market import MarketService
from portfolio_client.domain.market.models import KnowledgeMode


def create_quotes_router(market: MarketService) -> APIRouter:
    router = APIRouter(prefix="/v1/market", tags=["market"])

    @router.get("/export", response_model=Envelope[ExportDocument])
    def export(
        request: Request,
        dataset_id: str,
        security_id: str,
        as_of: datetime | None = None,
        knowledge_mode: KnowledgeMode = "public_pit",
    ) -> Envelope[ExportDocument]:
        context = market.context(dataset_id, as_of, knowledge_mode)
        # Export the complete bounded history, never the chart's visual aggregation.
        return Envelope(
            data=export_market_csv(context, security_id, market.bars(context, security_id, 10000)),
            meta=market_metadata(request, context),
        )

    @router.get("/search", response_model=Envelope[list[SecurityDTO]])
    def search(
        request: Request,
        dataset_id: str,
        q: Annotated[str, Query(max_length=120)] = "",
        as_of: datetime | None = None,
        knowledge_mode: KnowledgeMode = "public_pit",
        limit: Annotated[int, Query(ge=1, le=200)] = 20,
    ) -> Envelope[list[SecurityDTO]]:
        context = market.context(dataset_id, as_of, knowledge_mode)
        return Envelope(
            data=[
                SecurityDTO.model_validate(item)
                for item in market.search_securities(context, q, limit)
            ],
            meta=market_metadata(request, context),
        )

    @router.get("/securities", response_model=Envelope[list[SecurityDTO]])
    def securities(
        request: Request,
        dataset_id: str,
        as_of: datetime | None = None,
        knowledge_mode: KnowledgeMode = "public_pit",
    ) -> Envelope[list[SecurityDTO]]:
        context = market.context(dataset_id, as_of, knowledge_mode)
        return Envelope(
            data=[SecurityDTO.model_validate(item) for item in market.securities(context)],
            meta=market_metadata(request, context),
        )

    @router.get("/kline", response_model=Envelope[list[BarDTO]])
    def kline(
        request: Request,
        dataset_id: str,
        security_id: str,
        as_of: datetime | None = None,
        knowledge_mode: KnowledgeMode = "public_pit",
        limit: Annotated[int, Query(ge=1, le=10000)] = 1000,
    ) -> Envelope[list[BarDTO]]:
        context = market.context(dataset_id, as_of, knowledge_mode)
        return Envelope(
            data=[BarDTO.model_validate(item) for item in market.bars(context, security_id, limit)],
            meta=market_metadata(request, context),
        )

    @router.get("/snapshot", response_model=Envelope[list[FeatureDTO]])
    def snapshot(
        request: Request,
        dataset_id: str,
        as_of: datetime | None = None,
        knowledge_mode: KnowledgeMode = "public_pit",
    ) -> Envelope[list[FeatureDTO]]:
        context = market.context(dataset_id, as_of, knowledge_mode)
        return Envelope(
            data=[feature_dto(item) for item in market.features(context)],
            meta=market_metadata(request, context),
        )

    return router
