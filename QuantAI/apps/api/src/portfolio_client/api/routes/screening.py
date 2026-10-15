from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, ConfigDict, Field

from portfolio_client.api.market_schemas import FeatureDTO, feature_dto, market_metadata
from portfolio_client.api.schemas import Envelope, Metadata
from portfolio_client.application.screening import ScreeningService
from portfolio_client.domain.screening.dsl import Filter, ScreeningSpec
from portfolio_client.domain.screening.fields import FIELD_DEFINITIONS, FieldName


class ScreeningStepDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    index: int
    condition: Filter
    input_count: int
    remaining_count: int
    exclusions: dict[str, int]


class ScreeningResultDTO(BaseModel):
    universe_count: int
    total_matching_count: int
    rows: list[FeatureDTO]
    steps: list[ScreeningStepDTO]


class SavedScreeningDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    name: str
    spec: ScreeningSpec
    created_at: datetime


class SaveScreeningRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=80)
    spec: ScreeningSpec


class FieldDefinitionDTO(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    key: FieldName
    label: str
    data_type: Literal["number", "text", "boolean"]
    unit: str
    display_unit: str
    display_scale: int


def create_screening_router(screening: ScreeningService) -> APIRouter:
    router = APIRouter(prefix="/v1/screening", tags=["screening"])

    @router.post("", response_model=Envelope[ScreeningResultDTO])
    def run(payload: ScreeningSpec, request: Request) -> Envelope[ScreeningResultDTO]:
        run = screening.run(payload)
        result = ScreeningResultDTO(
            universe_count=run.result.universe_count,
            total_matching_count=run.result.total_matching_count,
            rows=[feature_dto(item) for item in run.result.rows],
            steps=[ScreeningStepDTO.model_validate(step) for step in run.result.steps],
        )
        return Envelope(data=result, meta=market_metadata(request, run.context))

    @router.get("/fields", response_model=Envelope[list[FieldDefinitionDTO]])
    def fields(request: Request) -> Envelope[list[FieldDefinitionDTO]]:
        return Envelope(
            data=[FieldDefinitionDTO.model_validate(field) for field in FIELD_DEFINITIONS.values()],
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.post("/presets", response_model=Envelope[SavedScreeningDTO], status_code=201)
    def save(payload: SaveScreeningRequest, request: Request) -> Envelope[SavedScreeningDTO]:
        return Envelope(
            data=SavedScreeningDTO.model_validate(screening.save(payload.name, payload.spec)),
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.get("/presets", response_model=Envelope[list[SavedScreeningDTO]])
    def presets(request: Request) -> Envelope[list[SavedScreeningDTO]]:
        return Envelope(
            data=[SavedScreeningDTO.model_validate(item) for item in screening.list()],
            meta=Metadata(trace_id=request.state.trace_id),
        )

    @router.get("/presets/{preset_id}", response_model=Envelope[SavedScreeningDTO])
    def preset(preset_id: str, request: Request) -> Envelope[SavedScreeningDTO]:
        return Envelope(
            data=SavedScreeningDTO.model_validate(screening.get(preset_id)),
            meta=Metadata(trace_id=request.state.trace_id),
        )

    return router
