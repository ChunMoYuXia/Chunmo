from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from portfolio_client.domain.market.clock import as_utc
from portfolio_client.domain.market.models import KnowledgeMode

OptimizationMethod = Literal["equal", "inverse_volatility", "minimum_variance"]
Weight = Annotated[Decimal, Field(ge=0, le=1, allow_inf_nan=False)]
Quantity = Annotated[int, Field(ge=0, strict=True)]


class PortfolioSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=80)
    dataset_id: str
    as_of: datetime
    knowledge_mode: KnowledgeMode = "public_pit"
    target_weights: dict[str, Weight] = Field(min_length=1, max_length=500)
    max_weight: Weight = Decimal("1")
    holdings: dict[str, Quantity] = Field(default_factory=dict, max_length=500)
    cash: Decimal = Field(default=Decimal("100000"), ge=0, allow_inf_nan=False)

    @field_validator("as_of")
    @classmethod
    def aware_time(cls, value: datetime) -> datetime:
        return as_utc(value)

    @model_validator(mode="after")
    def valid_budget(self) -> Self:
        if sum(self.target_weights.values()) > 1:
            raise ValueError("目标权重总和不得超过1，剩余部分为现金")
        if any(weight > self.max_weight for weight in self.target_weights.values()):
            raise ValueError("目标权重超过个股上限")
        return self


@dataclass(frozen=True)
class SavedPortfolio:
    id: str
    spec: PortfolioSpec
    created_at: datetime


@dataclass(frozen=True)
class WeightSolution:
    weights: tuple[float, ...]
    method: OptimizationMethod
    observations: int
    covariance_shrinkage: float | None = None
