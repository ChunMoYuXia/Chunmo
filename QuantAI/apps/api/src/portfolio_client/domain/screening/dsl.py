from datetime import datetime
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from portfolio_client.domain.market.clock import as_utc
from portfolio_client.domain.market.models import KnowledgeMode
from portfolio_client.domain.screening.fields import FIELD_DEFINITIONS, FieldName


class CompareOperator(StrEnum):
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    EQ = "eq"
    IN = "in"


class Filter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    field: FieldName
    operator: CompareOperator
    value: Decimal | str | bool | tuple[str, ...] = Field(
        description="比例用小数（0.1=10%），金额用人民币元；文本in使用字符串数组"
    )

    @model_validator(mode="after")
    def validate_operation(self) -> Self:
        kind = FIELD_DEFINITIONS[self.field].data_type
        if kind == "number":
            if self.operator == CompareOperator.IN or isinstance(self.value, (bool, tuple)):
                raise ValueError("数值条件需要有限数字和数值比较运算")
            try:
                number = Decimal(str(self.value))
            except InvalidOperation:
                raise ValueError("筛选数值无效") from None
            if not number.is_finite():
                raise ValueError("筛选数值必须有限")
            object.__setattr__(self, "value", number)
        elif kind == "boolean":
            if self.operator != CompareOperator.EQ or not isinstance(self.value, bool):
                raise ValueError("布尔条件需要eq和布尔值")
        elif self.operator == CompareOperator.IN:
            if not isinstance(self.value, tuple) or not self.value or len(self.value) > 500:
                raise ValueError("文本in条件需要非空字符串数组（最多500项）")
        elif self.operator != CompareOperator.EQ or not isinstance(self.value, str):
            raise ValueError("文本条件只支持eq或in")
        return self


class ScreeningSpec(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    version: Literal[1]
    dataset_id: str = Field(min_length=1, max_length=64)
    as_of: datetime
    universe: tuple[str, ...] = Field(max_length=10000)
    filters: tuple[Filter, ...] = Field(default=(), max_length=20)
    sort_by: FieldName = FieldName.MOMENTUM
    descending: bool = Field(default=True, strict=True)
    limit: int = Field(default=20, ge=1, le=1000)
    knowledge_mode: KnowledgeMode = "public_pit"

    @field_validator("as_of")
    @classmethod
    def normalize_as_of(cls, value: datetime) -> datetime:
        return as_utc(value)

    @field_validator("universe")
    @classmethod
    def validate_universe(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if len(values) != len(set(values)):
            raise ValueError("候选池不能重复包含同一标的")
        if any(not value or len(value) > 64 for value in values):
            raise ValueError("候选池标的标识无效")
        return values
