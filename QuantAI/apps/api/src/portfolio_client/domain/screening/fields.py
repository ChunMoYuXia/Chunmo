from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal


class FieldName(StrEnum):
    CODE = "code"
    NAME = "name"
    INDUSTRY = "industry"
    CLOSE = "close"
    MOMENTUM = "momentum_20d"
    VOLATILITY = "volatility_20d"
    AVERAGE_AMOUNT = "average_amount_20d"
    STALE = "stale"


@dataclass(frozen=True)
class FieldDefinition:
    key: FieldName
    label: str
    data_type: Literal["number", "text", "boolean"]
    unit: str
    display_unit: str
    display_scale: int = 1


FIELD_DEFINITIONS = MappingProxyType(
    {
        FieldName.CODE: FieldDefinition(FieldName.CODE, "证券代码", "text", "text", ""),
        FieldName.NAME: FieldDefinition(FieldName.NAME, "证券名称", "text", "text", ""),
        FieldName.INDUSTRY: FieldDefinition(FieldName.INDUSTRY, "行业", "text", "category", ""),
        FieldName.CLOSE: FieldDefinition(FieldName.CLOSE, "最新可见收盘价", "number", "CNY", "元"),
        FieldName.MOMENTUM: FieldDefinition(
            FieldName.MOMENTUM, "20日价格动量", "number", "fraction", "%", 100
        ),
        FieldName.VOLATILITY: FieldDefinition(
            FieldName.VOLATILITY, "20日年化波动", "number", "fraction", "%", 100
        ),
        FieldName.AVERAGE_AMOUNT: FieldDefinition(
            FieldName.AVERAGE_AMOUNT, "20日平均成交额", "number", "CNY", "元"
        ),
        FieldName.STALE: FieldDefinition(FieldName.STALE, "行情过期", "boolean", "boolean", ""),
    }
)
