from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import cast

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.features import FeatureSnapshot
from portfolio_client.domain.screening.dsl import CompareOperator, Filter, ScreeningSpec
from portfolio_client.domain.screening.fields import FIELD_DEFINITIONS, FieldName

Scalar = Decimal | float | str | bool


@dataclass(frozen=True)
class ScreeningStep:
    index: int
    condition: Filter
    input_count: int
    remaining_count: int
    exclusions: Mapping[str, int]


@dataclass(frozen=True)
class ScreeningResult:
    universe_count: int
    total_matching_count: int
    rows: tuple[FeatureSnapshot, ...]
    steps: tuple[ScreeningStep, ...]


def field_value(row: FeatureSnapshot, field: FieldName) -> Scalar | None:
    if field in {FieldName.CODE, FieldName.NAME, FieldName.INDUSTRY}:
        return (
            cast(str | None, getattr(row.security, field.value))
            if row.security is not None
            else None
        )
    return cast(Scalar | None, getattr(row, field.value))


def _matches(value: Scalar, condition: Filter) -> bool:
    kind = FIELD_DEFINITIONS[condition.field].data_type
    if kind == "number":
        actual = Decimal(str(value))
        target = cast(Decimal, condition.value)
        match condition.operator:
            case CompareOperator.GT:
                return actual > target
            case CompareOperator.GTE:
                return actual >= target
            case CompareOperator.LT:
                return actual < target
            case CompareOperator.LTE:
                return actual <= target
            case CompareOperator.EQ:
                return actual == target
        raise DomainError("INVALID_OPERATOR", "不支持该数值筛选运算")
    if condition.operator == CompareOperator.IN:
        return value in cast(tuple[str, ...], condition.value)
    return value == condition.value


def _failure_reason(condition: Filter) -> str:
    if condition.operator in {CompareOperator.GT, CompareOperator.GTE}:
        return "below_threshold"
    if condition.operator in {CompareOperator.LT, CompareOperator.LTE}:
        return "above_threshold"
    return "not_in" if condition.operator == CompareOperator.IN else "not_equal"


def _sort_key(row: FeatureSnapshot, field: FieldName) -> Decimal | str:
    value = field_value(row, field)
    if FIELD_DEFINITIONS[field].data_type == "text":
        return str(value).casefold()
    return Decimal(int(value)) if isinstance(value, bool) else Decimal(str(value))


def screen(spec: ScreeningSpec, snapshot: list[FeatureSnapshot]) -> ScreeningResult:
    available = {row.security_id: row for row in snapshot}
    unknown = set(spec.universe) - available.keys()
    if unknown:
        raise DomainError(
            "SECURITY_NOT_FOUND", "候选池存在不可见标的", {"security_ids": sorted(unknown)}
        )
    remaining = [available[key] for key in spec.universe]
    steps: list[ScreeningStep] = []
    for index, condition in enumerate(spec.filters):
        kept: list[FeatureSnapshot] = []
        reasons: Counter[str] = Counter()
        for row in remaining:
            value = field_value(row, condition.field)
            if value is None:
                reasons["missing_value"] += 1
            elif _matches(value, condition):
                kept.append(row)
            else:
                reasons[_failure_reason(condition)] += 1
        steps.append(ScreeningStep(index, condition, len(remaining), len(kept), dict(reasons)))
        remaining = kept
    # Stable ID ordering is preserved for ties; missing values always remain last.
    remaining.sort(key=lambda row: row.security_id)
    present = [row for row in remaining if field_value(row, spec.sort_by) is not None]
    missing = [row for row in remaining if field_value(row, spec.sort_by) is None]
    present.sort(key=lambda row: _sort_key(row, spec.sort_by), reverse=spec.descending)
    return ScreeningResult(
        len(spec.universe), len(remaining), tuple((present + missing)[: spec.limit]), tuple(steps)
    )
