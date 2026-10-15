import json
from dataclasses import asdict
from datetime import date, datetime
from decimal import Decimal
from typing import Any, cast

from portfolio_client.domain.market.models import (
    CorporateActionRecord,
    DailyBar,
    ImportIssue,
    ImportPreview,
    ImportRow,
    Security,
    SourceFile,
)


def preview_payload(preview: ImportPreview) -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(json.dumps(asdict(preview), default=str)))


def action_from_payload(data: dict[str, Any]) -> CorporateActionRecord:
    values = dict(data)
    for field in ("record_date", "ex_date", "pay_date"):
        values[field] = date.fromisoformat(values[field])
    values["announced_at"] = datetime.fromisoformat(values["announced_at"])
    for field in ("cash_per_share", "bonus_ratio"):
        values[field] = Decimal(values[field])
    return CorporateActionRecord(**values)


def bar_from_payload(data: dict[str, Any]) -> DailyBar:
    values = dict(data)
    values["trade_date"] = date.fromisoformat(values["trade_date"])
    values["available_at"] = datetime.fromisoformat(values["available_at"])
    for field in ("open", "high", "low", "close", "amount", "limit_up", "limit_down", "adj_factor"):
        values[field] = Decimal(values[field]) if values[field] is not None else None
    return DailyBar(**values)


def preview_from_payload(data: dict[str, Any]) -> ImportPreview:
    rows = tuple(
        ImportRow(Security(**row["security"]), bar_from_payload(row["bar"]), row["line_number"])
        for row in data["rows"]
    )
    return ImportPreview(
        id=data["id"],
        rows=rows,
        errors=tuple(ImportIssue(**issue) for issue in data["errors"]),
        warnings=tuple(data["warnings"]),
        content_hash=data["content_hash"],
        source_kind=data["source_kind"],
        name=data["name"],
        calendar_dates=tuple(date.fromisoformat(value) for value in data["calendar_dates"]),
        actions=tuple(action_from_payload(action) for action in data["actions"]),
        volume_unit=data["volume_unit"],
        source_files=tuple(SourceFile(**item) for item in data.get("source_files", [])),
        generator_seed=data.get("generator_seed"),
        generator_version=data.get("generator_version"),
        parser_version=data.get("parser_version"),
        column_mapping=tuple(tuple(item) for item in data.get("column_mapping", [])),
    )
