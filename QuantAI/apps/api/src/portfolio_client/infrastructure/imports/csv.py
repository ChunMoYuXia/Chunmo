import csv
import hashlib
import io
import json
import math
import re
from dataclasses import replace
from datetime import UTC, date, datetime, time
from decimal import Decimal, InvalidOperation
from uuid import uuid4

from pypinyin import Style, lazy_pinyin

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.clock import CN_TIMEZONE
from portfolio_client.domain.market.models import (
    DailyBar,
    ImportIssue,
    ImportPreview,
    ImportRow,
    Security,
)

# This first release supports mainland daily equities; the market timezone is explicit.
ALIASES = {
    "代码": "code",
    "股票代码": "code",
    "证券代码": "code",
    "日期": "date",
    "交易日期": "date",
    "trade_date": "date",
    "开盘": "open",
    "开盘价": "open",
    "最高": "high",
    "最高价": "high",
    "最低": "low",
    "最低价": "low",
    "收盘": "close",
    "收盘价": "close",
    "成交量": "volume",
    "成交额": "amount",
    "成交金额": "amount",
    "名称": "name",
    "股票名称": "name",
    "简称": "name",
    "交易所": "exchange",
    "行业": "industry",
    "所属行业": "industry",
    "停牌": "suspended",
    "涨停价": "limit_up",
    "跌停价": "limit_down",
    "复权因子": "adj_factor",
    "是否交易": "is_open",
}
REQUIRED = {"code", "date", "open", "high", "low", "close", "volume"}
EXCHANGES = {"SH": "XSHG", "SZ": "XSHE", "BJ": "XBSE", "SSE": "XSHG", "SZSE": "XSHE", "BSE": "XBSE"}


def _decode(content: bytes) -> str:
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return content.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise DomainError("INVALID_ENCODING", "CSV编码无法识别，请使用UTF-8或GB18030")


def _price(value: str, field_name: str) -> Decimal:
    try:
        number = Decimal(value.strip())
    except (InvalidOperation, ValueError):
        raise DomainError("INVALID_PRICE", f"{field_name}必须为有效价格") from None
    if (
        not number.is_finite()
        or number <= 0
        or not math.isfinite(float(number))
        or float(number) <= 0
    ):
        raise DomainError("INVALID_PRICE", f"{field_name}必须为有限正数")
    return number


def _security(row: dict[str, str]) -> Security:
    raw = row["code"].strip().upper()
    if not re.fullmatch(r"\d{6}(\.(SH|SZ|BJ))?", raw):
        raise DomainError("INVALID_SECURITY", "股票代码须为6位文本，请保留前导零")
    parts = raw.split(".")
    code = parts[0]
    requested = row.get("exchange", "").strip().upper()
    explicit = EXCHANGES.get(requested, requested)
    suffix = EXCHANGES.get(parts[1]) if len(parts) == 2 else None
    inferred = (
        "XBSE"
        if code.startswith("920")
        else "XSHG"
        if code.startswith(("5", "6", "9"))
        else "XBSE"
        if code.startswith(("4", "8"))
        else "XSHE"
    )
    exchange = explicit or suffix or inferred
    if exchange not in {"XSHG", "XSHE", "XBSE"} or (explicit and suffix and explicit != suffix):
        raise DomainError("INVALID_EXCHANGE", "交易所标识无效或与代码后缀冲突")
    if code.startswith("920") and exchange != "XBSE":
        raise DomainError("INVALID_EXCHANGE", "920号段须使用北交所标识")
    name = row.get("name", "").strip() or f"标的{code}"
    board = (
        "star"
        if code.startswith("688")
        else "chinext"
        if code.startswith(("300", "301"))
        else "bse"
        if exchange == "XBSE"
        else "main"
    )
    initials = "".join(lazy_pinyin(name, style=Style.FIRST_LETTER)).lower()
    return Security(
        f"{exchange}:{code}", code, exchange, name, row.get("industry") or None, initials, board
    )


def _bar(row: dict[str, str], security: Security, volume_unit: str) -> DailyBar:
    try:
        day = date.fromisoformat(row["date"].strip().replace("/", "-"))
    except ValueError:
        raise DomainError("INVALID_DATE", "交易日期须为YYYY-MM-DD") from None
    opening, high, low, close = [_price(row[key], key) for key in ("open", "high", "low", "close")]
    if high < max(opening, close, low) or low > min(opening, close, high):
        raise DomainError("INVALID_OHLC", "最高价或最低价不包含开盘及收盘价")
    try:
        volume = Decimal(row["volume"])
    except InvalidOperation:
        raise DomainError("INVALID_VOLUME", "成交量必须为非负整数") from None
    factor = 100 if volume_unit == "lots" else 1
    if (
        not volume.is_finite()
        or volume < 0
        or volume != volume.to_integral_value()
        or volume > (2**63 - 1) // factor
    ):
        raise DomainError("INVALID_VOLUME", "成交量必须为非负整数")
    amount = Decimal(row["amount"]) if row.get("amount") else None
    if amount is not None and (
        not amount.is_finite() or amount < 0 or not math.isfinite(float(amount))
    ):
        raise DomainError("INVALID_AMOUNT", "成交额必须为有限非负数")
    suspended = row.get("suspended", "").strip().lower() in {"1", "true", "是"} or volume == 0
    return DailyBar(
        security.security_id,
        day,
        opening,
        high,
        low,
        close,
        int(volume) * factor,
        amount,
        datetime.combine(day, time(15), CN_TIMEZONE).astimezone(UTC),
        suspended,
        _price(row["limit_up"], "涨停价") if row.get("limit_up") else None,
        _price(row["limit_down"], "跌停价") if row.get("limit_down") else None,
        _price(row["adj_factor"], "复权因子") if row.get("adj_factor") else None,
    )


def preview_import(
    content: bytes, mapping: dict[str, str], *, volume_unit: str = "shares"
) -> ImportPreview:
    if volume_unit not in {"shares", "lots"}:
        raise DomainError("INVALID_VOLUME_UNIT", "成交量单位须为股或手")
    reader = csv.DictReader(io.StringIO(_decode(content)))
    headers = {
        name: mapping.get(name, ALIASES.get(name.strip(), name.strip()))
        for name in reader.fieldnames or []
    }
    rows: list[ImportRow] = []
    errors: list[ImportIssue] = []
    seen: set[tuple[str, date]] = set()
    if reader.fieldnames and len(reader.fieldnames) != len(set(reader.fieldnames)):
        errors.append(ImportIssue(1, "DUPLICATE_COLUMNS", "CSV表头存在重复列名"))
    missing = sorted(REQUIRED - set(headers.values()))
    if missing:
        errors.append(ImportIssue(1, "MISSING_COLUMNS", f"缺少字段：{', '.join(missing)}"))
    if len(set(headers.values())) != len(headers):
        errors.append(ImportIssue(1, "DUPLICATE_COLUMNS", "多列映射到了同一个字段"))
    if not errors:
        for raw in reader:
            line = reader.line_num
            row = {
                target: (raw.get(original) or "").strip() for original, target in headers.items()
            }
            try:
                if None in raw:
                    raise DomainError("INVALID_CSV_ROW", "该行字段数量与表头不一致")
                security = _security(row)
                bar = _bar(row, security, volume_unit)
                key = (bar.security_id, bar.trade_date)
                if key in seen:
                    raise DomainError("DUPLICATE_BAR", "同一标的和交易日出现重复记录")
                seen.add(key)
                rows.append(ImportRow(security, bar, line))
            except (DomainError, InvalidOperation) as error:
                issue = (
                    error
                    if isinstance(error, DomainError)
                    else DomainError("INVALID_AMOUNT", "成交额必须为有效数字")
                )
                errors.append(ImportIssue(line, issue.code, issue.message))
    if not rows and not errors:
        errors.append(ImportIssue(1, "EMPTY_DATASET", "CSV中没有行情记录"))
    digest = hashlib.sha256(
        content + json.dumps(mapping, sort_keys=True).encode() + volume_unit.encode() + b"csv-v2"
    ).hexdigest()
    warnings = [
        "价格单位按人民币元、成交量按选定单位解释；请在提交前核对。",
        "未附公司行为资料，复权及分红相关结论须单独核实。",
    ]
    if "amount" not in headers.values():
        warnings.append("未提供成交额，相应流动性指标不可用。")
    return ImportPreview(
        id=uuid4().hex,
        rows=tuple(rows),
        errors=tuple(errors),
        warnings=tuple(warnings),
        content_hash=digest,
        source_kind="imported",
        volume_unit=volume_unit,
        parser_version="csv-v2",
        column_mapping=tuple(sorted(mapping.items())),
    )


def parse_calendar(content: bytes) -> tuple[date, ...]:
    reader = csv.DictReader(io.StringIO(_decode(content)))
    dates: list[date] = []
    for raw in reader:
        row = {
            ALIASES.get(key.strip(), key.strip()): value or ""
            for key, value in raw.items()
            if key is not None
        }
        flag = row.get("is_open", "1").strip().lower()
        if flag not in {"0", "1", "true", "false", "是", "否"}:
            raise DomainError("INVALID_CALENDAR", f"日历第{reader.line_num}行开市标记无效")
        if flag in {"0", "false", "否"}:
            continue
        try:
            dates.append(date.fromisoformat(row.get("date", "").strip()))
        except ValueError:
            raise DomainError("INVALID_CALENDAR", f"日历第{reader.line_num}行日期无效") from None
    if not dates or len(dates) != len(set(dates)):
        raise DomainError("INVALID_CALENDAR", "交易日历为空或有重复日期")
    return tuple(sorted(dates))


class CsvImporter:
    def preview(
        self,
        content: bytes,
        mapping: dict[str, str],
        calendar_content: bytes | None,
        volume_unit: str,
    ) -> ImportPreview:
        result = preview_import(content, mapping, volume_unit=volume_unit)
        if calendar_content is None:
            return replace(result, warnings=(*result.warnings, "提交真实行情前需提供交易日历CSV。"))
        calendar = parse_calendar(calendar_content)
        dates = set(calendar)
        extra = tuple(
            ImportIssue(row.line_number, "UNKNOWN_SESSION", "行情日期不在提供的交易日历中")
            for row in result.rows
            if row.bar.trade_date not in dates
        )
        digest = hashlib.sha256(
            (result.content_hash + "|" + "|".join(day.isoformat() for day in calendar)).encode()
        ).hexdigest()
        return replace(
            result, calendar_dates=calendar, content_hash=digest, errors=(*result.errors, *extra)
        )
