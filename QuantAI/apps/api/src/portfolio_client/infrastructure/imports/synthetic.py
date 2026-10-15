"""Reproducible development market; never a source of investment evidence."""

import hashlib
import json
import random
from dataclasses import asdict
from datetime import UTC, date, datetime, time, timedelta
from decimal import ROUND_HALF_UP, Decimal
from uuid import uuid4

from pypinyin import Style, lazy_pinyin

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.clock import CN_TIMEZONE
from portfolio_client.domain.market.models import (
    CorporateActionRecord,
    DailyBar,
    ImportPreview,
    ImportRow,
    Security,
)


def _price(value: float | Decimal) -> Decimal:
    return Decimal(str(value)).quantize(Decimal(".01"), rounding=ROUND_HALF_UP)


def generate_dataset(seed: int = 42, securities: int = 12, sessions: int = 756) -> ImportPreview:
    if not 1 <= securities <= 200 or not 1 <= sessions <= 2520:
        raise DomainError("INVALID_SYNTHETIC_SIZE", "演示数据支持1–200只证券、1–2520个交易日")
    rng = random.Random(seed)
    calendar: list[date] = []
    cursor = date(2023, 1, 2)
    while len(calendar) < sessions:
        if cursor.weekday() < 5:
            calendar.append(cursor)
        cursor += timedelta(days=1)
    industries = ["金融", "消费", "科技", "工业", "医药", "能源"]
    identities: list[Security] = []
    for index in range(securities):
        exchange = "XSHG" if index % 2 == 0 else "XSHE"
        code = f"{600001 + index:06d}" if exchange == "XSHG" else f"{index + 1:06d}"
        industry = industries[index % len(industries)]
        name = f"演示{industry}{index + 1:02d}"
        initials = "".join(lazy_pinyin(name, style=Style.FIRST_LETTER)).lower()
        identities.append(Security(f"{exchange}:{code}", code, exchange, name, industry, initials))
    actions: list[CorporateActionRecord] = []
    if sessions > 126:
        actions.append(
            CorporateActionRecord(
                "synthetic-dividend-1",
                identities[0].security_id,
                "dividend",
                calendar[125],
                calendar[126],
                calendar[min(129, sessions - 1)],
                datetime.combine(calendar[124], time(15), CN_TIMEZONE).astimezone(UTC),
                cash_per_share=Decimal(".15"),
            )
        )
    prior = [_price(8 + index * 2.8) for index in range(securities)]
    previous_returns = [0.0] * securities
    rows: list[ImportRow] = []
    for day_index, day in enumerate(calendar):
        market_return = rng.gauss(0.00025, 0.008)
        sector_returns = [rng.gauss(0, 0.004) for _ in industries]
        for index, security in enumerate(identities):
            reference = prior[index] - (Decimal(".15") if day_index == 126 and index == 0 else 0)
            up, down = _price(reference * Decimal("1.10")), _price(reference * Decimal(".90"))
            change = (
                market_return
                + sector_returns[index % len(industries)]
                + rng.gauss(0, 0.011)
                + 0.08 * previous_returns[index]
            )
            change = max(-0.09, min(0.09, change))
            opening = max(down, min(up, _price(reference * Decimal(str(1 + rng.gauss(0, 0.003))))))
            close = max(down, min(up, _price(max(0.5, float(reference) * (1 + change)))))
            high = min(up, _price(max(opening, close) * Decimal(str(1 + rng.uniform(0, 0.009)))))
            low = max(down, _price(min(opening, close) * Decimal(str(1 - rng.uniform(0, 0.009)))))
            volume = rng.randint(800_000, 6_000_000)
            suspended = day_index == 47 and index == 0
            if suspended:
                opening = high = low = close = prior[index]
                volume = 0
            elif day_index == 83 and index == 1:
                opening = high = low = close = up
            elif day_index == 97 and index == 2:
                opening = high = low = close = down
            amount = _price((opening + close) / 2 * volume)
            bar = DailyBar(
                security.security_id,
                day,
                opening,
                high,
                low,
                close,
                volume,
                amount,
                datetime.combine(day, time(15), CN_TIMEZONE).astimezone(UTC),
                suspended,
                up,
                down,
            )
            rows.append(ImportRow(security, bar, len(rows) + 2))
            previous_returns[index] = float(close / reference - 1)
            prior[index] = close
    payload = {
        "generator": "synthetic-v1",
        "seed": seed,
        "rows": [asdict(row) for row in rows],
        "actions": [asdict(action) for action in actions],
        "calendar": calendar,
    }
    content_hash = hashlib.sha256(
        json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return ImportPreview(
        id=uuid4().hex,
        rows=tuple(rows),
        errors=(),
        warnings=(
            "合成行情仅用于开发与流程验证，不能证明策略有效性。",
            "合成日历仅过滤周末，未还原真实市场节假日。",
        ),
        content_hash=content_hash,
        source_kind="synthetic",
        name=f"演示市场（{securities}只证券）",
        calendar_dates=tuple(calendar),
        actions=tuple(actions),
        generator_seed=seed,
        generator_version="synthetic-v1",
    )
