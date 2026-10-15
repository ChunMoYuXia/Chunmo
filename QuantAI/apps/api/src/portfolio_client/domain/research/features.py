from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

import numpy as np

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import DailyBar, Security


@dataclass(frozen=True)
class FeatureSnapshot:
    security_id: str
    close: Decimal | None
    momentum_20d: float | None
    volatility_20d: float | None
    average_amount_20d: Decimal | None
    sample_count: int
    last_session: date | None
    as_of: datetime | None
    price_basis: Literal["raw", "adjusted", "incomplete"] = "raw"
    security: Security | None = None
    data_version: int = 0
    warnings: tuple[str, ...] = ()
    missing_sessions: tuple[date, ...] = ()
    stale: bool = False


def compute_features(bars: list[DailyBar], window: int = 20) -> FeatureSnapshot:
    if window != 20:
        raise DomainError("UNSUPPORTED_WINDOW", "首版特征采用20个交易日窗口")
    ordered = sorted(bars, key=lambda bar: bar.trade_date)
    if len({bar.security_id for bar in ordered}) > 1:
        raise DomainError("INVALID_FEATURE_HISTORY", "单只证券的特征不能混入其他证券行情")
    latest = ordered[-1] if ordered else None
    warnings: list[str] = []
    complete_adjustment = bool(ordered) and all(bar.adj_factor is not None for bar in ordered)
    partial_adjustment = (
        any(bar.adj_factor is not None for bar in ordered) and not complete_adjustment
    )
    basis: Literal["raw", "adjusted", "incomplete"] = (
        "incomplete" if partial_adjustment else "adjusted" if complete_adjustment else "raw"
    )
    if basis == "raw":
        warnings.append("价格特征未包含分红送转调整。")
    elif basis == "incomplete":
        warnings.append("复权因子存在缺口，价格收益特征不可用。")
    momentum: float | None = None
    volatility: float | None = None
    average_amount: Decimal | None = None
    if len(ordered) >= window + 1 and not partial_adjustment:
        decimal_prices = [
            bar.close
            * (bar.adj_factor if complete_adjustment and bar.adj_factor is not None else Decimal(1))
            for bar in ordered[-window - 1 :]
        ]
        prices = np.asarray([float(value) for value in decimal_prices], dtype=float)
        if not np.all(np.isfinite(prices)) or np.any(prices <= 0):
            raise DomainError("INVALID_FEATURE_HISTORY", "特征价格须为可计算的有限正数")
        try:
            with np.errstate(over="raise", divide="raise", invalid="raise"):
                returns = np.diff(prices) / prices[:-1]
                # A mathematically exact 20% must pass a >=20% screening threshold.
                momentum = float(decimal_prices[-1] / decimal_prices[0] - 1)
                if not np.isfinite(momentum):
                    raise FloatingPointError("momentum outside float range")
                volatility = float(np.std(returns, ddof=1) * np.sqrt(244))
        except FloatingPointError:
            raise DomainError("NUMERIC_RANGE_EXCEEDED", "价格变化超出可计算范围") from None
    else:
        warnings.append("计算收益特征需要至少21个有效收盘价。")
    amounts = [bar.amount for bar in ordered[-window:] if bar.amount is not None]
    if len(amounts) == window:
        average_amount = sum(amounts, Decimal(0)) / Decimal(window)
    return FeatureSnapshot(
        security_id=latest.security_id if latest else "",
        close=latest.close if latest else None,
        momentum_20d=momentum,
        volatility_20d=volatility,
        average_amount_20d=average_amount,
        sample_count=len(ordered),
        last_session=latest.trade_date if latest else None,
        as_of=latest.available_at if latest else None,
        price_basis=basis,
        warnings=tuple(warnings),
    )
