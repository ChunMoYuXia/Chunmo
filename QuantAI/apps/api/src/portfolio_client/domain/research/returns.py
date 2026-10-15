from dataclasses import dataclass
from datetime import date

import numpy as np
from numpy.typing import NDArray

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import DailyBar


@dataclass(frozen=True)
class ReturnMatrix:
    values: NDArray[np.float64]
    price_basis: str
    warnings: tuple[str, ...]


def aligned_returns(
    bars: list[DailyBar], security_ids: list[str], sessions: list[date]
) -> ReturnMatrix:
    if len(sessions) < 3:
        raise DomainError("INSUFFICIENT_HISTORY", "缺少完整交易日收益窗口")
    by_key = {(bar.security_id, bar.trade_date): bar for bar in bars}
    if any((security_id, day) not in by_key for day in sessions for security_id in security_ids):
        raise DomainError("INSUFFICIENT_HISTORY", "收益窗口存在缺失交易日，请缩短窗口或补充数据")
    selected = [by_key[(security_id, day)] for day in sessions for security_id in security_ids]
    adjusted = [bar.adj_factor is not None for bar in selected]
    if any(adjusted) and not all(adjusted):
        raise DomainError("INCOMPLETE_ADJUSTMENT", "复权因子不完整，无法生成一致收益矩阵")
    prices = np.array(
        [
            float(
                bar.close * (bar.adj_factor if all(adjusted) and bar.adj_factor is not None else 1)
            )
            for bar in selected
        ]
    ).reshape((len(sessions), len(security_ids)))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        values = prices[1:] / prices[:-1] - 1
    if not np.all(np.isfinite(values)):
        raise DomainError("INVALID_RETURNS", "价格尺度产生非有限收益")
    return ReturnMatrix(
        values,
        "adjusted" if all(adjusted) else "raw",
        () if all(adjusted) else ("使用原始价格收益，除权除息可能影响风险估计。",),
    )
