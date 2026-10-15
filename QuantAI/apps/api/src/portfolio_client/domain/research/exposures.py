from dataclasses import dataclass
from decimal import Decimal

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import Security


@dataclass(frozen=True)
class ExposureResult:
    total_value: Decimal
    cash_value: Decimal
    receivables: Decimal
    asset_weights: dict[str, float]
    industry_weights: dict[str, float]
    cash_weight: float | None
    receivables_weight: float | None
    max_asset_weight: float | None
    stock_hhi: float | None
    warnings: tuple[str, ...]


def exposures_from_values(
    values: dict[str, Decimal],
    securities: list[Security],
    *,
    cash: Decimal = Decimal("0"),
    receivables: Decimal = Decimal("0"),
) -> ExposureResult:
    if any(not value.is_finite() or value < 0 for value in [cash, receivables, *values.values()]):
        raise DomainError("INVALID_EXPOSURE", "敞口金额必须为非负有限数")
    total = cash + receivables + sum(values.values())
    identities = {item.security_id: item for item in securities}
    assets: dict[str, float] = {}
    industries: dict[str, float] = {}
    unknown = False
    if total:
        for key, value in values.items():
            if not value:
                continue
            assets[key] = float(value / total)
            security = identities.get(key)
            industry = security.industry if security is not None and security.industry else "未分类"
            unknown = unknown or industry == "未分类"
            industries[industry] = industries.get(industry, 0.0) + assets[key]
    warnings = ("部分资产缺少行业分类，保留在未分类敞口中。",) if unknown else ()
    return ExposureResult(
        total,
        cash,
        receivables,
        assets,
        industries,
        float(cash / total) if total else None,
        float(receivables / total) if total else None,
        max(assets.values(), default=0.0) if total else None,
        sum(weight * weight for weight in assets.values()) if total else None,
        warnings,
    )


def exposures(
    holdings: dict[str, int],
    prices: dict[str, Decimal],
    securities: list[Security],
    *,
    cash: Decimal = Decimal("0"),
    receivables: Decimal = Decimal("0"),
) -> ExposureResult:
    values: dict[str, Decimal] = {}
    for key, quantity in holdings.items():
        if type(quantity) is not int or quantity < 0:
            raise DomainError("INVALID_POSITION", "持仓必须为非负整数")
        if quantity:
            price = prices.get(key)
            if price is None or not price.is_finite() or price <= 0:
                raise DomainError("PRICE_NOT_FOUND", "敞口估值缺少有效价格", {"security_id": key})
            values[key] = quantity * price
    return exposures_from_values(values, securities, cash=cash, receivables=receivables)
