from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Literal

from portfolio_client.domain.accounting.models import Fill, Side
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.clock import as_utc


class OrderState(StrEnum):
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    EXPIRED = "expired"
    REJECTED = "rejected"


@dataclass(frozen=True)
class Order:
    order_id: str
    account_id: str
    security_id: str
    side: Side
    quantity: int
    order_type: Literal["market", "limit"]
    limit_price: Decimal | None
    submitted_at: datetime
    eligible_session: date
    state: OrderState

    def __post_init__(self) -> None:
        as_utc(self.submitted_at)
        if type(self.quantity) is not int or self.quantity <= 0:
            raise DomainError("INVALID_ORDER", "订单股数必须为正整数")
        if self.order_type == "limit" and (
            self.limit_price is None or not self.limit_price.is_finite() or self.limit_price <= 0
        ):
            raise DomainError("INVALID_ORDER", "限价单需要正有限限价")
        if self.order_type == "market" and self.limit_price is not None:
            raise DomainError("INVALID_ORDER", "市价单不能附加限价")


@dataclass(frozen=True)
class KnownLiquidity:
    average_daily_volume: Decimal
    as_of: datetime
    consumed_quantity: int = 0


@dataclass(frozen=True)
class MatchResult:
    status: Literal["filled", "partially_filled", "unfilled"]
    fill: Fill | None
    reason: str | None = None
