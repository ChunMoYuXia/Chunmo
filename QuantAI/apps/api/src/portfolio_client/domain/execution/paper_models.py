from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from portfolio_client.domain.accounting.models import AccountState, FeeSchedule, Position, Side
from portfolio_client.domain.execution.models import Order
from portfolio_client.domain.market.models import SourceKind
from portfolio_client.domain.research.backtest_models import Trade


class CreateAccountRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)
    name: str = Field(min_length=1, max_length=80)
    dataset_id: str
    capital: Decimal = Field(gt=0, le=Decimal("1e14"), allow_inf_nan=False)
    start: date | None = None
    slippage_bps: Decimal = Field(default=Decimal("8"), ge=0, le=1000, allow_inf_nan=False)
    participation_rate: Decimal = Field(default=Decimal(".05"), gt=0, le=1, allow_inf_nan=False)
    require_price_limits: bool = True
    fee_schedules: tuple[FeeSchedule, ...] | None = None

    @field_validator("capital", "slippage_bps", "participation_rate")
    @classmethod
    def canonical_decimal(cls, value: Decimal) -> Decimal:
        return value.normalize()


class OrderRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    security_id: str
    side: Side
    quantity: int = Field(gt=0, le=2_147_483_647, strict=True)
    order_type: Literal["market", "limit"] = "market"
    limit_price: Decimal | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("limit_price")
    @classmethod
    def canonical_price(cls, value: Decimal | None) -> Decimal | None:
        return value.normalize() if value is not None else None

    @model_validator(mode="after")
    def valid_limit(self) -> Self:
        if (self.order_type == "limit") != (self.limit_price is not None):
            raise ValueError("限价单必须提供限价，市价单不得提供限价")
        return self


class AdvanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    session: date | None = None


class ResumeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    confirmation: str


class EmptyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CancelRequest(BaseModel):
    order_id: str


@dataclass(frozen=True)
class PaperAccount:
    id: str
    name: str
    dataset_id: str
    calendar_id: str
    data_version: int
    source_kind: SourceKind
    status: Literal["active", "paused"]
    state: AccountState
    fee_schedules: tuple[FeeSchedule, ...]
    slippage_bps: Decimal
    participation_rate: Decimal
    require_price_limits: bool
    created_at: datetime


@dataclass(frozen=True)
class PaperOrder:
    order: Order
    sequence: int
    reserved_cash: Decimal
    filled_quantity: int = 0
    reason: str | None = None


@dataclass(frozen=True)
class AccountSnapshot:
    id: str
    name: str
    dataset_id: str
    calendar_id: str
    data_version: int
    source_kind: SourceKind
    session: date
    status: Literal["active", "paused"]
    cash: Decimal
    frozen_cash: Decimal
    available_cash: Decimal
    withdrawable_cash: Decimal
    receivables: Decimal
    positions: dict[str, Position]
    reserved_shares: dict[str, int]
    available_to_sell: dict[str, int]
    nav: Decimal
    stale_securities: tuple[str, ...]
    ledger_count: int
    assumptions: tuple[str, ...]


@dataclass(frozen=True)
class HaltResult:
    account: AccountSnapshot
    canceled_order_ids: tuple[str, ...]
    unknown_order_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class PaperTransition:
    account: PaperAccount
    orders: tuple[PaperOrder, ...] = ()
    trades: tuple[Trade, ...] = ()
