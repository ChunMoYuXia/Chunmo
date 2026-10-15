from dataclasses import dataclass
from datetime import date
from decimal import Decimal
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from portfolio_client.domain.accounting.models import (
    AccountState,
    FeeBreakdown,
    FeeSchedule,
    Fill,
    Position,
)
from portfolio_client.domain.execution.models import Order
from portfolio_client.domain.market.models import KnowledgeMode
from portfolio_client.domain.portfolio.models import Weight

Frequency = Literal["daily", "weekly", "monthly", "yearly"]


class BacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dataset_id: str
    calendar_id: str
    security_ids: list[str] = Field(min_length=1, max_length=200)
    weights: dict[str, Weight]
    frequency: Frequency = "monthly"
    start: date
    end: date
    capital: Decimal = Field(
        default=Decimal("100000"), gt=0, le=Decimal("1e14"), allow_inf_nan=False
    )
    fee_version: str = "cn-stock-default"
    fee_schedules: tuple[FeeSchedule, ...] | None = None
    slippage_bps: Decimal = Field(default=Decimal("8"), ge=0, le=1000, allow_inf_nan=False)
    participation_rate: Decimal = Field(default=Decimal(".05"), gt=0, le=1, allow_inf_nan=False)
    require_price_limits: bool = True
    seed: int = 42
    knowledge_mode: KnowledgeMode = "public_pit"
    benchmark_id: str | None = None
    annualization: int = Field(default=244, ge=1, le=1000)
    risk_free_annual: float = Field(default=0, gt=-1, allow_inf_nan=False)

    @model_validator(mode="after")
    def valid_strategy(self) -> Self:
        if self.start > self.end or len(set(self.security_ids)) != len(self.security_ids):
            raise ValueError("日期范围或重复标的无效")
        if set(self.weights) != set(self.security_ids) or sum(self.weights.values()) > 1:
            raise ValueError("目标权重必须覆盖候选池且总和不超过1")
        if self.fee_version != "cn-stock-default" and not self.fee_schedules:
            raise ValueError("自定义费率版本需要完整费率表")
        return self


@dataclass(frozen=True)
class Progress:
    completed: int
    total: int
    message: str


@dataclass(frozen=True)
class NavPoint:
    session: date
    nav: Decimal
    cash: Decimal
    receivables: Decimal
    stale_securities: tuple[str, ...]


@dataclass(frozen=True)
class HoldingsPoint:
    session: date
    positions: dict[str, Position]


@dataclass(frozen=True)
class Trade:
    order_id: str
    fill: Fill
    fees: FeeBreakdown
    reference_price: Decimal


@dataclass(frozen=True)
class Signal:
    session: date
    eligible_session: date
    orders: tuple[Order, ...]


@dataclass(frozen=True)
class UnfilledOrder:
    order: Order
    reason: str


@dataclass(frozen=True)
class BacktestEvidence:
    dataset_id: str
    calendar_id: str
    engine_version: str
    fee_versions: tuple[str, ...]
    seed: int
    knowledge_mode: KnowledgeMode
    price_basis: str = "raw_execution_and_valuation"


@dataclass(frozen=True)
class BacktestResult:
    request: BacktestRequest
    nav: tuple[NavPoint, ...]
    positions: tuple[HoldingsPoint, ...]
    trades: tuple[Trade, ...]
    signals: tuple[Signal, ...]
    unfilled: tuple[UnfilledOrder, ...]
    costs: Decimal
    slippage_costs: Decimal
    final_account: AccountState
    evidence: BacktestEvidence
    assumptions: tuple[str, ...]
    warnings: tuple[str, ...]
    benchmark: tuple[NavPoint, ...] | None = None
