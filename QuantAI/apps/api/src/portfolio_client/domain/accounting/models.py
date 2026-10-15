from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal
from enum import StrEnum

from portfolio_client.domain.errors import DomainError

ZERO = Decimal("0")


class Side(StrEnum):
    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True)
class FeeSchedule:
    version: str
    valid_from: date
    valid_to: date | None
    commission_bps: Decimal
    minimum_commission: Decimal
    transfer_bps: Decimal
    stamp_bps: Decimal
    commission_includes_regulatory: bool = True

    def __post_init__(self) -> None:
        values = (self.commission_bps, self.minimum_commission, self.transfer_bps, self.stamp_bps)
        if any(not value.is_finite() or value < 0 for value in values):
            raise DomainError("INVALID_FEES", "费率及最低佣金必须为非负有限数")
        if self.valid_to is not None and self.valid_to < self.valid_from:
            raise DomainError("INVALID_FEES", "费率生效区间无效")


@dataclass(frozen=True)
class FeeBreakdown:
    commission: Decimal
    transfer: Decimal
    stamp: Decimal
    schedule_version: str

    @property
    def total(self) -> Decimal:
        return self.commission + self.transfer + self.stamp


@dataclass(frozen=True)
class Fill:
    security_id: str
    side: Side
    quantity: int
    price: Decimal
    session: date

    def __post_init__(self) -> None:
        if (
            type(self.quantity) is not int
            or self.quantity <= 0
            or not self.price.is_finite()
            or self.price <= 0
        ):
            raise DomainError("INVALID_FILL", "成交必须包含正整数股数与正有限价格")


@dataclass(frozen=True)
class Position:
    quantity: int
    sellable: int
    cost_basis: Decimal

    def __post_init__(self) -> None:
        if (
            self.quantity < 0
            or not 0 <= self.sellable <= self.quantity
            or not self.cost_basis.is_finite()
            or self.cost_basis < 0
        ):
            raise DomainError("INVALID_POSITION", "持仓数量、可卖数量或成本不合法")


@dataclass(frozen=True)
class DividendReceivable:
    security_id: str
    amount: Decimal
    pay_date: date


@dataclass(frozen=True)
class PendingShares:
    security_id: str
    quantity: int
    cost_basis: Decimal
    delivery_date: date


@dataclass(frozen=True)
class LedgerEvent:
    seq: int
    session: date | None
    kind: str
    security_id: str | None = None
    delta_cash: Decimal = ZERO
    delta_frozen: Decimal = ZERO
    delta_receivables: Decimal = ZERO
    delta_quantity: int = 0
    fees: FeeBreakdown | None = None
    realized_pnl: Decimal | None = None
    description: str = ""


@dataclass(frozen=True)
class AccountState:
    cash: Decimal
    frozen_cash: Decimal = ZERO
    withdrawable_cash: Decimal = ZERO
    receivables: Decimal = ZERO
    positions: dict[str, Position] = field(default_factory=dict)
    session: date | None = None
    dividends: dict[str, DividendReceivable] = field(default_factory=dict)
    pending_shares: dict[str, PendingShares] = field(default_factory=dict)
    entitlements: dict[str, int] = field(default_factory=dict)
    processed_actions: tuple[str, ...] = ()
    events: tuple[LedgerEvent, ...] = ()
    ledger_offset: int = 0

    def __post_init__(self) -> None:
        amounts = (self.cash, self.frozen_cash, self.withdrawable_cash, self.receivables)
        if (
            any(not amount.is_finite() or amount < 0 for amount in amounts)
            or self.frozen_cash > self.cash
            or self.withdrawable_cash > self.cash
        ):
            raise DomainError("INVALID_ACCOUNT", "账户资金或冻结金额不合法")

    @property
    def available_cash(self) -> Decimal:
        return self.cash - self.frozen_cash

    @property
    def available_to_withdraw(self) -> Decimal:
        return min(self.withdrawable_cash, self.available_cash)
