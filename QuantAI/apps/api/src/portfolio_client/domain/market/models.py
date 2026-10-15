from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Literal

SourceKind = Literal["synthetic", "imported"]
KnowledgeMode = Literal["public_pit", "system_replay"]


@dataclass(frozen=True)
class Security:
    security_id: str
    code: str
    exchange: str
    name: str
    industry: str | None
    initials: str
    board: str = "main"
    currency: str = "CNY"


@dataclass(frozen=True)
class DailyBar:
    security_id: str
    trade_date: date
    open: Decimal
    high: Decimal
    low: Decimal
    close: Decimal
    volume: int
    amount: Decimal | None
    available_at: datetime
    suspended: bool = False
    limit_up: Decimal | None = None
    limit_down: Decimal | None = None
    adj_factor: Decimal | None = None


@dataclass(frozen=True)
class CorporateActionRecord:
    event_id: str
    security_id: str
    kind: Literal["dividend", "bonus"]
    record_date: date
    ex_date: date
    pay_date: date
    announced_at: datetime
    cash_per_share: Decimal = Decimal("0")
    bonus_ratio: Decimal = Decimal("0")


@dataclass(frozen=True)
class ImportRow:
    security: Security
    bar: DailyBar
    line_number: int

    @property
    def code(self) -> str:
        return self.security.code


@dataclass(frozen=True)
class ImportIssue:
    row: int
    code: str
    message: str


@dataclass(frozen=True)
class SourceFile:
    role: Literal["quotes", "calendar"]
    content_hash: str


@dataclass(frozen=True)
class ImportPreview:
    id: str
    rows: tuple[ImportRow, ...]
    errors: tuple[ImportIssue, ...]
    warnings: tuple[str, ...]
    content_hash: str
    source_kind: SourceKind
    name: str = "导入行情"
    calendar_dates: tuple[date, ...] = ()
    actions: tuple[CorporateActionRecord, ...] = ()
    volume_unit: str = "shares"
    source_files: tuple[SourceFile, ...] = ()
    generator_seed: int | None = None
    generator_version: str | None = None
    parser_version: str | None = None
    column_mapping: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class DatasetSnapshot:
    dataset_id: str
    version: int
    calendar_id: str
    source_kind: SourceKind
    name: str
    content_hash: str
    row_count: int
    security_count: int
    first_session: date
    last_session: date
    ingested_at: datetime
    warnings: tuple[str, ...] = field(default_factory=tuple)
    source_files: tuple[SourceFile, ...] = ()
    generator_seed: int | None = None
    generator_version: str | None = None
    parser_version: str | None = None
    column_mapping: tuple[tuple[str, str], ...] = ()
    volume_unit: str = "shares"
