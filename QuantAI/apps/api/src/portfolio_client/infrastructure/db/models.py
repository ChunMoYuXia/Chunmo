from datetime import date
from typing import Any

from sqlalchemy import JSON, Boolean, Date, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class SchemaVersion(Base):
    __tablename__ = "schema_version"
    version: Mapped[int] = mapped_column(Integer, primary_key=True)


class AIRunRecord(Base):
    __tablename__ = "ai_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    idempotency_key: Mapped[str] = mapped_column(String, unique=True)
    request_hash: Mapped[str]
    state: Mapped[str]
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ImportPreviewRecord(Base):
    __tablename__ = "import_previews"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class DatasetRecord(Base):
    __tablename__ = "datasets"
    dataset_id: Mapped[str] = mapped_column(String, primary_key=True)
    version: Mapped[int]
    calendar_id: Mapped[str]
    source_kind: Mapped[str]
    name: Mapped[str]
    content_hash: Mapped[str] = mapped_column(String, unique=True)
    row_count: Mapped[int]
    security_count: Mapped[int]
    first_session: Mapped[date] = mapped_column(Date)
    last_session: Mapped[date] = mapped_column(Date)
    ingested_at: Mapped[str]
    calendar_dates: Mapped[list[str]] = mapped_column(JSON)
    warnings: Mapped[list[str]] = mapped_column(JSON)
    source_files: Mapped[list[dict[str, str]]] = mapped_column(JSON)
    generator_seed: Mapped[int | None]
    generator_version: Mapped[str | None]
    parser_version: Mapped[str | None]
    column_mapping: Mapped[list[list[str]]] = mapped_column(JSON)
    volume_unit: Mapped[str]


class SecurityRecord(Base):
    __tablename__ = "securities"
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"), primary_key=True)
    security_id: Mapped[str] = mapped_column(String, primary_key=True)
    valid_from: Mapped[date] = mapped_column(Date, primary_key=True)
    available_at: Mapped[str]
    code: Mapped[str]
    exchange: Mapped[str]
    name: Mapped[str]
    industry: Mapped[str | None]
    initials: Mapped[str]
    board: Mapped[str]
    currency: Mapped[str]


class BarRecord(Base):
    __tablename__ = "daily_bars"
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"), primary_key=True)
    security_id: Mapped[str] = mapped_column(String, primary_key=True)
    trade_date: Mapped[date] = mapped_column(Date, primary_key=True)
    # SQLite NUMERIC can round-trip through binary floats: retain exact decimal text.
    open: Mapped[str]
    high: Mapped[str]
    low: Mapped[str]
    close: Mapped[str]
    volume: Mapped[int]
    amount: Mapped[str | None]
    available_at: Mapped[str]
    suspended: Mapped[bool] = mapped_column(Boolean)
    limit_up: Mapped[str | None]
    limit_down: Mapped[str | None]
    adj_factor: Mapped[str | None]


class ActionRecord(Base):
    __tablename__ = "corporate_actions"
    __table_args__ = (UniqueConstraint("dataset_id", "event_id"),)
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"))
    event_id: Mapped[str]
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class ScreeningPresetRecord(Base):
    __tablename__ = "screening_presets"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"))
    name: Mapped[str]
    spec_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str]


class PortfolioRecord(Base):
    __tablename__ = "portfolios"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"))
    spec_payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str]


class PaperAccountRecord(Base):
    __tablename__ = "paper_accounts"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str]


class PaperOrderRecord(Base):
    __tablename__ = "paper_orders"
    __table_args__ = (UniqueConstraint("account_id", "sequence"),)
    id: Mapped[str] = mapped_column(String, primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    sequence: Mapped[int]
    state: Mapped[str]
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class PaperTradeRecord(Base):
    __tablename__ = "paper_trades"
    order_id: Mapped[str] = mapped_column(ForeignKey("paper_orders.id"), primary_key=True)
    account_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class LedgerRecord(Base):
    __tablename__ = "ledger_events"
    account_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)


class IdempotencyRecord(Base):
    __tablename__ = "idempotency_records"
    scope: Mapped[str] = mapped_column(String, primary_key=True)
    operation: Mapped[str] = mapped_column(String, primary_key=True)
    key: Mapped[str] = mapped_column(String, primary_key=True)
    request_hash: Mapped[str]
    response_json: Mapped[str]


class AuditEventRecord(Base):
    __tablename__ = "audit_events"
    chain_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), primary_key=True)
    seq: Mapped[int] = mapped_column(Integer, primary_key=True)
    previous_hash: Mapped[str]
    event_hash: Mapped[str]
    payload_json: Mapped[str]


class AuditHeadRecord(Base):
    __tablename__ = "audit_heads"
    chain_id: Mapped[str] = mapped_column(ForeignKey("paper_accounts.id"), primary_key=True)
    seq: Mapped[int]
    event_hash: Mapped[str]


class ResearchJobRecord(Base):
    __tablename__ = "research_jobs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    state: Mapped[str] = mapped_column(String, index=True)
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str]


class ResearchResultRecord(Base):
    __tablename__ = "research_results"
    id: Mapped[str] = mapped_column(ForeignKey("research_jobs.id"), primary_key=True)
    dataset_id: Mapped[str] = mapped_column(ForeignKey("datasets.dataset_id"))
    payload: Mapped[dict[str, Any]] = mapped_column(JSON)
    summary: Mapped[dict[str, Any]] = mapped_column(JSON)
    created_at: Mapped[str]
