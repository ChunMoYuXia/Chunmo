from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast

from pydantic import TypeAdapter
from sqlalchemy import select
from sqlalchemy.orm import Session

from portfolio_client.domain.accounting.models import LedgerEvent
from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.execution.audit import (
    AuditEntry,
    AuditVerification,
    audit_hash,
    canonical_json,
    verify_audit,
)
from portfolio_client.domain.execution.models import OrderState
from portfolio_client.domain.execution.paper_models import PaperAccount, PaperOrder
from portfolio_client.domain.execution.paper_ports import IdempotencyEntry, PaperUnitOfWork
from portfolio_client.domain.research.backtest_models import Trade
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.models import (
    AuditEventRecord,
    AuditHeadRecord,
    IdempotencyRecord,
    LedgerRecord,
    PaperAccountRecord,
    PaperOrderRecord,
    PaperTradeRecord,
)
from portfolio_client.infrastructure.db.paper_audit import verify_business_rows

ACCOUNT = TypeAdapter(PaperAccount)
ORDER = TypeAdapter(PaperOrder)
TRADE = TypeAdapter(Trade)
LEDGER = TypeAdapter(LedgerEvent)


def _json[T](adapter: TypeAdapter[T], value: T) -> dict[str, Any]:
    return cast(dict[str, Any], adapter.dump_python(value, mode="json"))


class SQLitePaperUnitOfWork:
    def __init__(self, session: Session) -> None:
        self.session = session

    def account(self, account_id: str) -> PaperAccount:
        record = self.session.get(PaperAccountRecord, account_id)
        if record is None:
            raise DomainError("ACCOUNT_NOT_FOUND", "模拟账户不存在")
        return ACCOUNT.validate_python(record.payload)

    def save_account(self, account: PaperAccount) -> None:
        state = account.state
        drained = replace(
            account,
            state=replace(state, events=(), ledger_offset=state.ledger_offset + len(state.events)),
        )
        record = self.session.get(PaperAccountRecord, account.id)
        if record is None:
            record = PaperAccountRecord(
                id=account.id,
                dataset_id=account.dataset_id,
                payload=_json(ACCOUNT, drained),
                created_at=account.created_at.isoformat(),
            )
            self.session.add(record)
        else:
            record.payload = _json(ACCOUNT, drained)
        self.session.flush()
        for entry in state.events:
            self.session.add(
                LedgerRecord(account_id=account.id, seq=entry.seq, payload=_json(LEDGER, entry))
            )

    def order(self, account_id: str, order_id: str) -> PaperOrder:
        record = self.session.get(PaperOrderRecord, order_id)
        if record is None or record.account_id != account_id:
            raise DomainError("ORDER_NOT_FOUND", "该账户没有对应订单")
        return ORDER.validate_python(record.payload)

    def orders(self, account_id: str, *, open_only: bool = False) -> list[PaperOrder]:
        statement = select(PaperOrderRecord).where(PaperOrderRecord.account_id == account_id)
        if open_only:
            statement = statement.where(PaperOrderRecord.state == OrderState.SUBMITTED)
        return [
            ORDER.validate_python(record.payload)
            for record in self.session.scalars(statement.order_by(PaperOrderRecord.sequence))
        ]

    def save_order(self, order: PaperOrder) -> None:
        record = self.session.get(PaperOrderRecord, order.order.order_id)
        if record is None:
            record = PaperOrderRecord(
                id=order.order.order_id,
                account_id=order.order.account_id,
                sequence=order.sequence,
                state=order.order.state.value,
                payload=_json(ORDER, order),
            )
            self.session.add(record)
        else:
            record.payload, record.state = _json(ORDER, order), order.order.state.value
        self.session.flush()

    def add_trade(self, account_id: str, trade: Trade) -> None:
        self.session.add(
            PaperTradeRecord(
                order_id=trade.order_id, account_id=account_id, payload=_json(TRADE, trade)
            )
        )

    def prior(self, scope: str, operation: str, key: str) -> IdempotencyEntry | None:
        record = self.session.get(IdempotencyRecord, (scope, operation, key))
        return (
            IdempotencyEntry(record.request_hash, record.response_json)
            if record is not None
            else None
        )

    def save_response(self, scope: str, operation: str, key: str, entry: IdempotencyEntry) -> None:
        self.session.add(
            IdempotencyRecord(
                scope=scope,
                operation=operation,
                key=key,
                request_hash=entry.request_hash,
                response_json=entry.response_json,
            )
        )

    def audit(self, account_id: str, operation: str, payload: dict[str, Any]) -> None:
        head = self.session.get(AuditHeadRecord, account_id)
        seq, previous = (head.seq + 1, head.event_hash) if head is not None else (1, "0" * 64)
        canonical = canonical_json(
            {
                "schema_version": 1,
                "chain_id": account_id,
                "seq": seq,
                "operation": operation,
                "event_at": datetime.now(UTC).isoformat(),
                "payload": payload,
            }
        )
        digest = audit_hash(previous, canonical)
        self.session.add(
            AuditEventRecord(
                chain_id=account_id,
                seq=seq,
                previous_hash=previous,
                event_hash=digest,
                payload_json=canonical,
            )
        )
        if head is None:
            self.session.add(AuditHeadRecord(chain_id=account_id, seq=seq, event_hash=digest))
        else:
            head.seq, head.event_hash = seq, digest


def _audit(record: AuditEventRecord) -> AuditEntry:
    return AuditEntry(
        record.chain_id, record.seq, record.previous_hash, record.event_hash, record.payload_json
    )


class SQLitePaperRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    @contextmanager
    def read(self) -> Iterator[PaperUnitOfWork]:
        with self.database.read() as session:
            session.connection().exec_driver_sql("BEGIN")
            yield SQLitePaperUnitOfWork(session)

    @contextmanager
    def transaction(self) -> Iterator[PaperUnitOfWork]:
        with self.database.transaction() as session:
            yield SQLitePaperUnitOfWork(session)

    def account(self, account_id: str) -> PaperAccount:
        with self.database.read() as session:
            return SQLitePaperUnitOfWork(session).account(account_id)

    def accounts(self) -> list[PaperAccount]:
        with self.database.read() as session:
            return [
                ACCOUNT.validate_python(record.payload)
                for record in session.scalars(
                    select(PaperAccountRecord).order_by(PaperAccountRecord.created_at.desc())
                )
            ]

    def orders(self, account_id: str) -> list[PaperOrder]:
        with self.database.read() as session:
            SQLitePaperUnitOfWork(session).account(account_id)
            return SQLitePaperUnitOfWork(session).orders(account_id)

    def trades(self, account_id: str) -> list[Trade]:
        with self.database.read() as session:
            SQLitePaperUnitOfWork(session).account(account_id)
            return [
                TRADE.validate_python(record.payload)
                for record in session.scalars(
                    select(PaperTradeRecord)
                    .where(PaperTradeRecord.account_id == account_id)
                    .order_by(PaperTradeRecord.order_id)
                )
            ]

    def ledger(self, account_id: str, after: int, limit: int) -> list[LedgerEvent]:
        with self.database.read() as session:
            SQLitePaperUnitOfWork(session).account(account_id)
            return [
                LEDGER.validate_python(record.payload)
                for record in session.scalars(
                    select(LedgerRecord)
                    .where(LedgerRecord.account_id == account_id, LedgerRecord.seq > after)
                    .order_by(LedgerRecord.seq)
                    .limit(limit)
                )
            ]

    def audit_entries(self, account_id: str, after: int, limit: int) -> list[AuditEntry]:
        with self.database.read() as session:
            SQLitePaperUnitOfWork(session).account(account_id)
            return [
                _audit(record)
                for record in session.scalars(
                    select(AuditEventRecord)
                    .where(AuditEventRecord.chain_id == account_id, AuditEventRecord.seq > after)
                    .order_by(AuditEventRecord.seq)
                    .limit(limit)
                )
            ]

    def verify(self, account_id: str) -> AuditVerification:
        with self.database.read() as session:
            # Keep the head and every event in one SQLite read snapshot during concurrent writes.
            session.connection().exec_driver_sql("BEGIN")
            account = session.get(PaperAccountRecord, account_id)
            if account is None:
                raise DomainError("ACCOUNT_NOT_FOUND", "模拟账户不存在")
            head = session.get(AuditHeadRecord, account_id)
            if head is None:
                return AuditVerification(False, 0, "0" * 64, "missing_head")
            entries = [
                _audit(record)
                for record in session.scalars(
                    select(AuditEventRecord)
                    .where(AuditEventRecord.chain_id == account_id)
                    .order_by(AuditEventRecord.seq)
                )
            ]
            verification = verify_audit(entries, head.seq, head.event_hash)
            if not verification.valid:
                return verification
            reason = verify_business_rows(session, account, entries)
            return AuditVerification(
                reason is None, verification.checked_events, head.event_hash, reason
            )
