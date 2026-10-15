from datetime import date, datetime
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import aliased

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.calendar import TradingCalendar
from portfolio_client.domain.market.clock import as_utc
from portfolio_client.domain.market.models import CorporateActionRecord, DailyBar, Security
from portfolio_client.infrastructure.db.codec import action_from_payload
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.models import (
    ActionRecord,
    BarRecord,
    DatasetRecord,
    SecurityRecord,
)


def bar_from_record(record: BarRecord) -> DailyBar:
    return DailyBar(
        security_id=record.security_id,
        trade_date=record.trade_date,
        open=Decimal(record.open),
        high=Decimal(record.high),
        low=Decimal(record.low),
        close=Decimal(record.close),
        volume=record.volume,
        amount=Decimal(record.amount) if record.amount is not None else None,
        available_at=datetime.fromisoformat(record.available_at),
        suspended=record.suspended,
        limit_up=Decimal(record.limit_up) if record.limit_up is not None else None,
        limit_down=Decimal(record.limit_down) if record.limit_down is not None else None,
        adj_factor=Decimal(record.adj_factor) if record.adj_factor is not None else None,
    )


class SQLiteMarketData:
    def __init__(self, database: Database) -> None:
        self.database = database

    def bars(
        self,
        dataset_id: str,
        security_ids: list[str],
        as_of: datetime,
        *,
        start: date | None = None,
        lookback: int | None = None,
    ) -> list[DailyBar]:
        if not security_ids:
            return []
        cutoff = as_utc(as_of).isoformat()
        base = select(BarRecord).where(
            BarRecord.dataset_id == dataset_id,
            BarRecord.security_id.in_(security_ids),
            BarRecord.available_at <= cutoff,
        )
        if start is not None:
            base = base.where(BarRecord.trade_date >= start)
        if lookback is not None:
            ranked = base.add_columns(
                func.row_number()
                .over(partition_by=BarRecord.security_id, order_by=BarRecord.trade_date.desc())
                .label("position")
            ).subquery()
            record_alias = aliased(BarRecord, ranked)
            statement = (
                select(record_alias)
                .where(ranked.c.position <= lookback)
                .order_by(record_alias.security_id, record_alias.trade_date)
            )
        else:
            statement = base.order_by(BarRecord.security_id, BarRecord.trade_date)
        with self.database.read() as session:
            return [bar_from_record(record) for record in session.scalars(statement)]

    def securities(self, dataset_id: str, as_of: datetime) -> list[Security]:
        statement = (
            select(SecurityRecord)
            .where(
                SecurityRecord.dataset_id == dataset_id,
                SecurityRecord.available_at <= as_utc(as_of).isoformat(),
            )
            .order_by(SecurityRecord.security_id, SecurityRecord.available_at.desc())
        )
        identities: dict[str, Security] = {}
        with self.database.read() as session:
            for record in session.scalars(statement):
                if record.security_id not in identities:
                    identities[record.security_id] = Security(
                        security_id=record.security_id,
                        code=record.code,
                        exchange=record.exchange,
                        name=record.name,
                        industry=record.industry,
                        initials=record.initials,
                        board=record.board,
                        currency=record.currency,
                    )
        return list(identities.values())

    def calendar(self, dataset_id: str) -> TradingCalendar:
        with self.database.read() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise DomainError("DATASET_NOT_FOUND", "数据集不存在")
            return TradingCalendar(tuple(date.fromisoformat(day) for day in record.calendar_dates))

    def actions(self, dataset_id: str, as_of: datetime) -> list[CorporateActionRecord]:
        with self.database.read() as session:
            records = session.scalars(
                select(ActionRecord).where(ActionRecord.dataset_id == dataset_id)
            )
            actions = [action_from_payload(record.payload) for record in records]
        return [action for action in actions if action.announced_at <= as_utc(as_of)]
