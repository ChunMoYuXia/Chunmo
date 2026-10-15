from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.portfolio.models import PortfolioSpec, SavedPortfolio
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.models import PortfolioRecord


def _portfolio(record: PortfolioRecord) -> SavedPortfolio:
    return SavedPortfolio(
        record.id,
        PortfolioSpec.model_validate(record.spec_payload),
        datetime.fromisoformat(record.created_at),
    )


class SQLitePortfolios:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, spec: PortfolioSpec) -> SavedPortfolio:
        with self.database.transaction() as session:
            record = PortfolioRecord(
                id=uuid4().hex,
                dataset_id=spec.dataset_id,
                spec_payload=spec.model_dump(mode="json"),
                created_at=datetime.now(UTC).isoformat(),
            )
            session.add(record)
            session.flush()
            return _portfolio(record)

    def get(self, portfolio_id: str) -> SavedPortfolio:
        with self.database.read() as session:
            record = session.get(PortfolioRecord, portfolio_id)
            if record is None:
                raise DomainError("PORTFOLIO_NOT_FOUND", "组合不存在")
            return _portfolio(record)

    def list(self) -> list[SavedPortfolio]:
        with self.database.read() as session:
            return [
                _portfolio(record)
                for record in session.scalars(
                    select(PortfolioRecord).order_by(PortfolioRecord.created_at.desc())
                )
            ]
