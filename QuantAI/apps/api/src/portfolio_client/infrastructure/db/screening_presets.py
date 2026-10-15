from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import select

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.screening.dsl import ScreeningSpec
from portfolio_client.domain.screening.models import SavedScreening
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.models import ScreeningPresetRecord


def _saved(record: ScreeningPresetRecord) -> SavedScreening:
    return SavedScreening(
        id=record.id,
        name=record.name,
        spec=ScreeningSpec.model_validate(record.spec_payload),
        created_at=datetime.fromisoformat(record.created_at),
    )


class SQLiteScreeningPresets:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save(self, name: str, spec: ScreeningSpec) -> SavedScreening:
        with self.database.transaction() as session:
            record = ScreeningPresetRecord(
                id=uuid4().hex,
                dataset_id=spec.dataset_id,
                name=name,
                spec_payload=spec.model_dump(mode="json"),
                created_at=datetime.now(UTC).isoformat(),
            )
            session.add(record)
            session.flush()
            return _saved(record)

    def get(self, preset_id: str) -> SavedScreening:
        with self.database.read() as session:
            record = session.get(ScreeningPresetRecord, preset_id)
            if record is None:
                raise DomainError("PRESET_NOT_FOUND", "选股方案不存在")
            return _saved(record)

    def list(self) -> list[SavedScreening]:
        with self.database.read() as session:
            records = session.scalars(
                select(ScreeningPresetRecord).order_by(ScreeningPresetRecord.created_at.desc())
            )
            return [_saved(record) for record in records]
