import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from itertools import batched
from typing import Any, Literal, cast

from sqlalchemy import func, insert, select

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import (
    DatasetSnapshot,
    ImportPreview,
    SourceFile,
    SourceKind,
)
from portfolio_client.infrastructure.db.codec import preview_from_payload, preview_payload
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.db.models import (
    ActionRecord,
    BarRecord,
    DatasetRecord,
    ImportPreviewRecord,
    SecurityRecord,
)


def dataset_snapshot(record: DatasetRecord) -> DatasetSnapshot:
    return DatasetSnapshot(
        dataset_id=record.dataset_id,
        version=record.version,
        calendar_id=record.calendar_id,
        source_kind=cast(SourceKind, record.source_kind),
        name=record.name,
        content_hash=record.content_hash,
        row_count=record.row_count,
        security_count=record.security_count,
        first_session=record.first_session,
        last_session=record.last_session,
        ingested_at=datetime.fromisoformat(record.ingested_at),
        warnings=tuple(record.warnings),
        source_files=tuple(
            SourceFile(
                role=cast(Literal["quotes", "calendar"], item["role"]),
                content_hash=item["content_hash"],
            )
            for item in record.source_files
        ),
        generator_seed=record.generator_seed,
        generator_version=record.generator_version,
        parser_version=record.parser_version,
        column_mapping=tuple((item[0], item[1]) for item in record.column_mapping),
        volume_unit=record.volume_unit,
    )


class SQLiteDatasetRepository:
    def __init__(self, database: Database) -> None:
        self.database = database

    def save_preview(self, preview: ImportPreview) -> None:
        with self.database.transaction() as session:
            session.add(ImportPreviewRecord(id=preview.id, payload=preview_payload(preview)))

    def get_preview(self, preview_id: str) -> ImportPreview:
        with self.database.read() as session:
            record = session.get(ImportPreviewRecord, preview_id)
            if record is None:
                raise DomainError("PREVIEW_NOT_FOUND", "导入预检不存在")
            return preview_from_payload(record.payload)

    def get_dataset(self, dataset_id: str) -> DatasetSnapshot:
        with self.database.read() as session:
            record = session.get(DatasetRecord, dataset_id)
            if record is None:
                raise DomainError("DATASET_NOT_FOUND", "数据集不存在")
            return dataset_snapshot(record)

    def list_datasets(self) -> list[DatasetSnapshot]:
        with self.database.read() as session:
            statement = select(DatasetRecord).order_by(DatasetRecord.version.desc())
            return [dataset_snapshot(record) for record in session.scalars(statement)]

    def commit_preview(self, preview: ImportPreview) -> DatasetSnapshot:
        with self.database.transaction() as session:
            existing = session.scalar(
                select(DatasetRecord).where(DatasetRecord.content_hash == preview.content_hash)
            )
            if existing is not None:
                return dataset_snapshot(existing)
            version = (session.scalar(select(func.max(DatasetRecord.version))) or 0) + 1
            dataset_id = f"ds_{preview.content_hash[:24]}"
            calendar_payload = (
                preview.source_kind
                + "|"
                + "|".join(day.isoformat() for day in preview.calendar_dates)
            )
            calendar_id = f"cal_{hashlib.sha256(calendar_payload.encode()).hexdigest()[:24]}"
            record = DatasetRecord(
                dataset_id=dataset_id,
                version=version,
                calendar_id=calendar_id,
                source_kind=preview.source_kind,
                name=preview.name,
                content_hash=preview.content_hash,
                row_count=len(preview.rows),
                security_count=len({row.security.security_id for row in preview.rows}),
                first_session=min(row.bar.trade_date for row in preview.rows),
                last_session=max(row.bar.trade_date for row in preview.rows),
                ingested_at=datetime.now(UTC).isoformat(),
                calendar_dates=[day.isoformat() for day in preview.calendar_dates],
                warnings=list(preview.warnings),
                source_files=[asdict(item) for item in preview.source_files],
                generator_seed=preview.generator_seed,
                generator_version=preview.generator_version,
                parser_version=preview.parser_version,
                column_mapping=[list(item) for item in preview.column_mapping],
                volume_unit=preview.volume_unit,
            )
            session.add(record)
            session.flush()
            # Persist metadata changes at their observation time, never one future master row.
            last_metadata: dict[str, dict[str, Any]] = {}
            for row in sorted(
                preview.rows, key=lambda item: (item.bar.trade_date, item.security.security_id)
            ):
                identity = asdict(row.security)
                if last_metadata.get(row.security.security_id) != identity:
                    session.add(
                        SecurityRecord(
                            dataset_id=dataset_id,
                            valid_from=row.bar.trade_date,
                            available_at=row.bar.available_at.isoformat(),
                            **identity,
                        )
                    )
                    last_metadata[row.security.security_id] = identity
            session.flush()
            for batch in batched(preview.rows, 1000):
                values: list[dict[str, Any]] = []
                for row in batch:
                    bar = row.bar
                    values.append(
                        {
                            "dataset_id": dataset_id,
                            "security_id": bar.security_id,
                            "trade_date": bar.trade_date,
                            "open": str(bar.open),
                            "high": str(bar.high),
                            "low": str(bar.low),
                            "close": str(bar.close),
                            "volume": bar.volume,
                            "amount": str(bar.amount) if bar.amount is not None else None,
                            "available_at": bar.available_at.isoformat(),
                            "suspended": bar.suspended,
                            "limit_up": str(bar.limit_up) if bar.limit_up is not None else None,
                            "limit_down": str(bar.limit_down)
                            if bar.limit_down is not None
                            else None,
                            "adj_factor": str(bar.adj_factor)
                            if bar.adj_factor is not None
                            else None,
                        }
                    )
                session.execute(insert(BarRecord), values)
            for action in preview.actions:
                payload = cast(dict[str, Any], json.loads(json.dumps(asdict(action), default=str)))
                session.add(
                    ActionRecord(dataset_id=dataset_id, event_id=action.event_id, payload=payload)
                )
            return dataset_snapshot(record)
