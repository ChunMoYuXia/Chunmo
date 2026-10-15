from datetime import date, datetime
from typing import Protocol

from portfolio_client.domain.market.calendar import TradingCalendar
from portfolio_client.domain.market.models import (
    CorporateActionRecord,
    DailyBar,
    DatasetSnapshot,
    ImportPreview,
    Security,
)


class DatasetRepositoryPort(Protocol):
    def save_preview(self, preview: ImportPreview) -> None: ...
    def get_preview(self, preview_id: str) -> ImportPreview: ...
    def commit_preview(self, preview: ImportPreview) -> DatasetSnapshot: ...
    def list_datasets(self) -> list[DatasetSnapshot]: ...
    def get_dataset(self, dataset_id: str) -> DatasetSnapshot: ...


class MarketImporterPort(Protocol):
    def preview(
        self,
        content: bytes,
        mapping: dict[str, str],
        calendar_content: bytes | None,
        volume_unit: str,
    ) -> ImportPreview: ...


class MarketDataPort(Protocol):
    def bars(
        self,
        dataset_id: str,
        security_ids: list[str],
        as_of: datetime,
        *,
        start: date | None = None,
        lookback: int | None = None,
    ) -> list[DailyBar]: ...
    def securities(self, dataset_id: str, as_of: datetime) -> list[Security]: ...
    def calendar(self, dataset_id: str) -> TradingCalendar: ...
    def actions(self, dataset_id: str, as_of: datetime) -> list[CorporateActionRecord]: ...


class SourceStorePort(Protocol):
    def put(self, content: bytes) -> str: ...
    def read(self, content_hash: str) -> bytes: ...
