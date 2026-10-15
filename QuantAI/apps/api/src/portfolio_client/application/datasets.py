from collections.abc import Callable
from dataclasses import replace

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.market.models import DatasetSnapshot, ImportPreview, SourceFile
from portfolio_client.domain.market.ports import (
    DatasetRepositoryPort,
    MarketImporterPort,
    SourceStorePort,
)


class DatasetService:
    def __init__(
        self,
        repository: DatasetRepositoryPort,
        importer: MarketImporterPort,
        synthetic_generator: Callable[[int, int, int], ImportPreview],
        source_store: SourceStorePort,
    ) -> None:
        self.repository = repository
        self.importer = importer
        self.synthetic_generator = synthetic_generator
        self.source_store = source_store

    def preview(
        self,
        content: bytes,
        mapping: dict[str, str],
        calendar: bytes | None,
        volume_unit: str,
        name: str,
    ) -> ImportPreview:
        preview = replace(
            self.importer.preview(content, mapping, calendar, volume_unit), name=name[:120]
        )
        sources = [SourceFile("quotes", self.source_store.put(content))]
        if calendar is not None:
            sources.append(SourceFile("calendar", self.source_store.put(calendar)))
        preview = replace(preview, source_files=tuple(sources))
        self.repository.save_preview(preview)
        return preview

    def commit_import(self, preview_id: str) -> DatasetSnapshot:
        preview = self.repository.get_preview(preview_id)
        if preview.errors or not preview.rows:
            raise DomainError("IMPORT_INVALID", "请先修正全部导入错误，当前批次未提交")
        if not preview.calendar_dates:
            raise DomainError("CALENDAR_REQUIRED", "真实行情需同时提供交易日历CSV")
        return self.repository.commit_preview(preview)

    def create_synthetic(self, seed: int, securities: int, sessions: int) -> DatasetSnapshot:
        preview = self.synthetic_generator(seed, securities, sessions)
        self.repository.save_preview(preview)
        return self.commit_import(preview.id)

    def list_datasets(self) -> list[DatasetSnapshot]:
        return self.repository.list_datasets()

    def get_dataset(self, dataset_id: str) -> DatasetSnapshot:
        return self.repository.get_dataset(dataset_id)

    def read_source(self, dataset_id: str, role: str) -> bytes:
        snapshot = self.repository.get_dataset(dataset_id)
        source = next((item for item in snapshot.source_files if item.role == role), None)
        if source is None:
            raise DomainError("SOURCE_NOT_FOUND", "此数据集没有对应原始文件")
        return self.source_store.read(source.content_hash)
