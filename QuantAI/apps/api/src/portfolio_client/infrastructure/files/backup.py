import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from portfolio_client.domain.errors import DomainError
from portfolio_client.infrastructure.db.migrations import SCHEMA_VERSION
from portfolio_client.infrastructure.files.workspace_lock import workspace_lock


@dataclass(frozen=True)
class BackupReceipt:
    directory: Path
    source_files: int
    schema_version: int


def create_backup(data_dir: Path, output: Path) -> BackupReceipt:
    data_dir, output = data_dir.resolve(), output.resolve()
    if output.is_relative_to(data_dir):
        raise DomainError("INVALID_BACKUP_DESTINATION", "备份目标必须在运行数据目录之外")
    database = data_dir / "portfolio.db"
    if not database.is_file():
        raise DomainError("BACKUP_DATA_NOT_FOUND", "数据目录中没有可备份的数据库")
    with workspace_lock(data_dir):
        sources = data_dir / "sources"
        files = list(sources.rglob("*")) if sources.exists() else []
        if sources.is_symlink() or any(file.is_symlink() for file in files):
            raise DomainError("UNSUPPORTED_BACKUP_SOURCE", "来源目录含符号链接，请先整理为普通文件")
        destination = output / (
            datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8]
        )
        destination.mkdir(parents=True)
        with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as source:
            version = source.execute("SELECT version FROM schema_version").fetchone()
            if version != (SCHEMA_VERSION,):
                raise DomainError("SCHEMA_VERSION_MISMATCH", "请使用匹配数据库版本的软件执行备份")
            with closing(sqlite3.connect(destination / "portfolio.db")) as target:
                source.backup(target)
        if sources.exists():
            shutil.copytree(sources, destination / "sources")
        else:
            (destination / "sources").mkdir()
        count = sum(file.is_file() for file in files)
        with (destination / "portfolio.db").open("rb") as backed_up:
            checksum = hashlib.file_digest(backed_up, "sha256").hexdigest()
        (destination / "backup.json").write_text(
            json.dumps(
                {
                    "state": "complete",
                    "schema_version": SCHEMA_VERSION,
                    "source_files": count,
                    "database_sha256": checksum,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return BackupReceipt(destination, count, SCHEMA_VERSION)
