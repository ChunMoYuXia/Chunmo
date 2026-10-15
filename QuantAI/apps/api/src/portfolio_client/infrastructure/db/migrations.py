from sqlalchemy import Engine, insert, inspect, select

from portfolio_client.domain.errors import DomainError
from portfolio_client.infrastructure.db.models import Base, SchemaVersion

SCHEMA_VERSION = 1


def migrate(engine: Engine) -> None:
    """Initial additive migration. Future revisions must explicitly upgrade this version."""
    with engine.begin() as connection:
        tables = inspect(connection).get_table_names()
        version = (
            connection.execute(select(SchemaVersion.version)).scalar_one_or_none()
            if "schema_version" in tables
            else None
        )
        if (version is not None and version != SCHEMA_VERSION) or (
            version is None and tables and "schema_version" not in tables
        ):
            raise DomainError("SCHEMA_VERSION_MISMATCH", "数据库版本与程序不兼容，请先备份并迁移")
        connection.exec_driver_sql("PRAGMA journal_mode=WAL")
        Base.metadata.create_all(connection)
        version = connection.execute(select(SchemaVersion.version)).scalar_one_or_none()
        if version is None:
            connection.execute(insert(SchemaVersion).values(version=SCHEMA_VERSION))
        elif version != SCHEMA_VERSION:
            raise DomainError("SCHEMA_VERSION_MISMATCH", "数据库版本与程序不兼容，请先备份并迁移")
