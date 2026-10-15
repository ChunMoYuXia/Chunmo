from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from portfolio_client.infrastructure.db.migrations import migrate


class Database:
    """Owns connections; each use case owns a short, explicit session/transaction."""

    def __init__(self, url: str) -> None:
        self.engine = create_engine(url, connect_args={"check_same_thread": False, "timeout": 15})

        @event.listens_for(self.engine, "connect")
        def sqlite_settings(connection: Any, _record: Any) -> None:
            cursor = connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()

        self.sessions = sessionmaker(self.engine, expire_on_commit=False)
        try:
            migrate(self.engine)
        except BaseException:
            self.engine.dispose()
            raise

    @contextmanager
    def read(self) -> Iterator[Session]:
        with self.sessions() as session:
            yield session

    @contextmanager
    def transaction(self) -> Iterator[Session]:
        with self.sessions.begin() as session:
            session.connection().exec_driver_sql("BEGIN IMMEDIATE")
            yield session

    def close(self) -> None:
        self.engine.dispose()
