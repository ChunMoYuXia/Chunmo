import sqlite3

import pytest
from fastapi.testclient import TestClient

from portfolio_client.domain.errors import DomainError
from portfolio_client.infrastructure.db.database import Database
from portfolio_client.infrastructure.files.workspace_lock import workspace_lock
from portfolio_client.main import create_app


def test_unknown_schema_version_is_rejected_without_creating_or_mutating_tables(tmp_path):
    path = tmp_path / "future.db"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE schema_version(version INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO schema_version VALUES(99)")
        connection.execute("CREATE TABLE future_data(value TEXT)")
        connection.execute("INSERT INTO future_data VALUES('preserve')")
    with pytest.raises(DomainError) as error:
        Database(f"sqlite:///{path}")
    assert error.value.code == "SCHEMA_VERSION_MISMATCH"
    with sqlite3.connect(path) as connection:
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall() == [("future_data",), ("schema_version",)]
        assert connection.execute("SELECT value FROM future_data").fetchone() == ("preserve",)
        assert connection.execute("PRAGMA journal_mode").fetchone() == ("delete",)


def test_offline_backup_preserves_original_sources_and_account_audit(auth, settings, tmp_path):
    settings = settings.model_copy(update={"database_url": f"sqlite:///{tmp_path}/portfolio.db"})
    with TestClient(create_app(settings)) as client:
        from portfolio_client.infrastructure.files.backup import create_backup

        quotes = b"code,date,open,high,low,close,volume\n600001,2024-01-02,10,10,10,10,100000\n"
        preview = client.post(
            "/v1/datasets/import/preview",
            headers=auth,
            files={
                "file": ("quotes.csv", quotes, "text/csv"),
                "calendar_file": ("calendar.csv", b"date\n2024-01-02\n", "text/csv"),
            },
        ).json()["data"]
        dataset = client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).json()[
            "data"
        ]
        account = client.post(
            "/v1/paper/accounts",
            headers={**auth, "Idempotency-Key": "backup-account"},
            json={
                "dataset_id": dataset["dataset_id"],
                "start": "2024-01-02",
                "name": "backup",
                "capital": "1000",
            },
        ).json()["data"]
        with workspace_lock(settings.data_dir), pytest.raises(DomainError) as error:
            create_backup(settings.data_dir, tmp_path.parent / (tmp_path.name + "-backups"))
        assert error.value.code == "DATA_DIRECTORY_IN_USE"
        receipt = create_backup(settings.data_dir, tmp_path.parent / (tmp_path.name + "-backups"))
        assert receipt.source_files == 2
        restored = settings.model_copy(
            update={
                "database_url": f"sqlite:///{receipt.directory}/portfolio.db",
                "data_dir": receipt.directory,
            }
        )
        with TestClient(create_app(restored)) as reopened:
            assert (
                reopened.get("/v1/paper/accounts/" + account["id"], headers=auth).json()["data"]
                == account
            )
            assert (
                reopened.get(
                    "/v1/paper/accounts/" + account["id"] + "/audit/verify", headers=auth
                ).json()["data"]["valid"]
                is True
            )
            assert (
                reopened.get(
                    "/v1/datasets/" + dataset["dataset_id"] + "/sources/quotes", headers=auth
                ).content
                == quotes
            )
