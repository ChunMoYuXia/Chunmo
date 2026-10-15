from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import event, func, select

from portfolio_client.config import Settings
from portfolio_client.main import create_app


def test_dataset_survives_application_restart(
    client: TestClient, settings: Settings, auth: dict[str, str]
) -> None:
    response = client.post(
        "/v1/datasets/synthetic", json={"seed": 42, "securities": 3, "sessions": 30}, headers=auth
    )
    assert response.status_code == 201
    dataset = response.json()["data"]
    assert dataset["row_count"] == 90
    assert dataset["source_kind"] == "synthetic"
    assert dataset["generator_seed"] == 42
    with TestClient(create_app(settings)) as reopened:
        datasets = reopened.get("/v1/datasets", headers=auth).json()["data"]
    assert datasets[0]["dataset_id"] == dataset["dataset_id"]
    repeated = client.post(
        "/v1/datasets/synthetic", json={"seed": 42, "securities": 3, "sessions": 30}, headers=auth
    ).json()["data"]
    assert repeated["dataset_id"] == dataset["dataset_id"]


def test_invalid_import_cannot_commit_partial_dataset(
    client: TestClient, auth: dict[str, str]
) -> None:
    content = b"code,date,open,high,low,close,volume\n000001,2026-01-05,12,10,11,12,1000\n"
    response = client.post(
        "/v1/datasets/import/preview", files={"file": ("invalid.csv", content)}, headers=auth
    )
    assert response.status_code == 200
    preview = response.json()["data"]
    commit = client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth)
    assert commit.status_code == 422
    assert client.get("/v1/datasets", headers=auth).json()["data"] == []


def test_import_requires_calendar_and_returns_decimal_strings(
    client: TestClient, auth: dict[str, str]
) -> None:
    content = b"code,date,open,high,low,close,volume\n000001,2026-01-05,10,11,9,10.50,1000\n"
    preview = client.post(
        "/v1/datasets/import/preview", files={"file": ("quotes.csv", content)}, headers=auth
    ).json()["data"]
    assert preview["rows"][0]["close"] == "10.50"
    assert (
        client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).status_code == 422
    )
    assert (
        "CALENDAR_REQUIRED"
        in client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).text
    )


def test_calendar_mismatch_blocks_commit(client: TestClient, auth: dict[str, str]) -> None:
    quotes = b"code,date,open,high,low,close,volume\n000001,2026-01-05,10,11,9,10,1000\n"
    response = client.post(
        "/v1/datasets/import/preview",
        files={
            "file": ("quotes.csv", quotes),
            "calendar_file": ("calendar.csv", b"date\n2026-01-06\n"),
        },
        headers=auth,
    )
    preview = response.json()["data"]
    assert preview["errors"][0]["code"] == "UNKNOWN_SESSION"
    assert (
        client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).status_code == 422
    )


def test_valid_calendar_commits_and_preserves_exchange_identity(
    client: TestClient, auth: dict[str, str]
) -> None:
    quotes = b"code,exchange,date,open,high,low,close,volume\n000001,XSHE,2026-01-05,10,11,9,10,1000\n000001,XSHG,2026-01-05,10,11,9,10,1000\n"
    preview = client.post(
        "/v1/datasets/import/preview",
        files={
            "file": ("quotes.csv", quotes),
            "calendar_file": ("calendar.csv", b"date\n2026-01-05\n"),
        },
        headers=auth,
    ).json()["data"]
    assert preview["can_commit"] is True
    response = client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth)
    assert response.status_code == 201
    assert response.json()["data"]["security_count"] == 2


def test_failed_bar_insert_rolls_back_dataset_and_security_rows(tmp_path: Path) -> None:
    from portfolio_client.infrastructure.db.database import Database
    from portfolio_client.infrastructure.db.models import SecurityRecord
    from portfolio_client.infrastructure.db.repositories import SQLiteDatasetRepository
    from portfolio_client.infrastructure.imports.synthetic import generate_dataset

    database = Database(f"sqlite:///{tmp_path}/atomic.db")
    repository = SQLiteDatasetRepository(database)
    preview = generate_dataset(1, 2, 5)

    def fail_bar_insert(_connection, _cursor, statement, _parameters, _context, _many):
        if "INSERT INTO daily_bars" in statement:
            raise RuntimeError("simulated storage failure")

    event.listen(database.engine, "before_cursor_execute", fail_bar_insert)
    with pytest.raises(RuntimeError, match="simulated storage failure"):
        repository.commit_preview(preview)
    assert repository.list_datasets() == []
    with database.read() as session:
        assert session.scalar(select(func.count()).select_from(SecurityRecord)) == 0
    database.close()


def test_oversized_upload_is_rejected_before_parsing(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post(
        "/v1/datasets/import/preview",
        content=b"",
        headers={**auth, "Content-Length": str(53 * 1024 * 1024)},
    )
    assert response.status_code == 413


def test_original_import_bytes_remain_available_after_restart(
    client: TestClient, settings: Settings, auth: dict[str, str]
) -> None:
    quotes = b"code,date,open,high,low,close,volume\n000001,2026-01-05,10,11,9,10.50,1000\n"
    calendar = b"date\n2026-01-05\n"
    preview = client.post(
        "/v1/datasets/import/preview",
        files={"file": ("quotes.csv", quotes), "calendar_file": ("calendar.csv", calendar)},
        headers=auth,
    ).json()["data"]
    dataset = client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).json()[
        "data"
    ]
    with TestClient(create_app(settings)) as reopened:
        response = reopened.get(
            f"/v1/datasets/{dataset['dataset_id']}/sources/quotes", headers=auth
        )
    assert response.status_code == 200
    assert response.content == quotes
