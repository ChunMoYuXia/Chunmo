from collections.abc import Iterator
from pathlib import Path
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from portfolio_client.config import Settings
from portfolio_client.main import create_app


def pytest_configure(config: pytest.Config) -> None:
    """Keep test data in a new project-owned directory, never a shared temp tree."""
    if config.option.basetemp is None:
        cache = Path(__file__).resolve().parents[3] / ".cache" / "pytest"
        cache.mkdir(parents=True, exist_ok=True)
        config.option.basetemp = str(cache / uuid4().hex)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        database_url=f"sqlite:///{tmp_path}/test.db",
        data_dir=tmp_path,
        session_token="test-token",
        allowed_origins=["http://127.0.0.1:5173"],
        allowed_hosts=["testserver", "127.0.0.1", "localhost"],
    )


@pytest.fixture
def client(settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(settings)) as test_client:
        yield test_client


@pytest.fixture
def auth() -> dict[str, str]:
    return {"Authorization": "Bearer test-token"}


@pytest.fixture
def market_dataset(client: TestClient, auth: dict[str, str]) -> dict[str, Any]:
    response = client.post(
        "/v1/datasets/synthetic", json={"seed": 42, "securities": 3, "sessions": 30}, headers=auth
    )
    assert response.status_code == 201
    return response.json()["data"]
