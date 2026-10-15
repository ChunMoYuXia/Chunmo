from datetime import date, timedelta
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient


def create_dataset(client: TestClient, auth: dict[str, str]):
    return client.post(
        "/v1/datasets/synthetic", json={"seed": 3, "securities": 3, "sessions": 30}, headers=auth
    ).json()["data"]


def test_search_supports_code_initials_and_industry(
    client: TestClient, auth: dict[str, str]
) -> None:
    dataset = create_dataset(client, auth)
    for query in ["000002", "ys", "金融"]:
        response = client.get(
            "/v1/market/search",
            params={"dataset_id": dataset["dataset_id"], "q": query},
            headers=auth,
        )
        assert response.status_code == 200
        assert response.json()["data"]


def test_future_daily_bars_are_not_returned_before_close(
    client: TestClient, auth: dict[str, str]
) -> None:
    dataset = create_dataset(client, auth)
    response = client.get(
        "/v1/market/kline",
        params={
            "dataset_id": dataset["dataset_id"],
            "security_id": "XSHG:600001",
            "as_of": "2023-01-04T14:59:59+08:00",
        },
        headers=auth,
    )
    assert response.status_code == 200
    assert [bar["trade_date"] for bar in response.json()["data"]] == ["2023-01-02", "2023-01-03"]
    assert isinstance(response.json()["data"][0]["close"], str)


def test_system_replay_does_not_claim_backfilled_data_was_known(
    client: TestClient, auth: dict[str, str]
) -> None:
    dataset = create_dataset(client, auth)
    response = client.get(
        "/v1/market/search",
        params={
            "dataset_id": dataset["dataset_id"],
            "q": "ys",
            "as_of": "2023-01-31T15:00:00+08:00",
            "knowledge_mode": "system_replay",
        },
        headers=auth,
    )
    assert response.status_code == 200
    assert response.json()["data"] == []


def test_future_industry_update_is_not_used_in_historical_search(
    client: TestClient, auth: dict[str, str]
) -> None:
    quotes = "code,date,open,high,low,close,volume,industry\n000001,2026-01-05,10,11,9,10,1000,金融\n000001,2026-01-06,10,11,9,10,1000,科技\n".encode()
    preview = client.post(
        "/v1/datasets/import/preview",
        files={
            "file": ("quotes.csv", quotes),
            "calendar_file": ("calendar.csv", b"date\n2026-01-05\n2026-01-06\n"),
        },
        headers=auth,
    ).json()["data"]
    dataset = client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).json()[
        "data"
    ]
    response = client.get(
        "/v1/market/search",
        params={
            "dataset_id": dataset["dataset_id"],
            "q": "科技",
            "as_of": "2026-01-05T15:00:00+08:00",
        },
        headers=auth,
    )
    assert response.status_code == 200
    assert response.json()["data"] == []


def import_price_history(client, auth, omitted_index=None):
    days = []
    day = date(2026, 1, 5)
    while len(days) < 25:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    lines = ["code,date,open,high,low,close,volume"]
    for index, day in enumerate(days):
        if index != omitted_index:
            price = 999 if index > 20 else 100 + index
            lines.append(f"000001,{day.isoformat()},{price},{price},{price},{price},1000")
    calendar = "date\n" + "\n".join(day.isoformat() for day in days) + "\n"
    preview = client.post(
        "/v1/datasets/import/preview",
        files={
            "file": ("quotes.csv", ("\n".join(lines) + "\n").encode()),
            "calendar_file": ("calendar.csv", calendar.encode()),
        },
        headers=auth,
    ).json()["data"]
    dataset = client.post(f"/v1/datasets/import/{preview['id']}/commit", headers=auth).json()[
        "data"
    ]
    return dataset, days


def test_snapshot_momentum_cannot_read_future_prices(
    client: TestClient, auth: dict[str, str]
) -> None:
    dataset, days = import_price_history(client, auth)
    response = client.get(
        "/v1/market/snapshot",
        params={
            "dataset_id": dataset["dataset_id"],
            "as_of": f"{days[20]}T15:00:00+08:00",
        },
        headers=auth,
    )
    assert response.status_code == 200
    row = response.json()["data"][0]
    assert Decimal(row["close"]) == Decimal("120")
    assert row["momentum_20d"] == pytest.approx(0.2, abs=1e-12)


def test_missing_session_does_not_become_a_complete_feature_window(
    client: TestClient, auth: dict[str, str]
) -> None:
    dataset, days = import_price_history(client, auth, omitted_index=10)
    response = client.get(
        "/v1/market/snapshot",
        params={
            "dataset_id": dataset["dataset_id"],
            "as_of": f"{days[-1]}T15:00:00+08:00",
        },
        headers=auth,
    )
    assert response.status_code == 200
    row = response.json()["data"][0]
    assert row["momentum_20d"] is None
    assert row["volatility_20d"] is None
    assert row["missing_sessions"] == [days[10].isoformat()]
