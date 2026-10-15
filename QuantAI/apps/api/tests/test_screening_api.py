from typing import Any

import pytest
from fastapi.testclient import TestClient

from portfolio_client.config import Settings
from portfolio_client.main import create_app


def payload(dataset: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": 1,
        "dataset_id": dataset["dataset_id"],
        "as_of": dataset["last_session"] + "T15:00:00+08:00",
        "universe": ["XSHG:600001", "XSHE:000002", "XSHG:600003"],
        "filters": [{"field": "close", "operator": "gte", "value": "0"}],
        "sort_by": "momentum_20d",
        "descending": True,
        "limit": 2,
    }


@pytest.mark.parametrize(
    "condition",
    [
        {"field": "__class__", "operator": "eq", "value": "x"},
        {"field": "close", "operator": "execute", "value": "10"},
        {"field": "close", "operator": "gt", "value": "NaN"},
        {"field": "close", "operator": "gt", "value": True},
    ],
)
def test_invalid_dsl_is_rejected_before_execution(client, auth, market_dataset, condition) -> None:
    spec = payload(market_dataset)
    spec["filters"] = [condition]
    assert client.post("/v1/screening", json=spec, headers=auth).status_code == 422


def test_screening_returns_real_candidates_and_filter_evidence(
    client, auth, market_dataset
) -> None:
    response = client.post("/v1/screening", json=payload(market_dataset), headers=auth)
    assert response.status_code == 200
    result = response.json()["data"]
    assert result["universe_count"] == 3
    assert result["total_matching_count"] == 3
    assert len(result["rows"]) == 2
    assert result["steps"][0]["remaining_count"] == 3


def test_saved_screening_survives_restart_without_losing_conditions(
    client: TestClient, settings: Settings, auth, market_dataset
) -> None:
    spec = payload(market_dataset)
    created = client.post(
        "/v1/screening/presets", json={"name": "科技候选", "spec": spec}, headers=auth
    )
    assert created.status_code == 201
    preset_id = created.json()["data"]["id"]
    with TestClient(create_app(settings)) as reopened:
        loaded = reopened.get(f"/v1/screening/presets/{preset_id}", headers=auth).json()["data"]
        assert loaded["name"] == "科技候选"
        assert loaded["spec"]["universe"] == spec["universe"]
        assert loaded["spec"]["filters"][0]["value"] == "0"
        assert reopened.post("/v1/screening", json=loaded["spec"], headers=auth).status_code == 200
