from decimal import Decimal as D

from fastapi.testclient import TestClient

from portfolio_client.main import create_app


def draft(dataset):
    return {
        "name": "本地研究组合",
        "dataset_id": dataset["dataset_id"],
        "as_of": dataset["last_session"] + "T15:00:00+08:00",
        "target_weights": {"XSHG:600001": ".4", "XSHE:000002": ".4"},
        "max_weight": ".6",
        "holdings": {"XSHG:600001": 100},
        "cash": "10000",
    }


def test_portfolio_with_cash_budget_and_imported_holdings_survives_restart(
    client, auth, settings, market_dataset
):
    response = client.post("/v1/portfolios", json=draft(market_dataset), headers=auth)
    assert response.status_code == 201
    portfolio_id = response.json()["data"]["id"]
    with TestClient(create_app(settings)) as reopened:
        loaded = reopened.get(f"/v1/portfolios/{portfolio_id}", headers=auth).json()["data"]
        assert loaded["spec"]["holdings"] == {"XSHG:600001": 100}
        assert loaded["cash_weight"] == ".2" or loaded["cash_weight"] == "0.2"
        assert loaded["spec"]["cash"] == "10000"


def test_over_budget_or_unknown_asset_is_rejected(client, auth, market_dataset):
    invalid = draft(market_dataset)
    invalid["target_weights"] = {"XSHG:600001": ".6", "XSHE:000002": ".6"}
    assert client.post("/v1/portfolios", json=invalid, headers=auth).status_code == 422
    invalid = draft(market_dataset)
    invalid["holdings"] = {"XSHE:999999": 100}
    assert client.post("/v1/portfolios", json=invalid, headers=auth).status_code == 404


def test_optimizer_uses_only_visible_market_history(client, auth, market_dataset):
    result = client.post(
        "/v1/portfolios/optimize",
        headers=auth,
        json={
            "dataset_id": market_dataset["dataset_id"],
            "as_of": market_dataset["last_session"] + "T15:00:00+08:00",
            "universe": ["XSHG:600001", "XSHE:000002"],
            "method": "equal",
            "max_weight": ".6",
            "lookback": 20,
        },
    )
    assert result.status_code == 200
    data = result.json()["data"]
    assert data["target_weights"] == {"XSHG:600001": "0.5", "XSHE:000002": "0.5"}
    assert data["observations"] == 20
    assert data["price_basis"] == "raw"
    assert data["warnings"]


def test_optimizer_does_not_silently_shorten_requested_window(client, auth, market_dataset):
    result = client.post(
        "/v1/portfolios/optimize",
        headers=auth,
        json={
            "dataset_id": market_dataset["dataset_id"],
            "as_of": "2023-01-10T15:00:00+08:00",
            "universe": ["XSHG:600001", "XSHE:000002"],
            "method": "equal",
            "lookback": 20,
        },
    )
    assert result.status_code == 400
    assert result.json()["error"]["code"] == "INSUFFICIENT_HISTORY"


def test_rebalance_preview_returns_real_lots_and_fee_version(client, auth, market_dataset):
    created = client.post("/v1/portfolios", json=draft(market_dataset), headers=auth).json()["data"]
    response = client.post(f"/v1/portfolios/{created['id']}/rebalance", headers=auth, json={})
    assert response.status_code == 200
    plan = response.json()["data"]
    assert plan["orders"]
    assert D(plan["cash_after"]) >= 0
    assert plan["fee_schedule"]["version"] == "cn-stock-20220429"
    assert all(order["quantity"] % 100 == 0 for order in plan["orders"])


def test_rebalance_override_must_be_effective_at_portfolio_time(client, auth, market_dataset):
    created = client.post("/v1/portfolios", json=draft(market_dataset), headers=auth).json()["data"]
    response = client.post(
        f"/v1/portfolios/{created['id']}/rebalance",
        headers=auth,
        json={
            "fee_schedule": {
                "version": "future",
                "valid_from": "2030-01-01",
                "valid_to": None,
                "commission_bps": "2.5",
                "minimum_commission": "5",
                "transfer_bps": ".1",
                "stamp_bps": "5",
            }
        },
    )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "FEE_PERIOD_UNDEFINED"
