def save(client, auth, dataset, holdings):
    return client.post(
        "/v1/portfolios",
        headers=auth,
        json={
            "name": "风险样本",
            "dataset_id": dataset["dataset_id"],
            "as_of": dataset["last_session"] + "T15:00:00+08:00",
            "target_weights": {"XSHG:600001": ".5"},
            "holdings": holdings,
            "cash": "10000",
        },
    ).json()["data"]["id"]


def test_risk_api_is_based_on_actual_holdings_and_fixed_dataset(client, auth, market_dataset):
    portfolio_id = save(client, auth, market_dataset, {"XSHG:600001": 100})
    response = client.post("/v1/risk/portfolio", json={"portfolio_id": portfolio_id}, headers=auth)
    assert response.status_code == 200
    data = response.json()["data"]
    assert 0 < data["exposure"]["cash_weight"] < 1
    assert data["tail_risk"]["sample_count"] == 20
    assert data["tail_risk"]["var"]["source"] == "fixed_current_weights_historical_returns"
    assert response.json()["meta"]["data_version"] == market_dataset["version"]


def test_all_cash_has_no_invented_risk_history_and_targets_are_explicit_estimates(
    client, auth, market_dataset
):
    portfolio_id = save(client, auth, market_dataset, {})
    actual = client.post(
        "/v1/risk/portfolio", json={"portfolio_id": portfolio_id}, headers=auth
    ).json()["data"]
    assert actual["exposure"]["cash_weight"] == 1
    assert actual["tail_risk"]["var"]["value"] is None
    assert actual["tail_risk"]["var"]["reason"] == "all_cash_no_market_returns"
    target = client.post(
        "/v1/risk/portfolio", json={"portfolio_id": portfolio_id, "mode": "target"}, headers=auth
    ).json()["data"]
    assert target["exposure"]["asset_weights"] == {"XSHG:600001": 0.5}
    assert target["source"] == "target_weight_historical_estimate"


def test_metrics_api_preserves_null_reasons_and_price_benchmark_basis(client, auth):
    points = [{"session": "2024-01-02", "value": "100"}, {"session": "2024-01-03", "value": "110"}]
    response = client.post(
        "/v1/risk/metrics", headers=auth, json={"nav": points, "benchmark": points}
    )
    assert response.status_code == 200
    result = response.json()["data"]
    assert result["total_return"]["value"] == 0.1
    assert result["annualized_volatility"]["value"] is None
    assert result["annualized_volatility"]["reason"] == "insufficient_history"
    assert result["benchmark_basis"] == "price"
    assert result["excess_return"]["value"] == 0
