from decimal import Decimal

from fastapi.testclient import TestClient

from portfolio_client.main import create_app
from tests.paper_helpers import create_account, order_body


def test_idempotent_order_payload_conflict(client, auth, market_dataset):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}/orders"
    headers = {**auth, "Idempotency-Key": "order-1"}
    first = client.post(path, json=order_body(), headers=headers)
    assert first.status_code == 201
    assert (
        client.post(path, json=order_body(), headers=headers).json()["data"] == first.json()["data"]
    )
    assert client.post(path, json=order_body(200), headers=headers).status_code == 409
    loaded = client.get(f"/v1/paper/accounts/{account['id']}", headers=auth).json()["data"]
    assert float(loaded["frozen_cash"]) > 0


def test_advance_fills_once_and_restart_keeps_money_positions_and_idempotency(
    client, auth, settings, market_dataset
):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}"
    client.post(path + "/orders", json=order_body(), headers={**auth, "Idempotency-Key": "buy"})
    advanced = client.post(path + "/advance", json={}, headers={**auth, "Idempotency-Key": "day-1"})
    assert advanced.status_code == 200
    snapshot = advanced.json()["data"]
    assert snapshot["positions"]["XSHG:600001"]["quantity"] == 100
    assert snapshot["positions"]["XSHG:600001"]["sellable"] == 0
    assert Decimal(snapshot["frozen_cash"]) == 0
    with TestClient(create_app(settings)) as reopened:
        assert reopened.get(path, headers=auth).json()["data"] == snapshot
        assert (
            reopened.post(
                path + "/advance", json={}, headers={**auth, "Idempotency-Key": "day-1"}
            ).json()["data"]
            == snapshot
        )
        assert len(reopened.get(path + "/trades", headers=auth).json()["data"]) == 1
        assert reopened.get(path + "/ledger", headers=auth).json()["data"]
        verification = reopened.get(path + "/audit/verify", headers=auth).json()["data"]
        assert verification["valid"] and verification["checked_events"] == 3


def test_halt_cancels_open_orders_and_resume_requires_explicit_phrase(client, auth, market_dataset):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}"
    client.post(path + "/orders", json=order_body(), headers={**auth, "Idempotency-Key": "buy"})
    halted = client.post(path + "/halt", json={}, headers={**auth, "Idempotency-Key": "halt"})
    assert halted.status_code == 200
    assert halted.json()["data"]["account"]["status"] == "paused"
    assert Decimal(halted.json()["data"]["account"]["frozen_cash"]) == 0
    assert len(halted.json()["data"]["canceled_order_ids"]) == 1
    blocked = client.post(
        path + "/orders", json=order_body(), headers={**auth, "Idempotency-Key": "buy-2"}
    )
    assert blocked.status_code == 409 and blocked.json()["error"]["code"] == "ACCOUNT_PAUSED"
    assert (
        client.post(
            path + "/resume",
            json={"confirmation": "yes"},
            headers={**auth, "Idempotency-Key": "resume"},
        ).status_code
        == 400
    )
    resumed = client.post(
        path + "/resume",
        json={"confirmation": "恢复模拟交易"},
        headers={**auth, "Idempotency-Key": "resume"},
    )
    assert resumed.json()["data"]["status"] == "active"
    assert client.get(path + "/orders", headers=auth).json()["data"][0]["state"] == "canceled"


def test_every_financial_write_requires_a_key_and_accounts_have_separate_scopes(
    client, auth, market_dataset
):
    assert (
        client.post(
            "/v1/paper/accounts",
            json={"name": "x", "capital": "1000", "dataset_id": market_dataset["dataset_id"]},
            headers=auth,
        ).status_code
        == 422
    )
    one, two = [create_account(client, auth, market_dataset) for _ in range(2)]
    for account in (one, two):
        response = client.post(
            f"/v1/paper/accounts/{account['id']}/orders",
            json=order_body(),
            headers={**auth, "Idempotency-Key": "same-key"},
        )
        assert response.status_code == 201


def test_market_order_rejects_unknown_security_and_non_whole_lots(client, auth, market_dataset):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}/orders"
    assert (
        client.post(
            path,
            json={**order_body(), "security_id": "XSHE:999999"},
            headers={**auth, "Idempotency-Key": "unknown"},
        ).status_code
        == 404
    )
    response = client.post(path, json=order_body(1), headers={**auth, "Idempotency-Key": "odd"})
    assert response.status_code == 400 and response.json()["error"]["code"] == "INVALID_LOT"


def test_t_plus_one_sell_reservations_and_cancel_release(client, auth, market_dataset):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}"
    client.post(path + "/orders", json=order_body(), headers={**auth, "Idempotency-Key": "buy"})
    client.post(path + "/advance", json={}, headers={**auth, "Idempotency-Key": "day-1"})
    sale = {**order_body(), "side": "sell"}
    blocked = client.post(
        path + "/orders", json=sale, headers={**auth, "Idempotency-Key": "sell-1"}
    )
    assert blocked.status_code == 400 and blocked.json()["error"]["code"] == "INSUFFICIENT_SELLABLE"
    client.post(path + "/advance", json={}, headers={**auth, "Idempotency-Key": "day-2"})
    submitted = client.post(
        path + "/orders", json=sale, headers={**auth, "Idempotency-Key": "sell-1"}
    )
    assert submitted.status_code == 201
    assert (
        client.post(
            path + "/orders", json=sale, headers={**auth, "Idempotency-Key": "sell-2"}
        ).status_code
        == 400
    )
    assert client.get(path, headers=auth).json()["data"]["available_to_sell"]["XSHG:600001"] == 0
    cancel_path = path + "/orders/" + submitted.json()["data"]["id"] + "/cancel"
    canceled = client.post(cancel_path, json={}, headers={**auth, "Idempotency-Key": "cancel"})
    assert canceled.status_code == 200 and canceled.json()["data"]["state"] == "canceled"
    assert (
        client.post(cancel_path, json={}, headers={**auth, "Idempotency-Key": "cancel"}).json()[
            "data"
        ]
        == canceled.json()["data"]
    )
    assert client.get(path, headers=auth).json()["data"]["available_to_sell"]["XSHG:600001"] == 100


def test_decimal_representation_is_canonical_for_order_idempotency_and_drafts_rejected(
    client, auth, market_dataset
):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}/orders"
    body = {**order_body(), "order_type": "limit", "limit_price": "10.0"}
    headers = {**auth, "Idempotency-Key": "canonical"}
    first = client.post(path, json=body, headers=headers)
    assert first.status_code == 201
    assert (
        client.post(path, json={**body, "limit_price": "10.00"}, headers=headers).json()["data"]
        == first.json()["data"]
    )
    assert (
        client.post(
            path, json={**body, "state": "draft"}, headers={**auth, "Idempotency-Key": "draft"}
        ).status_code
        == 422
    )
