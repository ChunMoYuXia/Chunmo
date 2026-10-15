from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

from fastapi.testclient import TestClient
from sqlalchemy import event, text

from portfolio_client.infrastructure.db.database import Database
from portfolio_client.main import create_app
from tests.paper_helpers import create_account, order_body


def test_two_connections_retry_same_key_only_reserve_once(client, auth, settings, market_dataset):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}"
    barrier = Barrier(2)

    def submit(connection):
        barrier.wait(timeout=10)
        response = connection.post(
            path + "/orders", json=order_body(), headers={**auth, "Idempotency-Key": "concurrent"}
        )
        assert response.status_code == 201
        return response.json()["data"]

    # Schema construction is sequential; the two real SQL write requests still race.
    with (
        TestClient(create_app(settings)) as one,
        TestClient(create_app(settings)) as two,
        ThreadPoolExecutor(max_workers=2) as pool,
    ):
        futures = [pool.submit(submit, connection) for connection in (one, two)]
        results = [future.result(timeout=20) for future in futures]
    assert results[0] == results[1]
    assert len(client.get(path + "/orders", headers=auth).json()["data"]) == 1
    assert client.get(path + "/audit/verify", headers=auth).json()["data"]["checked_events"] == 2


def test_failed_trade_insert_rolls_back_account_order_ledger_audit_and_idempotency(
    client, auth, settings, market_dataset
):
    account = create_account(client, auth, market_dataset)
    path = f"/v1/paper/accounts/{account['id']}"
    client.post(path + "/orders", json=order_body(), headers={**auth, "Idempotency-Key": "buy"})
    before = client.get(path, headers=auth).json()["data"]
    # Inject a real database flush failure at the financial transaction boundary.
    from sqlalchemy.engine import Engine

    def fail_trade(_connection, _cursor, statement, _parameters, _context, _executemany):
        if statement.startswith("INSERT INTO paper_trades"):
            raise RuntimeError("injected trade persistence failure")

    event.listen(Engine, "before_cursor_execute", fail_trade)
    try:
        failed = client.post(
            path + "/advance", json={}, headers={**auth, "Idempotency-Key": "advance"}
        )
        assert failed.status_code == 500
        assert failed.json()["error"]["code"] == "INTERNAL_ERROR"
        assert "injected" not in failed.text
    finally:
        event.remove(Engine, "before_cursor_execute", fail_trade)
    assert client.get(path, headers=auth).json()["data"] == before
    assert client.get(path + "/trades", headers=auth).json()["data"] == []
    assert client.get(path + "/orders", headers=auth).json()["data"][0]["state"] == "submitted"
    assert client.get(path + "/audit/verify", headers=auth).json()["data"]["checked_events"] == 2
    retried = client.post(
        path + "/advance", json={}, headers={**auth, "Idempotency-Key": "advance"}
    )
    assert retried.status_code == 200


def test_audit_tampering_is_detected_without_claiming_admin_proof_security(
    client, auth, settings, market_dataset
):
    account = create_account(client, auth, market_dataset)
    database = Database(settings.database_url)
    try:
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE audit_events SET payload_json=:payload WHERE chain_id=:chain AND seq=1"
                ),
                {"payload": '{"schema_version":1,"changed":true}', "chain": account["id"]},
            )
    finally:
        database.close()
    verified = client.get(f"/v1/paper/accounts/{account['id']}/audit/verify", headers=auth)
    assert verified.status_code == 200
    assert not verified.json()["data"]["valid"]


def test_audit_verification_also_detects_ledger_row_tampering(
    client, auth, settings, market_dataset
):
    account = create_account(client, auth, market_dataset)
    database = Database(settings.database_url)
    try:
        with database.engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE ledger_events SET payload=:payload WHERE account_id=:account AND seq=1"
                ),
                {"payload": '{"seq":1,"kind":"changed"}', "account": account["id"]},
            )
    finally:
        database.close()
    verified = client.get(f"/v1/paper/accounts/{account['id']}/audit/verify", headers=auth)
    assert verified.status_code == 200
    assert not verified.json()["data"]["valid"]
    assert verified.json()["data"]["reason"] == "ledger_mismatch"
