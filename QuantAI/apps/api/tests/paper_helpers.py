from uuid import uuid4


def create_account(client, auth, dataset, **options):
    response = client.post(
        "/v1/paper/accounts",
        headers={**auth, "Idempotency-Key": uuid4().hex},
        json={
            "name": "模拟研究",
            "capital": "100000",
            "dataset_id": dataset["dataset_id"],
            **options,
        },
    )
    assert response.status_code == 201
    return response.json()["data"]


def order_body(quantity=100):
    return {
        "security_id": "XSHG:600001",
        "side": "buy",
        "quantity": quantity,
        "order_type": "market",
    }
