from fastapi.testclient import TestClient


def test_health_is_readable_without_access_to_private_config(client: TestClient) -> None:
    assert client.get("/health").status_code == 200
    assert client.get("/v1/config").status_code == 401


def test_config_requires_valid_session_and_never_returns_secret(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get("/v1/config", headers=auth)
    assert response.status_code == 200
    assert response.json()["data"]["mode"] == "local"
    assert "test-token" not in response.text
    assert response.json()["meta"]["trace_id"]


def test_external_origin_cannot_use_even_a_valid_token(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get("/v1/config", headers={**auth, "Origin": "https://untrusted.example"})
    assert response.status_code == 403
    assert response.json()["error"]["code"] == "ORIGIN_REJECTED"


def test_spoofed_hostname_is_rejected(client: TestClient, auth: dict[str, str]) -> None:
    response = client.get("/v1/config", headers={**auth, "Host": "untrusted.example"})
    assert response.status_code == 403


def test_wrong_token_has_traceable_error(client: TestClient) -> None:
    response = client.get("/v1/config", headers={"Authorization": "Bearer wrong"})
    assert response.status_code == 401
    assert response.json()["error"]["trace_id"] == response.headers["X-Trace-Id"]


def test_browser_session_can_reload_without_exposing_token(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.post("/v1/session", headers=auth)
    assert response.status_code == 200
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=strict" in response.headers["set-cookie"]
    assert client.get("/v1/config").status_code == 200


def test_cookie_authenticated_write_requires_trusted_origin(
    client: TestClient, auth: dict[str, str]
) -> None:
    client.post("/v1/session", headers=auth)
    assert client.post("/v1/session").status_code == 403
