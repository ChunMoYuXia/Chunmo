from fastapi.testclient import TestClient

from portfolio_client.main import create_app


def test_unknown_route_uses_same_traceable_error_envelope(client, auth):
    response = client.get("/v1/unknown", headers=auth)
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "ROUTE_NOT_FOUND"
    assert response.headers["X-Trace-Id"] == response.json()["error"]["trace_id"]


def test_unexpected_failure_is_traceable_without_exposing_exception_text(settings, auth):
    app = create_app(settings)

    def crash():
        raise RuntimeError("private-example-value")

    app.add_api_route("/v1/test-crash", crash)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/v1/test-crash", headers=auth)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "INTERNAL_ERROR"
    assert response.headers["X-Trace-Id"] == response.json()["error"]["trace_id"]
    assert "private-example-value" not in response.text
