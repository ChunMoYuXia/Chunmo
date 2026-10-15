import asyncio
import json
import threading
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.ai_models import ProviderChunk, ProviderRequest
from portfolio_client.domain.research.ai_validation import render_metric_tokens
from portfolio_client.domain.research.metric_models import Metric
from portfolio_client.infrastructure.ai.compatible import CompatibleProvider, ProviderSettings
from portfolio_client.main import create_app


def test_unknown_metric_token_cannot_publish() -> None:
    with pytest.raises(DomainError) as error:
        render_metric_tokens("收益 {{metric:unknown}}", {})
    assert error.value.code == "UNKNOWN_METRIC_REFERENCE"


@pytest.mark.parametrize(
    "text",
    [
        "预计收益 15%",
        "预计收益百分之五",
        "{{metric:bad-key}}",
        "预计收益Ⅻ%",
        "预计收益翻倍",
        "预计收益减半",
    ],
)
def test_unsourced_numeric_or_malformed_references_cannot_publish(text: str) -> None:
    with pytest.raises(DomainError):
        render_metric_tokens(text, {})


def test_known_metric_uses_server_values_and_missing_reason() -> None:
    known = Metric(0.125, "ratio", "v1", "nav", datetime(2025, 1, 2, tzinfo=UTC))
    missing = Metric(None, "ratio", "v1", "nav", None, "insufficient_history")
    assert (
        render_metric_tokens(
            "收益 {{metric:return}}，风险 {{metric:risk}}", {"return": known, "risk": missing}
        )
        == "收益 12.50%，风险 数据缺失（insufficient_history）"
    )


def test_unconfigured_ai_never_returns_a_fake_explanation(
    client: TestClient, auth: dict[str, str]
) -> None:
    response = client.get("/v1/ai/config", headers=auth)
    assert response.status_code == 200
    assert response.json()["data"]["configured"] is False
    response = client.post(
        "/v1/ai/runs",
        headers={**auth, "Idempotency-Key": "unconfigured"},
        json={"mode": "explain", "report_id": "unavailable", "question": "解释研究结果"},
    )
    assert response.json()["error"]["code"] == "AI_NOT_CONFIGURED"


def test_provider_timeout_is_sanitized_and_never_exposes_key_or_url() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret-key https://private.example", request=request)

    settings = ProviderSettings(
        base_url="https://private.example/v1", model="example", api_key=SecretStr("secret-key")
    )

    async def run() -> None:
        provider = CompatibleProvider(settings, transport=httpx.MockTransport(timeout))
        with pytest.raises(DomainError) as error:
            async for _ in provider.stream(
                ProviderRequest(
                    messages=({"role": "user", "content": "解释"},), max_output_tokens=256
                )
            ):
                pass
        assert error.value.code == "AI_PROVIDER_TIMEOUT"
        assert "secret-key" not in str(error.value) and "private.example" not in str(error.value)

    asyncio.run(run())


class ControlledProvider:
    def __init__(self, text: str, *, hold: bool = False) -> None:
        self.text, self.hold, self.calls = text, hold, 0
        self.started, self.stopped = threading.Event(), threading.Event()

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderChunk]:
        self.calls += 1
        self.started.set()
        try:
            if self.hold:
                await asyncio.Event().wait()
            yield ProviderChunk(text=self.text[:7])
            yield ProviderChunk(text=self.text[7:], prompt_tokens=100, completion_tokens=120)
        finally:
            self.stopped.set()


def configured_settings(settings):
    return settings.model_copy(
        update={
            "ai": ProviderSettings(
                base_url="https://example.invalid/v1",
                model="example",
                api_key=SecretStr("test-private-provider-key"),
            )
        }
    )


def screening_request(client, auth):
    dataset = client.post(
        "/v1/datasets/synthetic", headers=auth, json={"securities": 3, "sessions": 30, "seed": 42}
    ).json()["data"]
    return {
        "mode": "screen",
        "scope": {
            "version": 1,
            "dataset_id": dataset["dataset_id"],
            "as_of": dataset["last_session"] + "T15:00:00+08:00",
            "universe": ["XSHG:600001"],
        },
        "question": "筛选股价较高的证券",
    }


def finished_ai(client, auth, run_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        record = client.get(f"/v1/ai/runs/{run_id}", headers=auth).json()["data"]
        if record["state"] != "running":
            return record
        time.sleep(0.01)
    raise AssertionError("AI run did not reach a terminal state")


def test_validated_dsl_events_replay_without_new_provider_charge_and_survive_restart(
    settings, auth
):
    paragraph = json.dumps(
        {
            "text": "按价格条件筛选，结果需结合数据质量判断。",
            "evidence_ids": ["screening_fields"],
            "dsl": {"filters": [{"field": "close", "operator": "gt", "value": "5"}]},
        }
    )
    provider = ControlledProvider(paragraph)
    settings = configured_settings(settings)
    with TestClient(create_app(settings, ai_provider=provider)) as client:
        payload = screening_request(client, auth)
        headers = {**auth, "Idempotency-Key": "same-ai-request"}
        created = client.post("/v1/ai/runs", headers=headers, json=payload)
        assert created.status_code == 202
        run_id = created.json()["data"]["id"]
        record = finished_ai(client, auth, run_id)
        assert record["state"] == "succeeded"
        events = record["events"]
        assert [event["seq"] for event in events] == list(range(1, len(events) + 1))
        explanation = next(event for event in events if event["type"] == "paragraph")
        assert explanation["payload"]["spec"]["universe"] == ["XSHG:600001"]
        assert explanation["payload"]["spec"]["filters"][0]["value"] == "5"
        stream = client.post(f"/v1/ai/runs/{run_id}/events", headers=auth, json={"after": 2})
        assert stream.headers["content-type"].startswith("text/event-stream")
        assert '"type":"paragraph"' in stream.text and "id: 1\n" not in stream.text
        assert (
            client.post("/v1/ai/runs", headers=headers, json=payload).json()["data"]["id"] == run_id
        )
        assert (
            client.post(
                "/v1/ai/runs", headers=headers, json={**payload, "question": "改变请求"}
            ).status_code
            == 409
        )
        assert provider.calls == 1
        assert "test-private-provider-key" not in client.get("/v1/ai/config", headers=auth).text
    with TestClient(create_app(settings, ai_provider=provider)) as reopened:
        assert reopened.get(f"/v1/ai/runs/{run_id}", headers=auth).json()["data"] == record
        assert reopened.post("/v1/ai/runs", headers=headers, json=payload).json()["data"] == record
        assert provider.calls == 1


@pytest.mark.parametrize(
    ("paragraph", "code"),
    [
        ({"text": "预计收益15%", "evidence_ids": ["screening_fields"]}, "UNSOURCED_NUMBER"),
        (
            {"text": "收益 {{metric:unknown}}", "evidence_ids": ["screening_fields"]},
            "UNKNOWN_METRIC_REFERENCE",
        ),
        (
            {
                "text": "条件可供研究。",
                "evidence_ids": ["screening_fields"],
                "dsl": {"filters": [{"field": "sql", "operator": "eq", "value": "DROP TABLE"}]},
            },
            "INVALID_AI_DSL",
        ),
        (
            {"text": "条件可供研究。", "evidence_ids": ["made_up"], "dsl": {"filters": []}},
            "UNKNOWN_EVIDENCE_REFERENCE",
        ),
    ],
)
def test_invalid_model_paragraph_never_reaches_published_events(settings, auth, paragraph, code):
    provider = ControlledProvider(json.dumps(paragraph))
    with TestClient(create_app(configured_settings(settings), ai_provider=provider)) as client:
        response = client.post(
            "/v1/ai/runs",
            headers={**auth, "Idempotency-Key": "reject"},
            json=screening_request(client, auth),
        )
        record = finished_ai(client, auth, response.json()["data"]["id"])
        assert record["state"] == "failed"
        assert record["events"][-1]["payload"]["code"] == code
        assert not any(event["type"] == "paragraph" for event in record["events"])


def test_cancellation_stops_provider_and_leaves_a_durable_terminal_record(settings, auth):
    provider = ControlledProvider("", hold=True)
    with TestClient(create_app(configured_settings(settings), ai_provider=provider)) as client:
        response = client.post(
            "/v1/ai/runs",
            headers={**auth, "Idempotency-Key": "cancel"},
            json=screening_request(client, auth),
        )
        run_id = response.json()["data"]["id"]
        assert provider.started.wait(2)
        cancelled = client.post(f"/v1/ai/runs/{run_id}/cancel", headers=auth).json()["data"]
        assert cancelled["state"] == "cancelled" and provider.stopped.wait(2)
        assert client.get(f"/v1/ai/runs/{run_id}", headers=auth).json()["data"] == cancelled


def test_provider_invalid_json_shape_has_a_public_domain_error() -> None:
    settings = ProviderSettings(
        base_url="https://example.invalid/v1", model="example", api_key=SecretStr("secret-key")
    )
    transport = httpx.MockTransport(
        lambda _: httpx.Response(200, text="data: []\n\ndata: [DONE]\n\n")
    )

    async def run() -> None:
        with pytest.raises(DomainError) as error:
            async for _ in CompatibleProvider(settings, transport=transport).stream(
                ProviderRequest(messages=(), max_output_tokens=256)
            ):
                pass
        assert error.value.code == "AI_PROVIDER_INVALID_RESPONSE"

    asyncio.run(run())


def test_provider_bounds_bytes_before_buffering_a_line_without_newlines() -> None:
    class OversizedStream(httpx.AsyncByteStream):
        consumed = 0

        async def __aiter__(self):
            for _ in range(128):
                self.consumed += 1
                yield b"x" * 4096

    stream = OversizedStream()
    transport = httpx.MockTransport(lambda _: httpx.Response(200, stream=stream))
    settings = ProviderSettings(
        base_url="https://example.invalid/v1", model="example", api_key=SecretStr("secret-key")
    )

    async def run() -> None:
        with pytest.raises(DomainError) as error:
            async for _ in CompatibleProvider(settings, transport=transport).stream(
                ProviderRequest(messages=(), max_output_tokens=256)
            ):
                pass
        assert error.value.code == "AI_OUTPUT_BUDGET"
        assert stream.consumed <= 49

    asyncio.run(run())


def test_repeated_provider_usage_updates_are_compacted_before_durable_publication(settings, auth):
    class RepeatedUsageProvider(ControlledProvider):
        async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderChunk]:
            for _ in range(50):
                yield ProviderChunk(prompt_tokens=100, completion_tokens=120)
            yield ProviderChunk(text=self.text)

    text = json.dumps(
        {
            "text": "价格条件仅用于研究。",
            "evidence_ids": ["screening_fields"],
            "dsl": {"filters": []},
        }
    )
    with TestClient(
        create_app(configured_settings(settings), ai_provider=RepeatedUsageProvider(text))
    ) as client:
        created = client.post(
            "/v1/ai/runs",
            headers={**auth, "Idempotency-Key": "usage"},
            json=screening_request(client, auth),
        )
        record = finished_ai(client, auth, created.json()["data"]["id"])
        assert record["state"] == "succeeded"
        assert len([event for event in record["events"] if event["type"] == "usage"]) == 1
