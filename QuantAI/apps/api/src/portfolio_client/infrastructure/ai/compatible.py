import json
from collections.abc import AsyncIterator
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, field_validator

from portfolio_client.domain.errors import DomainError
from portfolio_client.domain.research.ai_models import ProviderChunk, ProviderRequest
from portfolio_client.infrastructure.ai.sse import bounded_lines


class ProviderSettings(BaseModel):
    model_config = ConfigDict(frozen=True)
    base_url: str
    model: str = Field(min_length=1, max_length=120)
    api_key: SecretStr
    timeout_seconds: float = Field(default=30, gt=0, le=120)

    @field_validator("base_url")
    @classmethod
    def controlled_endpoint(cls, value: str) -> str:
        url = urlsplit(value)
        if (
            url.username
            or url.password
            or url.query
            or url.fragment
            or not url.hostname
            or (
                url.scheme != "https"
                and not (url.scheme == "http" and url.hostname in {"127.0.0.1", "localhost"})
            )
        ):
            raise ValueError("AI端点必须为HTTPS或本机HTTP，不得包含凭证或查询参数")
        return value.rstrip("/")


class CompatibleProvider:
    def __init__(
        self, settings: ProviderSettings, *, transport: httpx.AsyncBaseTransport | None = None
    ) -> None:
        self.settings, self.transport = settings, transport

    async def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderChunk]:
        payload = {
            "model": self.settings.model,
            "messages": request.messages,
            "stream": True,
            "max_completion_tokens": request.max_output_tokens,
            "stream_options": {"include_usage": True},
        }
        try:
            async with (
                httpx.AsyncClient(
                    timeout=self.settings.timeout_seconds,
                    follow_redirects=False,
                    transport=self.transport,
                    trust_env=False,
                ) as client,
                client.stream(
                    "POST",
                    f"{self.settings.base_url}/chat/completions",
                    json=payload,
                    headers={"Authorization": f"Bearer {self.settings.api_key.get_secret_value()}"},
                ) as response,
            ):
                if response.status_code != 200:
                    raise DomainError(
                        "AI_PROVIDER_REJECTED",
                        "模型服务拒绝请求，请检查后端配置与额度",
                        {"status": response.status_code},
                    )
                done = False
                async for line in bounded_lines(response):
                    if not line.startswith("data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == "[DONE]":
                        done = True
                        break
                    chunk = json.loads(raw)
                    if not isinstance(chunk, dict):
                        raise ValueError("invalid chunk object")
                    usage = chunk.get("usage") or {}
                    choices = chunk.get("choices") or []
                    if (
                        not isinstance(usage, dict)
                        or not isinstance(choices, list)
                        or (choices and not isinstance(choices[0], dict))
                    ):
                        raise ValueError("invalid chunk fields")
                    delta = choices[0].get("delta", {}) if choices else {}
                    if not isinstance(delta, dict):
                        raise ValueError("invalid delta object")
                    text = delta.get("content") or ""
                    if not isinstance(text, str):
                        raise ValueError("invalid delta")
                    yield ProviderChunk(
                        text=text,
                        prompt_tokens=usage.get("prompt_tokens"),
                        completion_tokens=usage.get("completion_tokens"),
                    )
                if not done:
                    raise DomainError(
                        "AI_PROVIDER_DISCONNECTED",
                        "模型流未正常结束；已保存部分事件，请勿自动重新生成",
                    )
        except httpx.TimeoutException:
            raise DomainError("AI_PROVIDER_TIMEOUT", "模型服务响应超时；可查询原运行记录") from None
        except httpx.HTTPError:
            raise DomainError(
                "AI_PROVIDER_UNAVAILABLE", "无法连接模型服务；请检查后端配置"
            ) from None
        except (ValueError, TypeError, KeyError, IndexError):
            raise DomainError(
                "AI_PROVIDER_INVALID_RESPONSE", "模型服务返回不兼容的流式响应"
            ) from None
