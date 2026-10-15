from collections.abc import AsyncIterator
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from portfolio_client.domain.screening.dsl import ScreeningSpec


class ExplanationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    mode: Literal["explain", "screen"] = "explain"
    report_id: str | None = Field(default=None, max_length=64)
    scope: ScreeningSpec | None = None
    question: str = Field(min_length=1, max_length=1000)
    max_output_tokens: int = Field(default=1024, ge=256, le=2048)

    @model_validator(mode="after")
    def valid_context(self) -> Self:
        if (self.mode == "explain" and (not self.report_id or self.scope is not None)) or (
            self.mode == "screen" and (self.scope is None or self.report_id is not None)
        ):
            raise ValueError("解释需要报告，选股需要固定研究上下文")
        return self


class ProviderRequest(BaseModel):
    messages: tuple[dict[str, str], ...]
    max_output_tokens: int


class ProviderChunk(BaseModel):
    text: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class AIProvider(Protocol):
    def stream(self, request: ProviderRequest) -> AsyncIterator[ProviderChunk]: ...


class AIEventPayload(BaseModel):
    message: str = ""
    text: str | None = None
    evidence_ids: tuple[str, ...] = ()
    spec: ScreeningSpec | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    code: str | None = None
    trace_id: str | None = None


class AIEvent(BaseModel):
    run_id: str
    seq: int
    type: Literal[
        "started", "tool_result", "paragraph", "usage", "succeeded", "failed", "cancelled"
    ]
    version: Literal[1] = 1
    payload: AIEventPayload


class AIRun(BaseModel):
    id: str
    state: Literal["running", "succeeded", "failed", "cancelled"]
    model: str
    request: ExplanationRequest
    events: tuple[AIEvent, ...]
    trace_id: str


class AIRunRepository(Protocol):
    def find(self, key: str, request: ExplanationRequest) -> AIRun | None: ...
    def create(
        self, key: str, request: ExplanationRequest, model: str, trace_id: str
    ) -> tuple[AIRun, bool]: ...
    def get(self, run_id: str) -> AIRun: ...
    def append(self, run_id: str, kind: str, payload: AIEventPayload) -> AIRun: ...
    def recover(self) -> None: ...
