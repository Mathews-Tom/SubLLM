from __future__ import annotations

import json
import logging
from pathlib import Path
from collections.abc import AsyncIterator, Iterator
from typing import Any

from fastapi.testclient import TestClient
import pytest

from subllm.providers.base import Provider, ProviderCapabilities
from subllm.router import Router
from subllm.server import create_app
from subllm.server_api.settings import ServerSettings
from subllm.telemetry import (
    TelemetryConfig,
    bind_request_context,
    configure_telemetry,
    make_request_context,
    shutdown_telemetry,
)
from subllm.types import (
    AuthStatus,
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    CompletionRequest,
    Message,
    ProviderMessage,
    SessionRequest,
)


class FakeProvider(Provider):
    @property
    def name(self) -> str:
        return "fake"

    @property
    def supported_models(self) -> list[str]:
        return ["demo"]

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities()

    async def check_auth(self) -> AuthStatus:
        return AuthStatus(provider=self.name, authenticated=True)

    async def complete(
        self,
        messages: list[ProviderMessage],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session: SessionRequest | None = None,
    ) -> ChatCompletionResponse:
        assert model == "demo"
        assert messages == [{"role": "user", "content": "hello", "images": []}]
        return ChatCompletionResponse(
            model="fake/demo",
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    def stream(self, *args: Any, **kwargs: Any) -> AsyncIterator[ChatCompletionChunk]:
        async def _unexpected() -> AsyncIterator[ChatCompletionChunk]:
            raise AssertionError("stream should not be called in this test")
            yield ChatCompletionChunk()

        return _unexpected()


@pytest.fixture(autouse=True)
def reset_telemetry() -> Iterator[None]:
    shutdown_telemetry()
    yield
    shutdown_telemetry()


def test_server_sets_request_and_correlation_headers(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def fake_complete_request(self, request):  # type: ignore[no-untyped-def]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", fake_complete_request)
    caplog.set_level(logging.INFO, logger="subllm.server")
    app = create_app(ServerSettings())
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"x-request-id": "req-123", "x-correlation-id": "corr-456"},
    )

    assert response.status_code == 200
    assert response.headers["x-request-id"] == "req-123"
    assert response.headers["x-correlation-id"] == "corr-456"
    assert any("request.completed" in record.message for record in caplog.records)
    assert any('"request_id": "req-123"' in record.message for record in caplog.records)
    assert any('"correlation_id": "corr-456"' in record.message for record in caplog.records)


def test_server_error_response_includes_generated_request_headers() -> None:
    app = create_app(ServerSettings(auth_token="secret"))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 401
    assert response.headers["x-request-id"]
    assert response.headers["x-correlation-id"] == response.headers["x-request-id"]


@pytest.mark.asyncio
async def test_router_exports_local_spans_with_request_context(tmp_path: Path) -> None:
    trace_path = tmp_path / "spans.jsonl"
    configure_telemetry(TelemetryConfig(service_name="subllm-test", export_path=str(trace_path)))

    router = Router()
    router._providers = {"fake": FakeProvider()}

    request = CompletionRequest.from_mapping(
        {
            "model": "fake/demo",
            "messages": [{"role": "user", "content": "hello"}],
        }
    )

    with bind_request_context(
        make_request_context(request_id="req-local", correlation_id="corr-local")
    ):
        response = await router.complete_request(request)

    shutdown_telemetry()

    assert response.choices[0].message.content == "ok"
    exported = [json.loads(line) for line in trace_path.read_text().splitlines()]
    completion_span = next(span for span in exported if span["name"] == "subllm.completion")
    assert completion_span["attributes"]["gen_ai.system"] == "fake"
    assert completion_span["attributes"]["gen_ai.request.model"] == "fake/demo"
    assert completion_span["attributes"]["subllm.request.id"] == "req-local"
    assert completion_span["attributes"]["subllm.correlation.id"] == "corr-local"
