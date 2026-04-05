from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
import pytest

from subllm.server import create_app
from subllm.server_api.settings import ServerSettings
from subllm.types import ChatCompletionChunk, ChatCompletionResponse, Choice, Delta, Message, StreamChoice


def test_sync_completion_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete_request(self, request):  # type: ignore[no-untyped-def]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", fake_complete_request)
    client = TestClient(create_app(ServerSettings()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"x-request-id": "req-int-1"},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
    assert response.headers["x-request-id"] == "req-int-1"
    assert response.headers["x-correlation-id"] == "req-int-1"


def test_stream_completion_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_stream_request(self, request) -> AsyncIterator[ChatCompletionChunk]:  # type: ignore[no-untyped-def]
        yield ChatCompletionChunk(
            model=request.model,
            choices=[StreamChoice(delta=Delta(role="assistant", content="first"))],
        )
        yield ChatCompletionChunk(
            model=request.model,
            choices=[StreamChoice(delta=Delta(content="second"))],
        )
        yield ChatCompletionChunk(
            model=request.model,
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.stream_request", fake_stream_request)
    client = TestClient(create_app(ServerSettings()))

    with client.stream(
        "POST",
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
        headers={"x-correlation-id": "corr-int-2"},
    ) as response:
        body = "".join(response.iter_text())
        headers = dict(response.headers)

    assert response.status_code == 200
    assert "first" in body
    assert "second" in body
    assert "data: [DONE]" in body
    assert headers["x-correlation-id"] == "corr-int-2"
    assert headers["x-request-id"]


def test_invalid_payload_integration() -> None:
    client = TestClient(create_app(ServerSettings()))

    response = client.post(
        "/v1/chat/completions",
        content=b"{invalid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_request"
    assert response.headers["x-request-id"]


def test_auth_rejection_integration() -> None:
    client = TestClient(create_app(ServerSettings(auth_token="secret")))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 401
    assert response.json()["error"]["code"] == "authentication_error"
    assert response.headers["x-request-id"]


def test_timeout_integration(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow_complete_request(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.05)
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", slow_complete_request)
    client = TestClient(create_app(ServerSettings(request_timeout_seconds=0.01)))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"
    assert response.headers["x-request-id"]
