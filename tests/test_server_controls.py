from __future__ import annotations

import asyncio

from fastapi.testclient import TestClient
import pytest

from subllm.cli import _is_local_host, _run_server
from subllm.server import create_app
from subllm.server_api.settings import ServerSettings
from subllm.types import ChatCompletionResponse, Choice, Message


def test_server_requires_bearer_token_when_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete_request(self, request):  # type: ignore[no-untyped-def]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", fake_complete_request)
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
    assert response.json()["error"]["code"] == "authentication_error"


def test_server_accepts_valid_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete_request(self, request):  # type: ignore[no-untyped-def]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", fake_complete_request)
    app = create_app(ServerSettings(auth_token="secret"))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
        headers={"authorization": "Bearer secret"},
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


def test_server_rejects_oversized_request_body() -> None:
    app = create_app(ServerSettings(max_request_bytes=10))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"


def test_server_rate_limits_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete_request(self, request):  # type: ignore[no-untyped-def]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", fake_complete_request)
    app = create_app(ServerSettings(rate_limit_per_minute=1))
    client = TestClient(app)
    payload = {
        "model": "claude-code/sonnet-4-5",
        "messages": [{"role": "user", "content": "hello"}],
    }

    first = client.post("/v1/chat/completions", json=payload)
    second = client.post("/v1/chat/completions", json=payload)

    assert first.status_code == 200
    assert second.status_code == 429
    assert second.json()["error"]["code"] == "rate_limit_exceeded"


def test_server_request_timeout_returns_structured_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def slow_complete_request(self, request):  # type: ignore[no-untyped-def]
        await asyncio.sleep(0.05)
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", slow_complete_request)
    app = create_app(ServerSettings(request_timeout_seconds=0.01))
    client = TestClient(app)

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
        },
    )

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "request_timeout"


def test_is_local_host_handles_loopback_variants() -> None:
    assert _is_local_host("127.0.0.1")
    assert _is_local_host("::1")
    assert _is_local_host("localhost")
    assert not _is_local_host("0.0.0.0")


def test_server_refuses_non_local_host_without_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(SystemExit) as exc_info:
        _run_server(
            "0.0.0.0",
            8080,
            auth_token=None,
            max_request_bytes=None,
            request_timeout_seconds=None,
            rate_limit_per_minute=None,
        )

    assert exc_info.value.code == 1
