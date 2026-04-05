from __future__ import annotations

from collections.abc import AsyncIterator

from fastapi.testclient import TestClient
import pytest

from subllm.errors import UnknownModelError
from subllm.server import create_app
from subllm.server_api.errors import build_error_response
from subllm.server_api.responses import build_model_list
from subllm.types import ChatCompletionChunk, Delta, StreamChoice


def test_build_model_list_returns_typed_response() -> None:
    payload = build_model_list(
        [
            {"id": "claude-code/sonnet-4-5", "provider": "claude-code"},
            {"id": "codex/gpt-5.2", "provider": "codex"},
        ]
    )

    dumped = payload.model_dump()
    assert dumped["object"] == "list"
    assert dumped["data"][0]["owned_by"] == "claude-code"
    assert dumped["data"][1]["id"] == "codex/gpt-5.2"


def test_build_error_response_wraps_subllm_errors() -> None:
    response = build_error_response(
        UnknownModelError(
            model="missing",
            supported_models=["claude-code/sonnet-4-5"],
        )
    )

    assert response.model_dump() == {
        "error": {
            "message": "Unsupported model 'missing'. Use one of: claude-code/sonnet-4-5",
            "type": "invalid_request_error",
            "param": "model",
            "code": "model_not_found",
        }
    }


def test_chat_completions_returns_structured_model_error() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={"model": "missing", "messages": [{"role": "user", "content": "hello"}]},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "model_not_found"
    assert response.json()["error"]["param"] == "model"


def test_chat_completions_rejects_non_object_json() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json=[{"model": "claude-code/sonnet-4-5"}],
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Request body must be a JSON object",
            "type": "invalid_request_error",
            "param": None,
            "code": "malformed_request",
        }
    }


def test_chat_completions_rejects_invalid_json_payload() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        content=b"{invalid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_request"


def test_chat_completions_streams_with_explicit_stream_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_stream_request(self, request) -> AsyncIterator[ChatCompletionChunk]:  # type: ignore[no-untyped-def]
        assert request.stream is True
        yield ChatCompletionChunk(
            model=request.model,
            choices=[StreamChoice(delta=Delta(role="assistant", content="hello"))],
        )
        yield ChatCompletionChunk(
            model=request.model,
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.stream_request", fake_stream_request)
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "stream": True,
        },
    )

    assert response.status_code == 200
    assert 'data: {"' in response.text
    assert "data: [DONE]" in response.text
