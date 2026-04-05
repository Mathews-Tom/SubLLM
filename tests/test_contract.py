from __future__ import annotations

from fastapi.testclient import TestClient
import pytest

from subllm.errors import PromptRenderError, UnknownPromptError, UnsupportedFeatureError
from subllm.server import create_app
from subllm.types import ChatCompletionResponse, Choice, CompletionRequest, Message


def test_completion_request_rejects_unsupported_fields() -> None:
    with pytest.raises(UnsupportedFeatureError) as exc_info:
        CompletionRequest.from_mapping(
            {
                "model": "claude-code/sonnet-4-5",
                "messages": [{"role": "user", "content": "hello"}],
                "tools": [{"type": "function"}],
            }
        )

    assert exc_info.value.param == "request"
    assert "tools" in exc_info.value.message


def test_completion_request_extracts_system_prompt_and_provider_messages() -> None:
    request = CompletionRequest.from_mapping(
        {
            "model": "claude-code/sonnet-4-5",
            "messages": [
                {"role": "system", "content": "message system prompt"},
                {"role": "user", "content": "hello"},
            ],
            "system_prompt": "top level system prompt",
        }
    )

    assert request.effective_system_prompt == "top level system prompt\n\nmessage system prompt"
    assert request.provider_messages == [{"role": "user", "content": "hello"}]


def test_completion_request_tracks_prompt_reference() -> None:
    request = CompletionRequest.from_mapping(
        {
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt": {"name": "chat-default", "version": "v1"},
        }
    )

    assert request.prompt is not None
    assert request.prompt.name == "chat-default"
    assert request.prompt.version == "v1"
    assert request.compose_system_prompt("registered prompt") == "registered prompt"


def test_chat_completions_returns_openai_style_error_for_unsupported_fields() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "tools": [{"type": "function"}],
        },
    )

    assert response.status_code == 400
    assert response.json() == {
        "error": {
            "message": "Unsupported request fields: tools",
            "type": "invalid_request_error",
            "param": "request",
            "code": "unsupported_feature",
        }
    }


def test_chat_completions_rejects_unknown_prompt_reference() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt": {"name": "missing-prompt"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "prompt_not_found"
    assert response.json()["error"]["param"] == "prompt"


def test_chat_completions_rejects_prompt_variable_mismatch() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt": {"name": "release-notes", "version": "v1"},
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "prompt_render_error"


def test_known_prompt_errors_are_exported() -> None:
    assert UnknownPromptError(prompt_name="x").code == "prompt_not_found"
    assert PromptRenderError(message="bad").code == "prompt_render_error"


def test_chat_completions_uses_typed_request_contract(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete_request(self, request):  # type: ignore[no-untyped-def]
        assert request.model == "claude-code/sonnet-4-5"
        assert request.provider_messages == [{"role": "user", "content": "hello"}]
        assert request.effective_system_prompt == "system message"
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", fake_complete_request)
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [
                {"role": "system", "content": "system message"},
                {"role": "user", "content": "hello"},
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"
