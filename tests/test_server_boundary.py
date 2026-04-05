from __future__ import annotations

from fastapi.testclient import TestClient

from subllm.errors import UnknownModelError
from subllm.server import create_app
from subllm.server_api.errors import build_error_response
from subllm.server_api.responses import build_model_list


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
        data="{invalid json",
        headers={"content-type": "application/json"},
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "malformed_request"
