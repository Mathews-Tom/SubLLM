from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from subllm.providers.codex import CodexProvider
from subllm.server import create_app
from subllm.types import (
    ChatCompletionResponse,
    Choice,
    Message,
    ProviderMessage,
    ResolvedImageInput,
)

PNG_DATA_URL = (
    "data:image/png;base64,"
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO7Z0s8AAAAASUVORK5CYII="
)


def test_server_rejects_image_inputs_for_provider_without_vision() -> None:
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "claude-code/sonnet-4-5",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "describe"},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 400
    assert response.json()["error"]["code"] == "unsupported_feature"
    assert "does not support image inputs" in response.json()["error"]["message"]


def test_codex_provider_receives_resolved_images(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete(self, messages, model, **kwargs):  # type: ignore[no-untyped-def]
        assert messages[0]["content"].startswith("look at this")
        assert len(messages[0]["images"]) == 1
        assert messages[0]["images"][0].media_type == "image/png"
        return ChatCompletionResponse(
            model=f"codex/{model}",
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    monkeypatch.setattr(CodexProvider, "complete", fake_complete)
    client = TestClient(create_app())

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "codex/gpt-5.2",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "look at this"},
                        {"type": "image_url", "image_url": {"url": PNG_DATA_URL}},
                    ],
                }
            ],
        },
    )

    assert response.status_code == 200
    assert response.json()["choices"][0]["message"]["content"] == "ok"


@pytest.mark.asyncio
async def test_codex_provider_materializes_embedded_images() -> None:
    provider = CodexProvider(cli_path="codex")
    image = ResolvedImageInput(
        filename="embedded.png",
        media_type="image/png",
        data=b"png-bytes",
    )
    messages: list[ProviderMessage] = [{"role": "user", "content": "hello", "images": [image]}]

    async with provider._prepared_image_paths(messages) as image_paths:
        assert len(image_paths) == 1
        path = Path(image_paths[0])
        assert path.exists()
        assert path.read_bytes() == b"png-bytes"

    assert not path.exists()
