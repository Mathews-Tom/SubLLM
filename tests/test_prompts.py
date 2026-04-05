from __future__ import annotations

import pytest

from subllm.errors import PromptRenderError, UnknownPromptError
from subllm.prompts import PromptRegistry, get_prompt_registry, list_registered_prompts
from subllm.router import Router
from subllm.types import (
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    CompletionRequest,
    Delta,
    Message,
    StreamChoice,
)


def test_default_prompt_registry_exposes_versions_and_metadata() -> None:
    prompts = list_registered_prompts()

    assert [prompt.name for prompt in prompts] == [
        "chat-default",
        "code-review",
        "release-notes",
    ]
    release_notes = get_prompt_registry().resolve(
        name="release-notes",
        version="v1",
        variables={"audience": "operators"},
    )
    assert release_notes.version == "v1"
    assert "operators" in release_notes.text
    assert release_notes.metadata["category"] == "documentation"


def test_prompt_registry_rejects_unknown_versions_and_bad_variables() -> None:
    registry = PromptRegistry()
    registry.register(
        name="deploy-summary",
        version="v1",
        template="Ship notes for {audience}",
        variables=("audience",),
        default=True,
    )

    with pytest.raises(UnknownPromptError):
        registry.resolve(name="deploy-summary", version="v2")

    with pytest.raises(PromptRenderError):
        registry.resolve(name="deploy-summary", version="v1")

    with pytest.raises(PromptRenderError):
        registry.resolve(
            name="deploy-summary",
            version="v1",
            variables={"audience": "ops", "extra": "nope"},
        )


@pytest.mark.asyncio
async def test_router_composes_registered_prompt_with_request_system_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = Router()

    async def fake_complete(self, messages, model, **kwargs):  # type: ignore[no-untyped-def]
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs["system_prompt"] is not None
        assert "You are a precise, direct assistant." in kwargs["system_prompt"]
        assert "extra system prompt" in kwargs["system_prompt"]
        return ChatCompletionResponse(
            model=f"codex/{model}",
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    provider, _ = router._resolve("codex/gpt-5.2")
    monkeypatch.setattr(provider, "complete", fake_complete.__get__(provider, type(provider)))

    response = await router.complete_request(
        CompletionRequest.from_mapping(
            {
                "model": "codex/gpt-5.2",
                "messages": [{"role": "user", "content": "hello"}],
                "system_prompt": "extra system prompt",
                "prompt": {"name": "chat-default"},
            }
        )
    )

    assert response.choices[0].message.content == "ok"


@pytest.mark.asyncio
async def test_router_stream_uses_registered_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = Router()

    async def fake_stream(self, messages, model, **kwargs):  # type: ignore[no-untyped-def]
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs["system_prompt"] is not None
        assert "Review the code with a strict engineering lens." in kwargs["system_prompt"]
        yield ChatCompletionChunk(
            model=f"codex/{model}",
            choices=[StreamChoice(delta=Delta(role="assistant", content="ok"))],
        )

    provider, _ = router._resolve("codex/gpt-5.2")
    monkeypatch.setattr(provider, "stream", fake_stream.__get__(provider, type(provider)))

    request = CompletionRequest.from_mapping(
        {
            "model": "codex/gpt-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "prompt": {"name": "code-review"},
            "stream": True,
        }
    )
    chunks = [chunk async for chunk in router.stream_request(request)]

    assert chunks[0].choices[0].delta.content == "ok"
