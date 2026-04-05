from __future__ import annotations

import pytest

from subllm.errors import UnknownModelError, UnsupportedFeatureError
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


def test_router_rejects_unprefixed_model_ids() -> None:
    router = Router()

    with pytest.raises(UnknownModelError) as exc_info:
        router._resolve("sonnet-4-5")

    assert exc_info.value.param == "model"
    assert "provider prefix" in exc_info.value.message


def test_router_rejects_unknown_provider_model_pair() -> None:
    router = Router()

    with pytest.raises(UnknownModelError) as exc_info:
        router._resolve("codex/not-a-real-model")

    assert exc_info.value.param == "model"
    assert "Supported models for provider 'codex'" in exc_info.value.message


def test_router_resolves_supported_prefixed_model_ids() -> None:
    router = Router()

    provider, model_alias = router._resolve("codex/gpt-5.2")

    assert provider.name == "codex"
    assert model_alias == "gpt-5.2"


@pytest.mark.asyncio
async def test_router_batch_rejects_stream_requests() -> None:
    router = Router()

    results = await router.batch(
        [
            {
                "model": "codex/gpt-5.2",
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            }
        ]
    )

    assert len(results) == 1
    assert isinstance(results[0], UnsupportedFeatureError)
    assert results[0].param == "stream"


@pytest.mark.asyncio
async def test_router_exposes_explicit_complete_and_stream_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    router = Router()

    async def fake_complete(self, messages, model, **kwargs):  # type: ignore[no-untyped-def]
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs["system_prompt"] == "system"
        return ChatCompletionResponse(
            model=f"codex/{model}",
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
        )

    async def fake_stream(self, messages, model, **kwargs):  # type: ignore[no-untyped-def]
        assert messages == [{"role": "user", "content": "hello"}]
        assert kwargs["system_prompt"] == "system"
        yield ChatCompletionChunk(
            model=f"codex/{model}",
            choices=[StreamChoice(delta=Delta(role="assistant", content="ok"))],
        )

    provider, _ = router._resolve("codex/gpt-5.2")
    monkeypatch.setattr(provider, "complete", fake_complete.__get__(provider, type(provider)))
    monkeypatch.setattr(provider, "stream", fake_stream.__get__(provider, type(provider)))

    request = {
        "model": "codex/gpt-5.2",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "hello"},
        ],
    }

    completion_result = await router.complete_request(CompletionRequest.from_mapping(request))
    stream_result = router.stream_request(
        CompletionRequest.from_mapping({**request, "stream": True})
    )
    chunks = [chunk async for chunk in stream_result]

    assert completion_result.choices[0].message.content == "ok"
    assert chunks[0].choices[0].delta.content == "ok"
