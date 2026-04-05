from __future__ import annotations

import asyncio

import pytest

from subllm.cache import CacheConfig, ResponseCache, build_cache_key
from subllm.prompts import resolve_prompt
from subllm.types import ChatCompletionResponse, Choice, CompletionRequest, Message


@pytest.mark.asyncio
async def test_response_cache_returns_deep_copied_entry() -> None:
    cache = ResponseCache(CacheConfig(ttl_seconds=60, max_entries=4))
    response = ChatCompletionResponse(
        model="codex/gpt-5.2",
        choices=[Choice(message=Message(role="assistant", content="cached"), finish_reason="stop")],
    )

    await cache.set("key", response)
    cached = await cache.get("key")
    assert cached is not None
    assert cached.choices[0].message.content == "cached"

    cached.choices[0].message.content = "changed"
    cached_again = await cache.get("key")
    assert cached_again is not None
    assert cached_again.choices[0].message.content == "cached"


@pytest.mark.asyncio
async def test_response_cache_expires_entries() -> None:
    cache = ResponseCache(CacheConfig(ttl_seconds=0.01, max_entries=2))
    await cache.set(
        "key",
        ChatCompletionResponse(
            model="codex/gpt-5.2",
            choices=[
                Choice(message=Message(role="assistant", content="cached"), finish_reason="stop")
            ],
        ),
    )

    await asyncio.sleep(0.02)

    assert await cache.get("key") is None


def test_cache_key_includes_prompt_identity_and_request_shape() -> None:
    request = CompletionRequest.from_mapping(
        {
            "model": "codex/gpt-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "prompt": {"name": "chat-default", "version": "v1"},
        }
    )
    prompt = resolve_prompt(name="chat-default", version="v1")
    first = build_cache_key(
        provider_name="codex",
        request=request,
        resolved_prompt=prompt,
        system_prompt=request.compose_system_prompt(prompt.text),
    )
    second_request = CompletionRequest.from_mapping(
        {
            "model": "codex/gpt-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "temperature": 0.2,
            "prompt": {"name": "code-review", "version": "v1"},
        }
    )
    second_prompt = resolve_prompt(name="code-review", version="v1")
    second = build_cache_key(
        provider_name="codex",
        request=second_request,
        resolved_prompt=second_prompt,
        system_prompt=second_request.compose_system_prompt(second_prompt.text),
    )

    assert first != second
