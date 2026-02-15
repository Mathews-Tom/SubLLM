"""SubLLM — Route LLM API calls through subscription-authenticated coding agents.

Supports Claude Code, OpenAI Codex, and Gemini CLI as backends.

Usage:
    import subllm

    # Non-streaming
    response = await subllm.completion(
        model="claude-code/sonnet-4-5",
        messages=[{"role": "user", "content": "Hello"}],
    )
    print(response.choices[0].message.content)

    # Streaming
    async for chunk in await subllm.completion(
        model="gemini/gemini-3-flash-preview",
        messages=[{"role": "user", "content": "Hello"}],
        stream=True,
    ):
        print(chunk.choices[0].delta.content, end="")

    # Batch (parallel across providers)
    results = await subllm.batch([
        {"model": "claude-code/sonnet-4-5", "messages": [...]},
        {"model": "gemini/gemini-3-flash-preview", "messages": [...]},
        {"model": "codex/gpt-5.2", "messages": [...]},
    ], concurrency=5)
"""

from __future__ import annotations

from subllm.providers.base import Provider, ProviderCapabilities
from subllm.router import (
    Router,
    batch,
    check_auth,
    check_auth_provider,
    close,
    completion,
    get_router,
    list_models,
)
from subllm.types import AuthStatus, ChatCompletionChunk, ChatCompletionResponse

__version__ = "0.4.0"

__all__ = [
    "Provider",
    "ProviderCapabilities",
    "Router",
    "batch",
    "check_auth",
    "check_auth_provider",
    "close",
    "completion",
    "get_router",
    "list_models",
    "AuthStatus",
    "ChatCompletionChunk",
    "ChatCompletionResponse",
]
