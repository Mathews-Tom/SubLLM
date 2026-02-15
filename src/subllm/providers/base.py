"""Abstract base for SubLLM providers."""

from __future__ import annotations

import abc
from collections.abc import AsyncIterator
from dataclasses import dataclass

from subllm.types import AuthStatus, ChatCompletionChunk, ChatCompletionResponse


def messages_to_prompt(messages: list[dict], system_prompt: str | None = None) -> str:
    """Flatten OpenAI-style messages into a single prompt string for CLI mode."""
    parts: list[str] = []
    if system_prompt:
        parts.append(f"[System: {system_prompt}]")
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            parts.append(f"[System: {content}]")
        elif role == "assistant":
            parts.append(f"[Assistant: {content}]")
        else:
            parts.append(content)
    return "\n\n".join(parts)


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 chars per token)."""
    return len(text) // 4


@dataclass
class ProviderCapabilities:
    """Declares what a provider supports — router adapts behavior accordingly.

    When a provider doesn't support a capability (e.g., sessions), the router
    compensates automatically (e.g., stateless message replay for multi-turn).
    """

    supports_streaming: bool = True
    supports_sessions: bool = False
    supports_system_prompt: bool = True
    supports_tool_use: bool = False  # Tool use executes inside CLI sandbox, not exposed
    supports_batch: bool = False
    supports_vision: bool = False
    max_context_tokens: int = 200_000
    subscription_auth: bool = True
    api_key_auth: bool = True


class Provider(abc.ABC):
    """Base class for all SubLLM providers.

    To add a new provider:
      1. Subclass Provider
      2. Implement all abstract methods
      3. Override `capabilities` to declare what you support
      4. Register via `router.register(MyProvider())`
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        ...

    @property
    @abc.abstractmethod
    def supported_models(self) -> list[str]:
        ...

    @property
    def capabilities(self) -> ProviderCapabilities:
        """Override in subclasses to declare specific capabilities."""
        return ProviderCapabilities()

    @abc.abstractmethod
    async def check_auth(self) -> AuthStatus:
        ...

    @abc.abstractmethod
    async def complete(
        self,
        messages: list[dict],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse:
        ...

    @abc.abstractmethod
    async def stream(
        self,
        messages: list[dict],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        ...

    def resolve_model(self, model_alias: str) -> str:
        """Map a SubLLM model alias (e.g., 'sonnet') to the provider's model string."""
        return model_alias
