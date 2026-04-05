"""SubLLM Router — dispatches completion calls to the appropriate provider.

Handles model resolution, provider selection, and batch execution.
Multi-turn conversations use stateless message replay (all providers).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Mapping, Sequence

from subllm.errors import UnknownModelError, UnsupportedFeatureError
from subllm.providers.base import Provider, ProviderCapabilities
from subllm.providers.claude_code import ClaudeCodeProvider
from subllm.providers.codex import CodexProvider
from subllm.providers.gemini_cli import GeminiCLIProvider
from subllm.types import (
    AuthStatus,
    ChatCompletionChunk,
    ChatCompletionResponse,
    CompletionRequest,
    ModelDescriptor,
)


class Router:
    """Routes model requests to the correct provider based on model prefix."""

    def __init__(self, *, auth_cache_ttl: float = 300.0) -> None:
        self._providers: dict[str, Provider] = {}
        self._auth_cache: list[AuthStatus] | None = None
        self._auth_cache_time: float = 0.0
        self._auth_cache_ttl: float = auth_cache_ttl
        self.register(ClaudeCodeProvider())
        self.register(CodexProvider())
        self.register(GeminiCLIProvider())

    def register(self, provider: Provider) -> None:
        """Register a provider. Use for custom/third-party providers."""
        self._providers[provider.name] = provider

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def list_models(self) -> list[ModelDescriptor]:
        models: list[ModelDescriptor] = []
        for pname, provider in self._providers.items():
            for m in provider.supported_models:
                models.append({"id": f"{pname}/{m}", "provider": pname})
        return models

    async def complete_request(self, request: CompletionRequest) -> ChatCompletionResponse:
        provider, model_alias = self._resolve(request.model)
        return await provider.complete(
            request.provider_messages,
            model_alias,
            system_prompt=request.effective_system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

    def stream_request(self, request: CompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        provider, model_alias = self._resolve(request.model)
        return provider.stream(
            request.provider_messages,
            model_alias,
            system_prompt=request.effective_system_prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
        )

    def supported_model_ids(self, provider_name: str | None = None) -> list[str]:
        models = self.list_models()
        if provider_name is None:
            return sorted(model["id"] for model in models)
        return sorted(model["id"] for model in models if model["provider"] == provider_name)

    def get_capabilities(self, provider_name: str) -> ProviderCapabilities | None:
        provider = self._providers.get(provider_name)
        return provider.capabilities if provider else None

    async def check_auth(self, *, force: bool = False) -> list[AuthStatus]:
        """Check auth for all providers concurrently, with TTL caching.

        Args:
            force: Bypass cache and re-run all auth checks.

        Returns:
            Auth status for each registered provider, in registration order.
        """
        now = time.monotonic()
        if (
            not force
            and self._auth_cache is not None
            and (now - self._auth_cache_time) < self._auth_cache_ttl
        ):
            return self._auth_cache

        results = list(
            await asyncio.gather(*(provider.check_auth() for provider in self._providers.values()))
        )
        self._auth_cache = results
        self._auth_cache_time = time.monotonic()
        return results

    async def check_auth_provider(self, provider_name: str) -> AuthStatus:
        """Check auth for a single provider by name.

        Args:
            provider_name: Registered provider name (e.g. "claude-code", "codex", "gemini").

        Returns:
            Auth status for the specified provider.

        Raises:
            ValueError: If provider_name is not registered.
        """
        provider = self._providers.get(provider_name)
        if provider is None:
            raise ValueError(
                f"Unknown provider '{provider_name}'. Available: {', '.join(self._providers)}"
            )
        return await provider.check_auth()

    async def close(self) -> None:
        """Close all provider connections (SDK clients, etc.)."""
        for provider in self._providers.values():
            await provider.close()

    def _resolve(self, model: str) -> tuple[Provider, str]:
        """Parse 'provider/model' and return the validated provider and alias."""
        if "/" not in model:
            raise UnknownModelError(
                model=model,
                supported_models=self.supported_model_ids(),
                detail=(
                    f"Unsupported model '{model}'. Model IDs must include a provider prefix. "
                    f"Use one of: {', '.join(self.supported_model_ids())}"
                ),
            )

        provider_name, model_alias = model.split("/", 1)
        provider = self._providers.get(provider_name)
        if provider is None:
            raise UnknownModelError(model=model, supported_models=self.supported_model_ids())

        if model_alias not in provider.supported_models:
            raise UnknownModelError(
                model=model,
                supported_models=self.supported_model_ids(provider_name),
                detail=(
                    f"Unsupported model '{model}'. Supported models for provider "
                    f"'{provider_name}': {', '.join(self.supported_model_ids(provider_name))}"
                ),
            )

        return provider, model_alias

    async def completion(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        request = CompletionRequest.from_inputs(
            model=model,
            messages=messages,
            stream=stream,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        if request.stream:
            return self.stream_request(request)

        return await self.complete_request(request)

    async def batch(
        self,
        requests: list[dict[str, Any] | CompletionRequest],
        *,
        concurrency: int = 3,
    ) -> list[ChatCompletionResponse | Exception]:
        """Execute multiple completions concurrently with a semaphore.

        Each request dict: model, messages, and optionally system_prompt, max_tokens, temperature.
        Returns results in same order. Failed requests return Exception instead of raising.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(
            req: dict[str, Any] | CompletionRequest,
        ) -> ChatCompletionResponse | Exception:
            async with sem:
                try:
                    request = (
                        req
                        if isinstance(req, CompletionRequest)
                        else CompletionRequest.from_mapping(req)
                    )
                    if request.stream:
                        raise UnsupportedFeatureError(
                            field="stream",
                            message="Batch requests do not support streaming responses",
                        )
                    return await self.complete_request(request)
                except Exception as e:
                    return e

        tasks = [_run_one(r) for r in requests]
        return await asyncio.gather(*tasks)


# ── Module-level singleton & convenience functions ─────────────────

_router = Router()


async def completion(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    stream: bool = False,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
    """OpenAI-compatible completion function.

    Usage:
        response = await subllm.completion(
            model="claude-code/sonnet-4-5",
            messages=[{"role": "user", "content": "Hello"}],
        )

        async for chunk in await subllm.completion(
            model="gemini/gemini-3-flash-preview",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        ):
            print(chunk.choices[0].delta.content, end="")
    """
    return await _router.completion(
        model=model,
        messages=messages,
        stream=stream,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )


async def complete_completion(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ChatCompletionResponse:
    request = CompletionRequest.from_inputs(
        model=model,
        messages=messages,
        stream=False,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return await _router.complete_request(request)


def stream_completion(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    request = CompletionRequest.from_inputs(
        model=model,
        messages=messages,
        stream=True,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return _router.stream_request(request)


async def batch(
    requests: list[dict[str, Any] | CompletionRequest],
    *,
    concurrency: int = 3,
) -> list[ChatCompletionResponse | Exception]:
    """Execute multiple completions concurrently.

    Usage:
        results = await subllm.batch([
            {"model": "claude-code/sonnet-4-5", "messages": [...]},
            {"model": "gemini/gemini-3-flash-preview", "messages": [...]},
            {"model": "codex/gpt-5.2", "messages": [...]},
        ], concurrency=5)
    """
    return await _router.batch(requests, concurrency=concurrency)


def list_models() -> list[ModelDescriptor]:
    return _router.list_models()


async def check_auth(*, force: bool = False) -> list[AuthStatus]:
    """Check auth for all providers (parallel, cached with TTL)."""
    return await _router.check_auth(force=force)


async def check_auth_provider(provider_name: str) -> AuthStatus:
    """Check auth for a single provider by name."""
    return await _router.check_auth_provider(provider_name)


async def close() -> None:
    """Close all provider connections on the module-level router."""
    await _router.close()


def get_router() -> Router:
    """Get the module-level router for advanced configuration."""
    return _router
