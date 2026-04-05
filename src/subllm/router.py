"""SubLLM Router — dispatches completion calls to the appropriate provider.

Handles model resolution, provider selection, and batch execution.
Multi-turn conversations use stateless message replay (all providers).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any, Mapping, Sequence

from subllm.attachments import message_has_images
from subllm.cache import ResponseCache, build_cache_key, set_cache_status
from subllm.errors import UnknownModelError, UnsupportedFeatureError
from subllm.model_registry import (
    all_model_descriptors,
    provider_capabilities,
    provider_model_aliases,
    registered_provider_names,
)
from subllm.prompts import PromptRegistry, ResolvedPrompt, get_prompt_registry
from subllm.providers.base import Provider, ProviderCapabilities
from subllm.providers.claude_code import ClaudeCodeProvider
from subllm.providers.codex import CodexProvider
from subllm.providers.gemini_cli import GeminiCLIProvider
from subllm.telemetry import (
    attach_request_context,
    get_tracer,
    mark_span_failure,
    mark_span_success,
)
from subllm.types import (
    AuthStatus,
    ChatCompletionChunk,
    ChatCompletionResponse,
    CompletionRequest,
    ModelDescriptor,
    ProviderMessage,
)


class Router:
    """Routes model requests to the correct provider based on model prefix."""

    def __init__(
        self,
        *,
        auth_cache_ttl: float = 300.0,
        prompt_registry: PromptRegistry | None = None,
        response_cache: ResponseCache | None = None,
    ) -> None:
        self._providers: dict[str, Provider] = {}
        self._auth_cache: list[AuthStatus] | None = None
        self._auth_cache_time: float = 0.0
        self._auth_cache_ttl: float = auth_cache_ttl
        self._prompt_registry = prompt_registry or get_prompt_registry()
        self._response_cache = response_cache
        self.register(ClaudeCodeProvider())
        self.register(CodexProvider())
        self.register(GeminiCLIProvider())

    def register(self, provider: Provider) -> None:
        """Register a provider. Use for custom/third-party providers."""
        self._providers[provider.name] = provider

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def list_models(self) -> list[ModelDescriptor]:
        models = list(all_model_descriptors())
        registry_providers = set(registered_provider_names())
        for provider_name, provider in self._providers.items():
            if provider_name in registry_providers:
                continue
            for model_alias in provider.supported_models:
                models.append({"id": f"{provider_name}/{model_alias}", "provider": provider_name})
        return models

    async def complete_request(self, request: CompletionRequest) -> ChatCompletionResponse:
        provider, model_alias = self._resolve(request.model)
        resolved_prompt = self._resolve_prompt(request)
        system_prompt = request.compose_system_prompt(
            resolved_prompt.text if resolved_prompt is not None else None
        )
        provider_messages = request.provider_messages
        self._validate_request_capabilities(provider.name, request, provider_messages)
        cache_key = self._build_cache_key(
            provider_name=provider.name,
            request=request,
            resolved_prompt=resolved_prompt,
            system_prompt=system_prompt,
        )
        started = time.perf_counter()
        with get_tracer("subllm.router").start_as_current_span("subllm.completion") as span:
            attach_request_context(span)
            span.set_attribute("gen_ai.system", provider.name)
            span.set_attribute("gen_ai.request.model", request.model)
            span.set_attribute("subllm.model_alias", model_alias)
            span.set_attribute("subllm.request.stream", False)
            span.set_attribute("subllm.session.enabled", request.session is not None)
            if request.session is not None:
                span.set_attribute("subllm.session.mode", request.session.mode)
            self._attach_prompt_attributes(span, resolved_prompt)
            if cache_key is None:
                set_cache_status("bypass")
                span.set_attribute("subllm.cache.enabled", False)
            else:
                cache = self._response_cache
                assert cache is not None
                set_cache_status("miss")
                span.set_attribute("subllm.cache.enabled", True)
                cached_response = await cache.get(cache_key)
                if cached_response is not None:
                    set_cache_status("hit")
                    span.set_attribute("subllm.cache.hit", True)
                    span.set_attribute(
                        "subllm.request.duration_ms", (time.perf_counter() - started) * 1000.0
                    )
                    mark_span_success(span)
                    return cached_response
                span.set_attribute("subllm.cache.hit", False)
            try:
                response = await provider.complete(
                    provider_messages,
                    model_alias,
                    system_prompt=system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    session=request.session,
                )
                if response.usage is not None:
                    span.set_attribute("gen_ai.usage.input_tokens", response.usage.prompt_tokens)
                    span.set_attribute(
                        "gen_ai.usage.output_tokens", response.usage.completion_tokens
                    )
                    span.set_attribute("gen_ai.usage.total_tokens", response.usage.total_tokens)
                span.set_attribute(
                    "subllm.request.duration_ms", (time.perf_counter() - started) * 1000.0
                )
                if cache_key is not None:
                    cache = self._response_cache
                    assert cache is not None
                    await cache.set(cache_key, response)
                mark_span_success(span)
                return response
            except Exception as exc:
                span.set_attribute(
                    "subllm.request.duration_ms", (time.perf_counter() - started) * 1000.0
                )
                mark_span_failure(span, exc)
                raise

    def stream_request(self, request: CompletionRequest) -> AsyncIterator[ChatCompletionChunk]:
        provider, model_alias = self._resolve(request.model)
        return self._stream_request(provider, model_alias, request)

    async def _stream_request(
        self,
        provider: Provider,
        model_alias: str,
        request: CompletionRequest,
    ) -> AsyncIterator[ChatCompletionChunk]:
        resolved_prompt = self._resolve_prompt(request)
        system_prompt = request.compose_system_prompt(
            resolved_prompt.text if resolved_prompt is not None else None
        )
        provider_messages = request.provider_messages
        self._validate_request_capabilities(provider.name, request, provider_messages)
        started = time.perf_counter()
        chunk_count = 0
        with get_tracer("subllm.router").start_as_current_span("subllm.stream") as span:
            attach_request_context(span)
            span.set_attribute("gen_ai.system", provider.name)
            span.set_attribute("gen_ai.request.model", request.model)
            span.set_attribute("subllm.model_alias", model_alias)
            span.set_attribute("subllm.request.stream", True)
            span.set_attribute("subllm.session.enabled", request.session is not None)
            if request.session is not None:
                span.set_attribute("subllm.session.mode", request.session.mode)
            self._attach_prompt_attributes(span, resolved_prompt)
            try:
                async for chunk in provider.stream(
                    provider_messages,
                    model_alias,
                    system_prompt=system_prompt,
                    max_tokens=request.max_tokens,
                    temperature=request.temperature,
                    session=request.session,
                ):
                    chunk_count += 1
                    yield chunk
                span.set_attribute("subllm.stream.chunk_count", chunk_count)
                span.set_attribute(
                    "subllm.request.duration_ms", (time.perf_counter() - started) * 1000.0
                )
                mark_span_success(span)
            except Exception as exc:
                span.set_attribute("subllm.stream.chunk_count", chunk_count)
                span.set_attribute(
                    "subllm.request.duration_ms", (time.perf_counter() - started) * 1000.0
                )
                mark_span_failure(span, exc)
                raise

    def supported_model_ids(self, provider_name: str | None = None) -> list[str]:
        models = self.list_models()
        if provider_name is None:
            return sorted(model["id"] for model in models)
        return sorted(model["id"] for model in models if model["provider"] == provider_name)

    def get_capabilities(self, provider_name: str) -> ProviderCapabilities | None:
        capabilities = provider_capabilities(provider_name)
        if capabilities is not None:
            return capabilities
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

        supported_models = provider_model_aliases(provider_name)
        if not supported_models:
            supported_models = provider.supported_models

        if model_alias not in supported_models:
            raise UnknownModelError(
                model=model,
                supported_models=self.supported_model_ids(provider_name),
                detail=(
                    f"Unsupported model '{model}'. Supported models for provider "
                    f"'{provider_name}': {', '.join(self.supported_model_ids(provider_name))}"
                ),
            )

        return provider, model_alias

    def _resolve_prompt(self, request: CompletionRequest) -> ResolvedPrompt | None:
        if request.prompt is None:
            return None
        return self._prompt_registry.resolve(
            name=request.prompt.name,
            version=request.prompt.version,
            variables=request.prompt.variables,
        )

    def _attach_prompt_attributes(
        self,
        span: Any,
        resolved_prompt: ResolvedPrompt | None,
    ) -> None:
        if resolved_prompt is None:
            return
        span.set_attribute("subllm.prompt.name", resolved_prompt.name)
        span.set_attribute("subllm.prompt.version", resolved_prompt.version)

    def _validate_request_capabilities(
        self,
        provider_name: str,
        request: CompletionRequest,
        messages: list[ProviderMessage],
    ) -> None:
        capabilities = self.get_capabilities(provider_name)
        if capabilities is None:
            return
        if request.session is not None and not capabilities.supports_sessions:
            raise UnsupportedFeatureError(
                field="session",
                message=f"Provider '{provider_name}' does not support explicit session mode",
            )
        if (
            any(message_has_images(message) for message in messages)
            and not capabilities.supports_vision
        ):
            raise UnsupportedFeatureError(
                field="messages",
                message=f"Provider '{provider_name}' does not support image inputs",
            )

    def _build_cache_key(
        self,
        *,
        provider_name: str,
        request: CompletionRequest,
        resolved_prompt: ResolvedPrompt | None,
        system_prompt: str | None,
    ) -> str | None:
        if self._response_cache is None or request.stream or request.session is not None:
            return None
        return build_cache_key(
            provider_name=provider_name,
            request=request,
            resolved_prompt=resolved_prompt,
            system_prompt=system_prompt,
        )

    async def completion(
        self,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prompt: Mapping[str, Any] | None = None,
        session: Mapping[str, Any] | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        request = CompletionRequest.from_inputs(
            model=model,
            messages=messages,
            stream=stream,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            prompt=prompt,
            session=session,
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
                    if request.session is not None:
                        raise UnsupportedFeatureError(
                            field="session",
                            message="Batch requests do not support explicit session mode",
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
    prompt: Mapping[str, Any] | None = None,
    session: Mapping[str, Any] | None = None,
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
        prompt=prompt,
        session=session,
    )


async def complete_completion(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    prompt: Mapping[str, Any] | None = None,
    session: Mapping[str, Any] | None = None,
) -> ChatCompletionResponse:
    request = CompletionRequest.from_inputs(
        model=model,
        messages=messages,
        stream=False,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        prompt=prompt,
        session=session,
    )
    return await _router.complete_request(request)


def stream_completion(
    model: str,
    messages: Sequence[Mapping[str, Any]],
    *,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    prompt: Mapping[str, Any] | None = None,
    session: Mapping[str, Any] | None = None,
) -> AsyncIterator[ChatCompletionChunk]:
    request = CompletionRequest.from_inputs(
        model=model,
        messages=messages,
        stream=True,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        prompt=prompt,
        session=session,
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
