"""SubLLM Router — dispatches completion calls to the appropriate provider.

Handles model resolution, provider selection, and batch execution.
Multi-turn conversations use stateless message replay (all providers).
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from subllm.providers.base import Provider, ProviderCapabilities
from subllm.providers.claude_code import ClaudeCodeProvider
from subllm.providers.codex import CodexProvider
from subllm.providers.gemini_cli import GeminiCLIProvider
from subllm.types import AuthStatus, ChatCompletionChunk, ChatCompletionResponse


class Router:
    """Routes model requests to the correct provider based on model prefix."""

    def __init__(self) -> None:
        self._providers: dict[str, Provider] = {}
        self.register(ClaudeCodeProvider())
        self.register(CodexProvider())
        self.register(GeminiCLIProvider())

    def register(self, provider: Provider) -> None:
        """Register a provider. Use for custom/third-party providers."""
        self._providers[provider.name] = provider

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())

    def list_models(self) -> list[dict]:
        models = []
        for pname, provider in self._providers.items():
            for m in provider.supported_models:
                models.append({"id": f"{pname}/{m}", "provider": pname})
        return models

    def get_capabilities(self, provider_name: str) -> ProviderCapabilities | None:
        provider = self._providers.get(provider_name)
        return provider.capabilities if provider else None

    async def check_auth(self) -> list[AuthStatus]:
        results = []
        for provider in self._providers.values():
            results.append(await provider.check_auth())
        return results

    def _resolve(self, model: str) -> tuple[Provider, str]:
        """Parse 'provider/model' and return (provider_instance, model_alias)."""
        if "/" in model:
            prefix, model_alias = model.split("/", 1)
            if prefix in self._providers:
                return self._providers[prefix], model_alias

        for name in ["claude-code", "codex", "gemini"]:
            if name in self._providers:
                return self._providers[name], model

        raise ValueError(
            f"No provider found for model '{model}'. "
            f"Available: {', '.join(self.list_providers())}"
        )

    async def completion(
        self,
        model: str,
        messages: list[dict],
        *,
        stream: bool = False,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
        provider, model_alias = self._resolve(model)

        if stream:
            return provider.stream(
                messages, model_alias,
                system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
            )

        return await provider.complete(
            messages, model_alias,
            system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
        )

    async def batch(
        self,
        requests: list[dict],
        *,
        concurrency: int = 3,
    ) -> list[ChatCompletionResponse | Exception]:
        """Execute multiple completions concurrently with a semaphore.

        Each request dict: model, messages, and optionally system_prompt, max_tokens, temperature.
        Returns results in same order. Failed requests return Exception instead of raising.
        """
        sem = asyncio.Semaphore(concurrency)

        async def _run_one(req: dict) -> ChatCompletionResponse | Exception:
            async with sem:
                try:
                    return await self.completion(
                        model=req["model"],
                        messages=req["messages"],
                        system_prompt=req.get("system_prompt"),
                        max_tokens=req.get("max_tokens"),
                        temperature=req.get("temperature"),
                    )
                except Exception as e:
                    return e

        tasks = [_run_one(r) for r in requests]
        return await asyncio.gather(*tasks)


# ── Module-level singleton & convenience functions ─────────────────

_router = Router()


async def completion(
    model: str,
    messages: list[dict],
    *,
    stream: bool = False,
    system_prompt: str | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> ChatCompletionResponse | AsyncIterator[ChatCompletionChunk]:
    """LiteLLM-style completion function.

    Usage:
        response = await subllm.completion(
            model="claude-code/sonnet",
            messages=[{"role": "user", "content": "Hello"}],
        )

        async for chunk in await subllm.completion(
            model="gemini/flash",
            messages=[{"role": "user", "content": "Hello"}],
            stream=True,
        ):
            print(chunk.choices[0].delta.content, end="")
    """
    return await _router.completion(
        model=model, messages=messages, stream=stream,
        system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
    )


async def batch(
    requests: list[dict],
    *,
    concurrency: int = 3,
) -> list[ChatCompletionResponse | Exception]:
    """Execute multiple completions concurrently.

    Usage:
        results = await subllm.batch([
            {"model": "claude-code/sonnet", "messages": [...]},
            {"model": "gemini/flash", "messages": [...]},
            {"model": "codex/gpt-5.3", "messages": [...]},
        ], concurrency=5)
    """
    return await _router.batch(requests, concurrency=concurrency)


def list_models() -> list[dict]:
    return _router.list_models()


async def check_auth() -> list[AuthStatus]:
    return await _router.check_auth()


def get_router() -> Router:
    """Get the module-level router for advanced configuration."""
    return _router
