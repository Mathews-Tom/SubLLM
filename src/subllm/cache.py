"""Optional response cache for deterministic completion requests."""

from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import time
from collections import OrderedDict
from contextvars import ContextVar
from dataclasses import dataclass

from subllm.prompts import ResolvedPrompt
from subllm.types import ChatCompletionResponse, CompletionRequest

_cache_status_var: ContextVar[str | None] = ContextVar("subllm_cache_status", default=None)


@dataclass(frozen=True)
class CacheConfig:
    ttl_seconds: float
    max_entries: int = 256


@dataclass
class _CacheEntry:
    response: ChatCompletionResponse
    expires_at: float


class ResponseCache:
    def __init__(self, config: CacheConfig) -> None:
        self._config = config
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()

    @property
    def config(self) -> CacheConfig:
        return self._config

    async def get(self, key: str) -> ChatCompletionResponse | None:
        async with self._lock:
            self._prune_expired(now=time.monotonic())
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return copy.deepcopy(entry.response)

    async def set(self, key: str, response: ChatCompletionResponse) -> None:
        async with self._lock:
            now = time.monotonic()
            self._prune_expired(now=now)
            self._entries[key] = _CacheEntry(
                response=copy.deepcopy(response),
                expires_at=now + self._config.ttl_seconds,
            )
            self._entries.move_to_end(key)
            while len(self._entries) > self._config.max_entries:
                self._entries.popitem(last=False)

    def _prune_expired(self, *, now: float) -> None:
        expired_keys = [key for key, entry in self._entries.items() if entry.expires_at <= now]
        for key in expired_keys:
            self._entries.pop(key, None)


def build_cache_key(
    *,
    provider_name: str,
    request: CompletionRequest,
    resolved_prompt: ResolvedPrompt | None,
    system_prompt: str | None,
) -> str:
    payload = {
        "provider": provider_name,
        "model": request.model,
        "messages": request.provider_messages,
        "system_prompt": system_prompt,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "prompt": (
            {
                "name": resolved_prompt.name,
                "version": resolved_prompt.version,
            }
            if resolved_prompt is not None
            else None
        ),
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def clear_cache_status() -> None:
    _cache_status_var.set(None)


def set_cache_status(status: str) -> None:
    _cache_status_var.set(status)


def current_cache_status() -> str | None:
    return _cache_status_var.get()


def response_cache_headers() -> dict[str, str]:
    status = current_cache_status()
    if status is None:
        return {}
    return {"x-subllm-cache": status}
