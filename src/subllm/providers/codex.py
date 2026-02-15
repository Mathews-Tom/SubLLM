"""Codex provider — routes LLM calls through the OpenAI Codex CLI.

Uses `codex exec` for non-interactive headless execution.
Auth: ChatGPT subscription OAuth (from `codex login`) or OPENAI_API_KEY.
Codex CLI is a Rust binary (~55ms cold start) — fastest provider to spawn.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from collections.abc import AsyncIterator

from subllm.providers.base import Provider, ProviderCapabilities, estimate_tokens, messages_to_prompt
from subllm.types import (
    AuthStatus,
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    Delta,
    Message,
    StreamChoice,
    Usage,
)

_MODEL_MAP: dict[str, str] = {
    "gpt-5.2": "gpt-5.2",
    "gpt-5.2-codex": "gpt-5.2-codex",
    "gpt-4.1": "gpt-4.1",
    "gpt-5-mini": "gpt-5-mini",
}


class CodexProvider(Provider):
    """Routes completions through the OpenAI Codex CLI."""

    def __init__(self, cli_path: str | None = None):
        self._cli_path = cli_path or shutil.which("codex") or "codex"

    @property
    def name(self) -> str:
        return "codex"

    @property
    def supported_models(self) -> list[str]:
        return list(_MODEL_MAP.keys())

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_sessions=True,
            supports_system_prompt=True,
            supports_vision=False,
            max_context_tokens=200_000,
            subscription_auth=True,
            api_key_auth=True,
        )

    def resolve_model(self, model_alias: str) -> str:
        return _MODEL_MAP.get(model_alias, model_alias)

    async def check_auth(self) -> AuthStatus:
        if os.environ.get("OPENAI_API_KEY"):
            return AuthStatus(provider=self.name, authenticated=True, method="api_key")
        if not shutil.which(self._cli_path):
            return AuthStatus(
                provider=self.name, authenticated=False,
                error=f"Codex CLI not found at '{self._cli_path}'. "
                "Install: npm install -g @openai/codex",
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path, "login", "status",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                return AuthStatus(provider=self.name, authenticated=True, method="subscription")
            return AuthStatus(
                provider=self.name, authenticated=False,
                error="Not authenticated. Run `codex login` to sign in.",
            )
        except (asyncio.TimeoutError, FileNotFoundError):
            return AuthStatus(
                provider=self.name, authenticated=False,
                error="Could not verify Codex authentication.",
            )

    async def complete(
        self, messages: list[dict], model: str, *,
        system_prompt: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse:
        prompt = messages_to_prompt(messages, system_prompt)
        resolved = self.resolve_model(model)
        args = [self._cli_path, "exec", prompt, "--model", resolved, "--full-auto", "--json"]

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Codex CLI failed (exit {proc.returncode}): {stderr.decode().strip()}"
            )

        content_parts: list[str] = []
        usage = None
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        content_parts.append(item.get("text", ""))
                elif event.get("type") == "turn.completed":
                    stats = event.get("usage", {})
                    inp = stats.get("input_tokens", 0)
                    out = stats.get("output_tokens", 0)
                    usage = Usage(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)
            except json.JSONDecodeError:
                continue

        content = "\n".join(content_parts)
        if not usage:
            usage = Usage(
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(content),
                total_tokens=estimate_tokens(prompt) + estimate_tokens(content),
            )

        return ChatCompletionResponse(
            model=f"codex/{model}",
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def stream(
        self, messages: list[dict], model: str, *,
        system_prompt: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = messages_to_prompt(messages, system_prompt)
        resolved = self.resolve_model(model)
        args = [self._cli_path, "exec", prompt, "--model", resolved, "--full-auto", "--json"]

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )

        chunk_id = f"chatcmpl-codex-{id(proc)}"
        first = True

        assert proc.stdout is not None
        async for line_bytes in proc.stdout:
            line = line_bytes.decode().strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        text = item.get("text", "")
                        if text:
                            yield ChatCompletionChunk(
                                id=chunk_id, model=f"codex/{model}",
                                choices=[StreamChoice(delta=Delta(
                                    role="assistant" if first else None, content=text,
                                ))],
                            )
                            first = False
                elif event.get("type") == "error":
                    raise RuntimeError(f"Codex error: {event.get('message', 'unknown')}")
            except json.JSONDecodeError:
                continue

        yield ChatCompletionChunk(
            id=chunk_id, model=f"codex/{model}",
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )
        await proc.wait()
