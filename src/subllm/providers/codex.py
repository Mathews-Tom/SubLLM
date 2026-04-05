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

from subllm.errors import ProviderFailureError, ProviderTimeoutError
from subllm.model_registry import (
    provider_capabilities,
    provider_model_aliases,
    resolve_provider_model,
)
from subllm.providers.base import (
    Provider,
    ProviderCapabilities,
    estimate_tokens,
    messages_to_prompt,
)
from subllm.types import (
    AuthStatus,
    ChatCompletionChunk,
    ChatCompletionResponse,
    Choice,
    Delta,
    Message,
    ProviderMessage,
    StreamChoice,
    Usage,
)


class CodexProvider(Provider):
    """Routes completions through the OpenAI Codex CLI."""

    def __init__(
        self,
        cli_path: str | None = None,
        command_timeout: float = 60.0,
        stream_idle_timeout: float = 30.0,
    ):
        self._cli_path = cli_path or shutil.which("codex") or "codex"
        self._command_timeout = command_timeout
        self._stream_idle_timeout = stream_idle_timeout

    @property
    def name(self) -> str:
        return "codex"

    @property
    def supported_models(self) -> list[str]:
        return provider_model_aliases(self.name)

    @property
    def capabilities(self) -> ProviderCapabilities:
        capabilities = provider_capabilities(self.name)
        if capabilities is None:
            raise RuntimeError(f"Missing model registry entry for provider '{self.name}'")
        return capabilities

    def resolve_model(self, model_alias: str) -> str:
        resolved = resolve_provider_model(self.name, model_alias)
        return resolved or model_alias

    def _build_cli_args(self, prompt: str, model: str) -> list[str]:
        resolved = self.resolve_model(model)
        return [self._cli_path, "exec", prompt, "--model", resolved, "--json"]

    async def _terminate_process(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        proc.kill()
        try:
            await proc.wait()
        except Exception:
            return

    async def _read_stderr(self, proc: asyncio.subprocess.Process) -> str:
        if proc.stderr is None:
            return ""
        return (await proc.stderr.read()).decode().strip()

    async def check_auth(self) -> AuthStatus:
        if os.environ.get("OPENAI_API_KEY"):
            return AuthStatus(provider=self.name, authenticated=True, method="api_key")
        if not shutil.which(self._cli_path):
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error=f"Codex CLI not found at '{self._cli_path}'. "
                "Install: npm install -g @openai/codex",
            )
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path,
                "login",
                "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
            if proc.returncode == 0:
                return AuthStatus(provider=self.name, authenticated=True, method="subscription")
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error="Not authenticated. Run `codex login` to sign in.",
            )
        except (asyncio.TimeoutError, FileNotFoundError):
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error="Could not verify Codex authentication.",
            )

    async def complete(
        self,
        messages: list[ProviderMessage],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse:
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model)

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=self._command_timeout,
            )
        except asyncio.TimeoutError as exc:
            await self._terminate_process(proc)
            raise ProviderTimeoutError(
                provider=self.name,
                operation="completion",
                timeout_seconds=self._command_timeout,
            ) from exc

        if proc.returncode != 0:
            raise ProviderFailureError(
                provider=self.name,
                message=f"Codex CLI failed (exit {proc.returncode}): {stderr.decode().strip()}",
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
            choices=[
                Choice(message=Message(role="assistant", content=content), finish_reason="stop")
            ],
            usage=usage,
        )

    async def stream(
        self,
        messages: list[ProviderMessage],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model)

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        chunk_id = f"chatcmpl-codex-{id(proc)}"
        first = True
        stream_error: str | None = None

        assert proc.stdout is not None
        while True:
            try:
                line_bytes = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=self._stream_idle_timeout,
                )
            except asyncio.TimeoutError as exc:
                await self._terminate_process(proc)
                raise ProviderTimeoutError(
                    provider=self.name,
                    operation="stream",
                    timeout_seconds=self._stream_idle_timeout,
                ) from exc

            if not line_bytes:
                break
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
                                id=chunk_id,
                                model=f"codex/{model}",
                                choices=[
                                    StreamChoice(
                                        delta=Delta(
                                            role="assistant" if first else None,
                                            content=text,
                                        )
                                    )
                                ],
                            )
                            first = False
                elif event.get("type") == "error":
                    stream_error = f"Codex error: {event.get('message', 'unknown')}"
                    break
            except json.JSONDecodeError:
                continue

        try:
            await asyncio.wait_for(proc.wait(), timeout=self._command_timeout)
        except asyncio.TimeoutError as exc:
            await self._terminate_process(proc)
            raise ProviderTimeoutError(
                provider=self.name,
                operation="stream shutdown",
                timeout_seconds=self._command_timeout,
            ) from exc

        stderr_text = await self._read_stderr(proc)
        if stream_error is not None:
            details = f"{stream_error}. {stderr_text}".strip().rstrip(".")
            raise ProviderFailureError(provider=self.name, message=details)
        if proc.returncode != 0:
            raise ProviderFailureError(
                provider=self.name,
                message=f"Codex CLI failed (exit {proc.returncode}): {stderr_text}",
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            model=f"codex/{model}",
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )
