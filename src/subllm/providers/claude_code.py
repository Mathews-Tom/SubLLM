"""Claude Code provider — routes LLM calls through the Claude Code CLI.

Two execution modes:
1. CLI subprocess (`claude --print`) — zero Python deps, works with any auth
2. Agent SDK (`claude-agent-sdk`) — richer streaming, typed errors, warm process

Auth priority (handled by CLI):
  ANTHROPIC_API_KEY > CLAUDE_CODE_OAUTH_TOKEN > keychain subscription OAuth

To use subscription auth, ensure ANTHROPIC_API_KEY is NOT set and run `claude login`.
For headless/CI: `claude setup-token` generates a long-lived CLAUDE_CODE_OAUTH_TOKEN.
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
    "opus-4-6": "claude-opus-4-6",
    "sonnet-4-5": "claude-sonnet-4-5",
    "haiku-4-5": "claude-haiku-4-5",
}


class ClaudeCodeProvider(Provider):
    """Routes completions through the Claude Code CLI."""

    def __init__(
        self,
        cli_path: str | None = None,
        use_sdk: bool = False,
        env_overrides: dict[str, str] | None = None,
    ):
        self._cli_path = cli_path or shutil.which("claude") or "claude"
        self._use_sdk = use_sdk
        self._env_overrides = env_overrides or {}

    @property
    def name(self) -> str:
        return "claude-code"

    @property
    def supported_models(self) -> list[str]:
        return list(_MODEL_MAP.keys())

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_sessions=True,
            supports_system_prompt=True,
            supports_vision=True,
            max_context_tokens=200_000,
            subscription_auth=True,
            api_key_auth=True,
        )

    def resolve_model(self, model_alias: str) -> str:
        return _MODEL_MAP.get(model_alias, model_alias)

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.update(self._env_overrides)
        if env.get("SUBLLM_FORCE_SUBSCRIPTION"):
            env.pop("ANTHROPIC_API_KEY", None)
        return env

    async def check_auth(self) -> AuthStatus:
        if os.environ.get("ANTHROPIC_API_KEY"):
            return AuthStatus(provider=self.name, authenticated=True, method="api_key")
        if os.environ.get("CLAUDE_CODE_OAUTH_TOKEN"):
            return AuthStatus(provider=self.name, authenticated=True, method="oauth_token")

        if not shutil.which(self._cli_path):
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error=f"Claude Code CLI not found at '{self._cli_path}'. "
                "Install: curl -fsSL https://claude.ai/install.sh | bash",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path, "--print", "-p", "say ok", "--max-turns", "1",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                return AuthStatus(provider=self.name, authenticated=True, method="subscription")
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error=f"Auth check failed: {stderr.decode().strip()}",
            )
        except asyncio.TimeoutError:
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error="Auth check timed out. Run `claude login` to authenticate.",
            )
        except FileNotFoundError:
            return AuthStatus(
                provider=self.name, authenticated=False, error="Claude Code CLI not found.",
            )

    async def complete(
        self,
        messages: list[dict],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse:
        if self._use_sdk:
            return await self._complete_sdk(messages, model, system_prompt=system_prompt)
        return await self._complete_cli(
            messages, model, system_prompt=system_prompt, max_tokens=max_tokens
        )

    async def stream(
        self,
        messages: list[dict],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        if self._use_sdk:
            async for chunk in self._stream_sdk(messages, model, system_prompt=system_prompt):
                yield chunk
        else:
            async for chunk in self._stream_cli(
                messages, model, system_prompt=system_prompt, max_tokens=max_tokens
            ):
                yield chunk

    # ── CLI subprocess ─────────────────────────────────────────────

    def _build_cli_args(
        self, prompt: str, model: str, *, max_tokens: int | None = None, output_format: str = "text",
    ) -> list[str]:
        resolved = self.resolve_model(model)
        args = [
            self._cli_path, "--print", "-p", prompt,
            "--model", resolved, "--max-turns", "1", "--output-format", output_format,
        ]
        if output_format == "stream-json":
            args.append("--verbose")
        if max_tokens:
            args.extend(["--max-tokens", str(max_tokens)])
        return args

    async def _complete_cli(
        self, messages: list[dict], model: str, *,
        system_prompt: str | None = None, max_tokens: int | None = None,
    ) -> ChatCompletionResponse:
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model, max_tokens=max_tokens)

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Claude Code CLI failed (exit {proc.returncode}): {stderr.decode().strip()}"
            )

        content = stdout.decode().strip()
        return ChatCompletionResponse(
            model=f"claude-code/{model}",
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(content),
                total_tokens=estimate_tokens(prompt) + estimate_tokens(content),
            ),
        )

    async def _stream_cli(
        self, messages: list[dict], model: str, *,
        system_prompt: str | None = None, max_tokens: int | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model, max_tokens=max_tokens, output_format="stream-json")

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
        )

        chunk_id = f"chatcmpl-cc-{id(proc)}"
        first = True

        assert proc.stdout is not None
        async for line_bytes in proc.stdout:
            line = line_bytes.decode().strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                if event.get("type") == "assistant" and "message" in event:
                    for block in event["message"].get("content", []):
                        if block.get("type") == "text" and block.get("text"):
                            yield ChatCompletionChunk(
                                id=chunk_id, model=f"claude-code/{model}",
                                choices=[StreamChoice(delta=Delta(
                                    role="assistant" if first else None, content=block["text"],
                                ))],
                            )
                            first = False
                elif event.get("type") == "result":
                    continue  # Summary event — content already yielded via assistant message
            except json.JSONDecodeError:
                if line:
                    yield ChatCompletionChunk(
                        id=chunk_id, model=f"claude-code/{model}",
                        choices=[StreamChoice(delta=Delta(
                            role="assistant" if first else None, content=line + "\n",
                        ))],
                    )
                    first = False

        yield ChatCompletionChunk(
            id=chunk_id, model=f"claude-code/{model}",
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )
        await proc.wait()

    # ── Agent SDK ──────────────────────────────────────────────────

    async def _complete_sdk(
        self, messages: list[dict], model: str, *, system_prompt: str | None = None,
    ) -> ChatCompletionResponse:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except ImportError:
            raise ImportError("claude-agent-sdk not installed. Run: pip install subllm[sdk]")

        prompt = messages_to_prompt(messages, system_prompt)
        resolved = self.resolve_model(model)
        options = ClaudeAgentOptions(
            model=resolved, max_turns=1, system_prompt=system_prompt or "",
        )

        content_parts: list[str] = []
        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        content_parts.append(block.text)

        content = "".join(content_parts)
        return ChatCompletionResponse(
            model=f"claude-code/{model}",
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=Usage(
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(content),
                total_tokens=estimate_tokens(prompt) + estimate_tokens(content),
            ),
        )

    async def _stream_sdk(
        self, messages: list[dict], model: str, *, system_prompt: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        try:
            from claude_agent_sdk import ClaudeAgentOptions, query
        except ImportError:
            raise ImportError("claude-agent-sdk not installed. Run: pip install subllm[sdk]")

        prompt = messages_to_prompt(messages, system_prompt)
        resolved = self.resolve_model(model)
        options = ClaudeAgentOptions(
            model=resolved, max_turns=1, system_prompt=system_prompt or "",
        )

        chunk_id = f"chatcmpl-sdk-{id(options)}"
        first = True

        async for message in query(prompt=prompt, options=options):
            if hasattr(message, "content"):
                for block in message.content:
                    if hasattr(block, "text"):
                        yield ChatCompletionChunk(
                            id=chunk_id, model=f"claude-code/{model}",
                            choices=[StreamChoice(delta=Delta(
                                role="assistant" if first else None, content=block.text,
                            ))],
                        )
                        first = False

        yield ChatCompletionChunk(
            id=chunk_id, model=f"claude-code/{model}",
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )
