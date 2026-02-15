"""Claude Code provider — routes LLM calls through the Agent SDK.

Uses a persistent SDK client (`claude-agent-sdk`) for all inference. The client
maintains a warm subprocess — subsequent calls skip spawn overhead entirely.

Auth checking still uses lightweight CLI subprocess calls (`claude auth status`)
because the SDK doesn't expose an auth-check API.

Auth priority (handled by CLI/SDK):
  ANTHROPIC_API_KEY > CLAUDE_CODE_OAUTH_TOKEN > keychain subscription OAuth

To use subscription auth, ensure ANTHROPIC_API_KEY is NOT set and run `claude login`.
For headless/CI: `claude setup-token` generates a long-lived CLAUDE_CODE_OAUTH_TOKEN.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
from collections.abc import AsyncIterator
from typing import Any, Literal

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import AssistantMessage, ResultMessage, TextBlock

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

logger = logging.getLogger(__name__)

_MODEL_MAP: dict[str, str] = {
    "opus-4-6": "claude-opus-4-6",
    "sonnet-4-5": "claude-sonnet-4-5",
    "haiku-4-5": "claude-haiku-4-5",
}

# SDK effort levels exposed through the provider interface
EffortLevel = Literal["low", "medium", "high", "max"]

# SDK thinking modes
ThinkingMode = Literal["adaptive", "enabled", "disabled"]


class ClaudeCodeProvider(Provider):
    """Routes completions through the Claude Code Agent SDK.

    Args:
        cli_path: Path to the `claude` CLI binary. Used for auth checks. Auto-detected if None.
        env_overrides: Extra environment variables passed to the SDK process.
        thinking: Thinking mode. "adaptive" (default), "enabled", or "disabled".
        thinking_budget: Token budget when thinking="enabled". Ignored for "adaptive"/"disabled".
        effort: Effort level. Controls thinking depth with adaptive mode.
    """

    def __init__(
        self,
        cli_path: str | None = None,
        env_overrides: dict[str, str] | None = None,
        thinking: ThinkingMode = "adaptive",
        thinking_budget: int | None = None,
        effort: EffortLevel | None = None,
    ):
        self._cli_path = cli_path or shutil.which("claude") or "claude"
        self._env_overrides = env_overrides or {}
        self._thinking = thinking
        self._thinking_budget = thinking_budget
        self._effort = effort
        # Persistent SDK client — connected on first use, reused across calls
        self._sdk_client: ClaudeSDKClient | None = None

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

        # Fast path: `claude auth status` (~275ms vs ~13s for full inference)
        result = await self._check_auth_fast()
        if result is not None:
            return result

        # Slow fallback: full inference roundtrip (older CLI without `auth status`)
        return await self._check_auth_slow()

    async def _check_auth_fast(self) -> AuthStatus | None:
        """Try `claude auth status` for lightweight auth verification.

        Returns None if the command is unavailable (older CLI), triggering slow fallback.
        """
        try:
            env = self._build_env()
            env["CLAUDECODE"] = ""  # Prevent error inside active Claude Code sessions
            proc = await asyncio.create_subprocess_exec(
                self._cli_path, "auth", "status",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=5)

            stderr_text = stderr.decode().strip()
            if proc.returncode != 0 or "Unknown command" in stderr_text:
                return None  # Command not available, use slow fallback

            data = json.loads(stdout.decode())
            if data.get("loggedIn"):
                method = data.get("subscriptionType") or data.get("authMethod") or "subscription"
                return AuthStatus(provider=self.name, authenticated=True, method=method)
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error="Not logged in. Run `claude login` to authenticate.",
            )
        except (asyncio.TimeoutError, json.JSONDecodeError, FileNotFoundError, OSError):
            return None  # Any failure → fall back to slow check

    async def _check_auth_slow(self) -> AuthStatus:
        """Fallback: run a minimal inference call to verify auth."""
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

    # ── SDK inference ────────────────────────────────────────────────

    def _build_sdk_options(
        self, model: str, *, system_prompt: str | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for the SDK client."""
        resolved = self.resolve_model(model)

        # Build thinking config based on provider settings
        thinking_config: dict[str, Any] | None = None
        if self._thinking == "adaptive":
            thinking_config = {"type": "adaptive"}
        elif self._thinking == "enabled":
            thinking_config = {"type": "enabled", "budget_tokens": self._thinking_budget or 10_000}
        elif self._thinking == "disabled":
            thinking_config = {"type": "disabled"}

        # Merge env overrides with CLAUDECODE unset to prevent nested-session errors.
        # The SDK's env param is additive to os.environ, so we must explicitly blank it.
        sdk_env = {"CLAUDECODE": "", **self._env_overrides}
        if os.environ.get("SUBLLM_FORCE_SUBSCRIPTION"):
            sdk_env["ANTHROPIC_API_KEY"] = ""

        return ClaudeAgentOptions(
            model=resolved,
            max_turns=1,
            system_prompt=system_prompt or "",
            permission_mode="bypassPermissions",
            thinking=thinking_config,
            effort=self._effort,
            cli_path=self._cli_path,
            env=sdk_env,
        )

    async def _ensure_sdk_client(self, model: str, *, system_prompt: str | None = None) -> ClaudeSDKClient:
        """Return the persistent SDK client, connecting on first use.

        The client maintains a warm subprocess — subsequent calls skip spawn overhead.
        """
        if self._sdk_client is not None:
            return self._sdk_client

        options = self._build_sdk_options(model, system_prompt=system_prompt)
        client = ClaudeSDKClient(options)
        await client.connect()
        self._sdk_client = client
        return client

    async def complete(
        self,
        messages: list[dict],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse:
        return await self._complete(messages, model, system_prompt=system_prompt)

    async def _complete(
        self, messages: list[dict], model: str, *, system_prompt: str | None = None,
    ) -> ChatCompletionResponse:
        prompt = messages_to_prompt(messages, system_prompt)

        try:
            client = await self._ensure_sdk_client(model, system_prompt=system_prompt)
            await client.query(prompt)

            content_parts: list[str] = []
            usage_data: dict[str, Any] = {}

            async for message in client.receive_response():
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            content_parts.append(block.text)
                elif isinstance(message, ResultMessage):
                    if message.usage:
                        usage_data = message.usage
                    if message.is_error:
                        raise RuntimeError(
                            f"SDK query failed: {message.result or 'unknown error'}"
                        )

            content = "".join(content_parts)

            # Use real usage data from ResultMessage if available, else estimate
            prompt_tokens = usage_data.get("input_tokens", estimate_tokens(prompt))
            completion_tokens = usage_data.get("output_tokens", estimate_tokens(content))

            return ChatCompletionResponse(
                model=f"claude-code/{model}",
                choices=[Choice(
                    message=Message(role="assistant", content=content),
                    finish_reason="stop",
                )],
                usage=Usage(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=prompt_tokens + completion_tokens,
                ),
            )

        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"SDK error: {exc}") from exc

    async def stream(
        self,
        messages: list[dict],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in self._stream_impl(messages, model, system_prompt=system_prompt):
            yield chunk

    async def _stream_impl(
        self, messages: list[dict], model: str, *, system_prompt: str | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = messages_to_prompt(messages, system_prompt)

        try:
            client = await self._ensure_sdk_client(model, system_prompt=system_prompt)
            await client.query(prompt)
        except Exception as exc:
            raise RuntimeError(f"SDK connection error: {exc}") from exc

        chunk_id = f"chatcmpl-sdk-{id(client)}"
        first = True

        async for message in client.receive_response():
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        yield ChatCompletionChunk(
                            id=chunk_id, model=f"claude-code/{model}",
                            choices=[StreamChoice(delta=Delta(
                                role="assistant" if first else None, content=block.text,
                            ))],
                        )
                        first = False
            elif isinstance(message, ResultMessage):
                if message.is_error:
                    logger.error("SDK stream error: %s", message.result)

        yield ChatCompletionChunk(
            id=chunk_id, model=f"claude-code/{model}",
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )

    async def close(self) -> None:
        """Disconnect the persistent SDK client if connected."""
        if self._sdk_client is not None:
            try:
                await self._sdk_client.disconnect()
            except Exception:
                logger.debug("SDK client disconnect error (ignored)", exc_info=True)
            finally:
                self._sdk_client = None
