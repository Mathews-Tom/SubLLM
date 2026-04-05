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
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal

from claude_agent_sdk import ClaudeAgentOptions, ClaudeSDKClient
from claude_agent_sdk.types import (
    AssistantMessage,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ThinkingConfigAdaptive,
    ThinkingConfigDisabled,
    ThinkingConfigEnabled,
    UserMessage,
)

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
    ResponseSession,
    SessionRequest,
    StreamChoice,
    Usage,
)

logger = logging.getLogger(__name__)

# SDK effort levels exposed through the provider interface
EffortLevel = Literal["low", "medium", "high", "max"]

# SDK thinking modes
ThinkingMode = Literal["adaptive", "enabled", "disabled"]
ClaudeResponseEvent = AssistantMessage | ResultMessage | UserMessage | SystemMessage | StreamEvent


@dataclass(frozen=True)
class ClaudeClientKey:
    model: str
    system_prompt: str
    thinking: ThinkingMode
    thinking_budget: int | None
    effort: EffortLevel | None


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
        request_timeout: float = 60.0,
        stream_idle_timeout: float = 30.0,
    ):
        self._cli_path = cli_path or shutil.which("claude") or "claude"
        self._env_overrides = env_overrides or {}
        self._thinking = thinking
        self._thinking_budget = thinking_budget
        self._effort = effort
        self._request_timeout = request_timeout
        self._stream_idle_timeout = stream_idle_timeout
        self._sdk_clients: dict[ClaudeClientKey, ClaudeSDKClient] = {}
        self._client_locks: dict[ClaudeClientKey, asyncio.Lock] = {}
        self._client_registry_lock = asyncio.Lock()

    @property
    def name(self) -> str:
        return "claude-code"

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
                self._cli_path,
                "auth",
                "status",
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
                self._cli_path,
                "--print",
                "-p",
                "say ok",
                "--max-turns",
                "1",
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
                provider=self.name,
                authenticated=False,
                error="Claude Code CLI not found.",
            )

    # ── SDK inference ────────────────────────────────────────────────

    def _build_sdk_options(
        self,
        model: str,
        *,
        system_prompt: str | None = None,
    ) -> ClaudeAgentOptions:
        """Build ClaudeAgentOptions for the SDK client."""
        resolved = self.resolve_model(model)

        # Build thinking config based on provider settings
        thinking_config: (
            ThinkingConfigAdaptive | ThinkingConfigEnabled | ThinkingConfigDisabled | None
        ) = None
        if self._thinking == "adaptive":
            thinking_config = ThinkingConfigAdaptive(type="adaptive")
        elif self._thinking == "enabled":
            thinking_config = ThinkingConfigEnabled(
                type="enabled",
                budget_tokens=self._thinking_budget or 10_000,
            )
        elif self._thinking == "disabled":
            thinking_config = ThinkingConfigDisabled(type="disabled")

        # Merge env overrides with CLAUDECODE unset to prevent nested-session errors.
        # The SDK's env param is additive to os.environ, so we must explicitly blank it.
        sdk_env = {"CLAUDECODE": "", **self._env_overrides}
        if os.environ.get("SUBLLM_FORCE_SUBSCRIPTION"):
            sdk_env["ANTHROPIC_API_KEY"] = ""

        return ClaudeAgentOptions(
            model=resolved,
            max_turns=1,
            system_prompt=system_prompt or "",
            permission_mode="default",
            thinking=thinking_config,
            effort=self._effort,
            cli_path=self._cli_path,
            env=sdk_env,
        )

    def _client_key(self, model: str, system_prompt: str | None = None) -> ClaudeClientKey:
        return ClaudeClientKey(
            model=self.resolve_model(model),
            system_prompt=system_prompt or "",
            thinking=self._thinking,
            thinking_budget=self._thinking_budget,
            effort=self._effort,
        )

    async def _create_sdk_client(self, options: ClaudeAgentOptions) -> ClaudeSDKClient:
        client = ClaudeSDKClient(options)
        await client.connect()
        return client

    async def _client_lock(self, key: ClaudeClientKey) -> asyncio.Lock:
        async with self._client_registry_lock:
            return self._client_locks.setdefault(key, asyncio.Lock())

    async def _ensure_sdk_client(
        self, model: str, *, system_prompt: str | None = None
    ) -> ClaudeSDKClient:
        """Return the keyed SDK client, connecting on first use.

        Clients are isolated by immutable request config. Calls sharing the same
        config reuse a warm subprocess; different configs get independent clients.
        """
        key = self._client_key(model, system_prompt=system_prompt)
        existing_client = self._sdk_clients.get(key)
        if existing_client is not None:
            return existing_client

        async with self._client_registry_lock:
            existing_client = self._sdk_clients.get(key)
            if existing_client is not None:
                return existing_client

            options = self._build_sdk_options(model, system_prompt=system_prompt)
            client = await self._create_sdk_client(options)
            self._sdk_clients[key] = client
            self._client_locks.setdefault(key, asyncio.Lock())
            return client

    async def complete(
        self,
        messages: list[ProviderMessage],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session: SessionRequest | None = None,
    ) -> ChatCompletionResponse:
        return await self._complete(messages, model, system_prompt=system_prompt, session=session)

    async def _query_with_timeout(
        self,
        client: ClaudeSDKClient,
        prompt: str,
        *,
        session_id: str,
    ) -> None:
        try:
            await asyncio.wait_for(
                client.query(prompt, session_id=session_id), timeout=self._request_timeout
            )
        except asyncio.TimeoutError as exc:
            raise ProviderTimeoutError(
                provider=self.name,
                operation="completion",
                timeout_seconds=self._request_timeout,
            ) from exc

    async def _next_response(
        self,
        response_stream: AsyncIterator[ClaudeResponseEvent],
    ) -> ClaudeResponseEvent | None:
        try:
            return await asyncio.wait_for(anext(response_stream), timeout=self._stream_idle_timeout)
        except StopAsyncIteration:
            return None
        except asyncio.TimeoutError as exc:
            raise ProviderTimeoutError(
                provider=self.name,
                operation="stream",
                timeout_seconds=self._stream_idle_timeout,
            ) from exc

    async def _complete(
        self,
        messages: list[ProviderMessage],
        model: str,
        *,
        system_prompt: str | None = None,
        session: SessionRequest | None = None,
    ) -> ChatCompletionResponse:
        prompt = messages_to_prompt(messages, system_prompt)
        session_id = _query_session_id(session)

        try:
            client_key = self._client_key(model, system_prompt=system_prompt)
            client_lock = await self._client_lock(client_key)
            async with client_lock:
                client = await self._ensure_sdk_client(model, system_prompt=system_prompt)
                await self._query_with_timeout(client, prompt, session_id=session_id)

                content_parts: list[str] = []
                usage_data: dict[str, Any] = {}
                response_session_id: str | None = None
                response_stream = client.receive_response()

                while (message := await self._next_response(response_stream)) is not None:
                    if isinstance(message, AssistantMessage):
                        for block in message.content:
                            if isinstance(block, TextBlock):
                                content_parts.append(block.text)
                    elif isinstance(message, ResultMessage):
                        response_session_id = getattr(message, "session_id", response_session_id)
                        if message.usage:
                            usage_data = message.usage
                        if message.is_error:
                            raise ProviderFailureError(
                                provider=self.name,
                                message=f"SDK query failed: {message.result or 'unknown error'}",
                            )

                content = "".join(content_parts)

                # Use real usage data from ResultMessage if available, else estimate
                prompt_tokens = usage_data.get("input_tokens", estimate_tokens(prompt))
                completion_tokens = usage_data.get("output_tokens", estimate_tokens(content))

                return ChatCompletionResponse(
                    model=f"claude-code/{model}",
                    choices=[
                        Choice(
                            message=Message(role="assistant", content=content),
                            finish_reason="stop",
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                    session=_response_session(session, response_session_id or session_id),
                )

        except (ProviderFailureError, ProviderTimeoutError):
            raise
        except Exception as exc:
            raise ProviderFailureError(provider=self.name, message=f"SDK error: {exc}") from exc

    async def stream(
        self,
        messages: list[ProviderMessage],
        model: str,
        *,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        session: SessionRequest | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in self._stream_impl(
            messages,
            model,
            system_prompt=system_prompt,
            session=session,
        ):
            yield chunk

    async def _stream_impl(
        self,
        messages: list[ProviderMessage],
        model: str,
        *,
        system_prompt: str | None = None,
        session: SessionRequest | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = messages_to_prompt(messages, system_prompt)
        session_id = _query_session_id(session)
        session_info = _response_session(session, session_id)

        try:
            client_key = self._client_key(model, system_prompt=system_prompt)
            client_lock = await self._client_lock(client_key)
        except Exception as exc:
            if isinstance(exc, (ProviderFailureError, ProviderTimeoutError)):
                raise
            raise ProviderFailureError(
                provider=self.name, message=f"SDK connection error: {exc}"
            ) from exc

        async with client_lock:
            try:
                client = await self._ensure_sdk_client(model, system_prompt=system_prompt)
                await self._query_with_timeout(client, prompt, session_id=session_id)
            except Exception as exc:
                if isinstance(exc, (ProviderFailureError, ProviderTimeoutError)):
                    raise
                raise ProviderFailureError(
                    provider=self.name, message=f"SDK connection error: {exc}"
                ) from exc

            chunk_id = f"chatcmpl-sdk-{id(client)}"
            first = True

            response_stream = client.receive_response()
            while (message := await self._next_response(response_stream)) is not None:
                if isinstance(message, AssistantMessage):
                    for block in message.content:
                        if isinstance(block, TextBlock):
                            yield ChatCompletionChunk(
                                id=chunk_id,
                                model=f"claude-code/{model}",
                                session=session_info,
                                choices=[
                                    StreamChoice(
                                        delta=Delta(
                                            role="assistant" if first else None,
                                            content=block.text,
                                        )
                                    )
                                ],
                            )
                            first = False
                elif isinstance(message, ResultMessage):
                    result_session_id = getattr(message, "session_id", session_id)
                    session_info = _response_session(session, result_session_id)
                    if message.is_error:
                        raise ProviderFailureError(
                            provider=self.name,
                            message=f"SDK stream failed: {message.result or 'unknown error'}",
                        )

            yield ChatCompletionChunk(
                id=chunk_id,
                model=f"claude-code/{model}",
                session=session_info,
                choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
            )

    async def close(self) -> None:
        """Disconnect all keyed SDK clients."""
        clients = list(self._sdk_clients.values())
        self._sdk_clients.clear()
        self._client_locks.clear()
        for client in clients:
            try:
                await client.disconnect()
            except Exception:
                logger.debug("SDK client disconnect error (ignored)", exc_info=True)


def _query_session_id(session: SessionRequest | None) -> str:
    if session is not None and session.mode == "resume" and session.id is not None:
        return session.id
    return str(uuid.uuid4())


def _response_session(
    session: SessionRequest | None,
    session_id: str,
) -> ResponseSession | None:
    if session is None:
        return None
    return ResponseSession(id=session_id, mode=session.mode)
