"""Gemini CLI provider — routes LLM calls through Google's Gemini CLI.

Uses `gemini -p "..."` for non-interactive headless execution.
Auth: Google OAuth (cached from `gemini` login), GEMINI_API_KEY, or GOOGLE_API_KEY.

Key notes:
- Free tier: 60 req/min, 1000 req/day with personal Google account
- Google AI Pro/Ultra subscribers get higher limits and model access
- Structured output via --output-format json (includes token stats)
- Streaming via --output-format stream-json (JSONL events)
- Session resume NOT supported in headless (-p) mode — stateless only
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections.abc import AsyncIterator
from pathlib import Path

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
    SessionRequest,
    StreamChoice,
    Usage,
)


class GeminiCLIProvider(Provider):
    """Routes completions through the Google Gemini CLI."""

    def __init__(
        self,
        cli_path: str | None = None,
        env_overrides: dict[str, str] | None = None,
        command_timeout: float = 60.0,
        stream_idle_timeout: float = 30.0,
    ):
        self._cli_path = cli_path or shutil.which("gemini") or "gemini"
        self._env_overrides = env_overrides or {}
        self._command_timeout = command_timeout
        self._stream_idle_timeout = stream_idle_timeout

    @property
    def name(self) -> str:
        return "gemini"

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
        env.update(self._env_overrides)
        return env

    async def check_auth(self) -> AuthStatus:
        if os.environ.get("GEMINI_API_KEY"):
            return AuthStatus(provider=self.name, authenticated=True, method="api_key")
        if os.environ.get("GOOGLE_API_KEY"):
            return AuthStatus(provider=self.name, authenticated=True, method="google_api_key")

        # Fast path: check OAuth credential file on disk (<10ms vs ~16s for inference)
        result = self._check_auth_from_file()
        if result is not None:
            return result

        if not shutil.which(self._cli_path):
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error=f"Gemini CLI not found at '{self._cli_path}'. "
                "Install: npm install -g @google/gemini-cli or see https://geminicli.com",
            )

        # Slow fallback: full inference roundtrip (no credential file found)
        return await self._check_auth_slow()

    def _check_auth_from_file(self) -> AuthStatus | None:
        """Check Gemini OAuth credentials on disk for fast auth verification.

        Returns None if credential file is missing or unparseable, triggering slow fallback.
        The presence of a refresh_token means the CLI will auto-refresh on next use,
        so we treat expired access tokens with valid refresh tokens as authenticated.
        """
        oauth_path = Path.home() / ".gemini" / "oauth_creds.json"
        try:
            data = json.loads(oauth_path.read_text())
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None  # File missing or corrupt → fall back to slow check

        access_token = data.get("access_token")
        refresh_token = data.get("refresh_token")
        if not access_token and not refresh_token:
            return None  # No usable credentials → fall back

        # If refresh_token exists, the CLI will auto-refresh — consider authenticated
        if refresh_token:
            return AuthStatus(provider=self.name, authenticated=True, method="google_oauth")

        # Access token only (no refresh) — check expiry
        expiry = data.get("expiry_date")
        if expiry is not None:
            # Normalize: >10^12 = milliseconds, otherwise seconds
            expiry_sec = expiry / 1000 if expiry > 1e12 else expiry
            if expiry_sec > time.time():
                return AuthStatus(provider=self.name, authenticated=True, method="google_oauth")
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error="Google OAuth token expired. Run `gemini` to re-authenticate.",
            )

        # Has access_token but no expiry field — assume valid
        return AuthStatus(provider=self.name, authenticated=True, method="google_oauth")

    async def _check_auth_slow(self) -> AuthStatus:
        """Fallback: run a minimal inference call to verify auth."""
        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path,
                "-p",
                "say ok",
                "--output-format",
                "json",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                return AuthStatus(provider=self.name, authenticated=True, method="google_oauth")
            err_msg = stderr.decode().strip()
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error=f"Not authenticated: {err_msg}. "
                "Run `gemini` and complete Google login, or set GEMINI_API_KEY.",
            )
        except asyncio.TimeoutError:
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error="Auth check timed out.",
            )
        except FileNotFoundError:
            return AuthStatus(
                provider=self.name,
                authenticated=False,
                error="Gemini CLI not found.",
            )

    def _build_cli_args(
        self,
        prompt: str,
        model: str,
        *,
        output_format: str = "json",
    ) -> list[str]:
        resolved = self.resolve_model(model)
        return [self._cli_path, "-p", prompt, "--model", resolved, "--output-format", output_format]

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
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model, output_format="json")

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
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
                message=f"Gemini CLI failed (exit {proc.returncode}): {stderr.decode().strip()}",
            )

        raw = stdout.decode().strip()
        content = raw
        usage = None

        # Gemini CLI JSON: {"response": "...", "stats": {...}, "error": null}
        try:
            data = json.loads(raw)
            content = data.get("response", raw)
            stats = data.get("stats", {})
            models_stats = stats.get("models", {})
            total_input = 0
            total_output = 0
            for model_info in models_stats.values():
                tokens = model_info.get("tokens", {})
                total_input += tokens.get("input", 0) or tokens.get("prompt", 0)
                total_output += tokens.get("output", 0) or tokens.get("response", 0)
            if total_input or total_output:
                usage = Usage(
                    prompt_tokens=total_input,
                    completion_tokens=total_output,
                    total_tokens=total_input + total_output,
                )
        except json.JSONDecodeError:
            pass

        if not usage:
            usage = Usage(
                prompt_tokens=estimate_tokens(prompt),
                completion_tokens=estimate_tokens(content),
                total_tokens=estimate_tokens(prompt) + estimate_tokens(content),
            )

        return ChatCompletionResponse(
            model=f"gemini/{model}",
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
        session: SessionRequest | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model, output_format="stream-json")

        proc = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
        )

        chunk_id = f"chatcmpl-gemini-{id(proc)}"
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
                text = None
                event_type = event.get("type", "")

                # Gemini stream-json events:
                # {"type":"init","session_id":"...","model":"..."}
                # {"type":"message","role":"assistant","content":"...","delta":true}
                # {"type":"content","value":"..."}
                # {"type":"tool_use","tool_name":"..."}
                # {"type":"tool_result","tool_id":"...","output":"..."}
                # {"type":"result","status":"success","stats":{...}}
                if event_type == "message" and event.get("role") == "assistant":
                    text = event.get("content", "")
                elif event_type == "content":
                    text = event.get("value", "")
                elif event_type == "result":
                    if event.get("status") not in {None, "success"}:
                        stream_error = f"Gemini stream failed: {event.get('status', 'unknown')}"
                        break
                    continue  # Stats-only, skip

                if text:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=f"gemini/{model}",
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
            except json.JSONDecodeError:
                if line:
                    yield ChatCompletionChunk(
                        id=chunk_id,
                        model=f"gemini/{model}",
                        choices=[
                            StreamChoice(
                                delta=Delta(
                                    role="assistant" if first else None,
                                    content=line + "\n",
                                )
                            )
                        ],
                    )
                    first = False

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
                message=f"Gemini CLI failed (exit {proc.returncode}): {stderr_text}",
            )

        yield ChatCompletionChunk(
            id=chunk_id,
            model=f"gemini/{model}",
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )
