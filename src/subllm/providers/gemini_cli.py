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
    "2.5-pro": "gemini-2.5-pro",
    "2.5-flash": "gemini-2.5-flash",
    "2.5-pro-exp": "gemini-2.5-pro-exp-03-25",
    "2.0-flash": "gemini-2.0-flash",
    "pro": "gemini-2.5-pro",
    "flash": "gemini-2.5-flash",
}


class GeminiCLIProvider(Provider):
    """Routes completions through the Google Gemini CLI."""

    def __init__(
        self,
        cli_path: str | None = None,
        env_overrides: dict[str, str] | None = None,
        yolo_mode: bool = False,
    ):
        self._cli_path = cli_path or shutil.which("gemini") or "gemini"
        self._env_overrides = env_overrides or {}
        self._yolo_mode = yolo_mode  # --yolo skips all tool confirmations

    @property
    def name(self) -> str:
        return "gemini"

    @property
    def supported_models(self) -> list[str]:
        return list(_MODEL_MAP.keys())

    @property
    def capabilities(self) -> ProviderCapabilities:
        return ProviderCapabilities(
            supports_streaming=True,
            supports_sessions=False,  # Not in headless -p mode
            supports_system_prompt=True,
            supports_vision=True,
            max_context_tokens=1_000_000,  # Gemini 2.5 Pro: 1M context
            subscription_auth=True,  # Google AI Pro/Ultra
            api_key_auth=True,  # GEMINI_API_KEY (free tier)
        )

    def resolve_model(self, model_alias: str) -> str:
        return _MODEL_MAP.get(model_alias, model_alias)

    def _build_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env.update(self._env_overrides)
        return env

    async def check_auth(self) -> AuthStatus:
        if os.environ.get("GEMINI_API_KEY"):
            return AuthStatus(provider=self.name, authenticated=True, method="api_key")
        if os.environ.get("GOOGLE_API_KEY"):
            return AuthStatus(provider=self.name, authenticated=True, method="google_api_key")

        if not shutil.which(self._cli_path):
            return AuthStatus(
                provider=self.name, authenticated=False,
                error=f"Gemini CLI not found at '{self._cli_path}'. "
                "Install: npm install -g @google/gemini-cli or see https://geminicli.com",
            )

        try:
            proc = await asyncio.create_subprocess_exec(
                self._cli_path, "-p", "say ok", "--output-format", "json",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                env=self._build_env(),
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=30)
            if proc.returncode == 0:
                return AuthStatus(provider=self.name, authenticated=True, method="google_oauth")
            err_msg = stderr.decode().strip()
            return AuthStatus(
                provider=self.name, authenticated=False,
                error=f"Not authenticated: {err_msg}. "
                "Run `gemini` and complete Google login, or set GEMINI_API_KEY.",
            )
        except asyncio.TimeoutError:
            return AuthStatus(
                provider=self.name, authenticated=False, error="Auth check timed out.",
            )
        except FileNotFoundError:
            return AuthStatus(
                provider=self.name, authenticated=False, error="Gemini CLI not found.",
            )

    def _build_cli_args(
        self, prompt: str, model: str, *, output_format: str = "json",
    ) -> list[str]:
        resolved = self.resolve_model(model)
        args = [self._cli_path, "-p", prompt, "--model", resolved, "--output-format", output_format]
        if self._yolo_mode:
            args.append("--yolo")
        return args

    async def complete(
        self, messages: list[dict], model: str, *,
        system_prompt: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> ChatCompletionResponse:
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model, output_format="json")

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            raise RuntimeError(
                f"Gemini CLI failed (exit {proc.returncode}): {stderr.decode().strip()}"
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
            choices=[Choice(message=Message(role="assistant", content=content), finish_reason="stop")],
            usage=usage,
        )

    async def stream(
        self, messages: list[dict], model: str, *,
        system_prompt: str | None = None, max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> AsyncIterator[ChatCompletionChunk]:
        prompt = messages_to_prompt(messages, system_prompt)
        args = self._build_cli_args(prompt, model, output_format="stream-json")

        proc = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            env=self._build_env(),
        )

        chunk_id = f"chatcmpl-gemini-{id(proc)}"
        first = True

        assert proc.stdout is not None
        async for line_bytes in proc.stdout:
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
                    continue  # Stats-only, skip

                if text:
                    yield ChatCompletionChunk(
                        id=chunk_id, model=f"gemini/{model}",
                        choices=[StreamChoice(delta=Delta(
                            role="assistant" if first else None, content=text,
                        ))],
                    )
                    first = False
            except json.JSONDecodeError:
                if line:
                    yield ChatCompletionChunk(
                        id=chunk_id, model=f"gemini/{model}",
                        choices=[StreamChoice(delta=Delta(
                            role="assistant" if first else None, content=line + "\n",
                        ))],
                    )
                    first = False

        yield ChatCompletionChunk(
            id=chunk_id, model=f"gemini/{model}",
            choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
        )
        await proc.wait()
