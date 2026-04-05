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
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

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
    ResolvedImageInput,
    ResponseSession,
    SessionRequest,
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

    def _build_cli_args(
        self,
        prompt: str,
        model: str,
        *,
        image_paths: list[str] | None = None,
        session: SessionRequest | None = None,
    ) -> list[str]:
        resolved = self.resolve_model(model)
        if session is not None and session.mode == "resume":
            args = [self._cli_path, "exec", "resume", session.id or "", prompt]
        else:
            args = [self._cli_path, "exec", prompt]
        args.extend(["--model", resolved, "--json"])
        for image_path in image_paths or []:
            args.extend(["--image", image_path])
        return args

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

    @asynccontextmanager
    async def _prepared_image_paths(
        self,
        messages: list[ProviderMessage],
    ) -> AsyncIterator[list[str]]:
        image_paths: list[str] = []
        temp_paths: list[str] = []
        try:
            for message in messages:
                for image in message.get("images", []):
                    image_path = self._prepare_image_path(image, temp_paths)
                    image_paths.append(image_path)
            yield image_paths
        finally:
            for temp_path in temp_paths:
                try:
                    os.remove(temp_path)
                except FileNotFoundError:
                    continue

    def _prepare_image_path(
        self,
        image: ResolvedImageInput,
        temp_paths: list[str],
    ) -> str:
        if image.file_path is not None:
            return image.file_path

        if image.data is None:
            raise ProviderFailureError(
                provider=self.name,
                message=f"Image '{image.filename}' is missing payload data",
            )

        suffix = _suffix_for_media_type(image.media_type)
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as handle:
            handle.write(image.data)
            temp_paths.append(handle.name)
            return handle.name

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
        session: SessionRequest | None = None,
    ) -> ChatCompletionResponse:
        prompt = messages_to_prompt(messages, system_prompt)
        async with self._prepared_image_paths(messages) as image_paths:
            args = self._build_cli_args(prompt, model, image_paths=image_paths, session=session)

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
        session_info = _session_response_for(session)
        for line in stdout.decode().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_type = event.get("type")
                if event_type == "thread.started":
                    session_info = _resolve_codex_session(session, event.get("thread_id"))
                elif event_type == "item.completed":
                    item = event.get("item", {})
                    if item.get("type") == "agent_message":
                        content_parts.append(item.get("text", ""))
                elif event_type == "turn.completed":
                    stats = event.get("usage", {})
                    inp = stats.get("input_tokens", 0)
                    out = stats.get("output_tokens", 0)
                    usage = Usage(prompt_tokens=inp, completion_tokens=out, total_tokens=inp + out)
            except json.JSONDecodeError:
                continue

        content = "\n".join(content_parts)
        session_info = _require_create_session(session, session_info)
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
            session=session_info,
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
        async with self._prepared_image_paths(messages) as image_paths:
            args = self._build_cli_args(prompt, model, image_paths=image_paths, session=session)

            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            chunk_id = f"chatcmpl-codex-{id(proc)}"
            first = True
            stream_error: str | None = None
            session_info = _session_response_for(session)

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
                    event_type = event.get("type")
                    if event_type == "thread.started":
                        session_info = _resolve_codex_session(session, event.get("thread_id"))
                    elif event_type == "item.completed":
                        item = event.get("item", {})
                        if item.get("type") == "agent_message":
                            text = item.get("text", "")
                            if text:
                                yield ChatCompletionChunk(
                                    id=chunk_id,
                                    model=f"codex/{model}",
                                    session=session_info,
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
                    elif event_type == "error":
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

            session_info = _require_create_session(session, session_info)
            yield ChatCompletionChunk(
                id=chunk_id,
                model=f"codex/{model}",
                session=session_info,
                choices=[StreamChoice(delta=Delta(), finish_reason="stop")],
            )


def _suffix_for_media_type(media_type: str) -> str:
    return {
        "image/gif": ".gif",
        "image/jpeg": ".jpg",
        "image/png": ".png",
        "image/webp": ".webp",
    }.get(media_type, ".bin")


def _session_response_for(session: SessionRequest | None) -> ResponseSession | None:
    if session is None or session.mode != "resume" or session.id is None:
        return None
    return ResponseSession(id=session.id, mode="resume")


def _resolve_codex_session(
    session: SessionRequest | None,
    thread_id: str | None,
) -> ResponseSession | None:
    if session is None:
        return None
    if thread_id:
        return ResponseSession(id=thread_id, mode=session.mode)
    if session.mode == "resume" and session.id is not None:
        return ResponseSession(id=session.id, mode="resume")
    raise ProviderFailureError(
        provider="codex",
        message="Codex did not report a resumable session identifier",
    )


def _require_create_session(
    session: SessionRequest | None,
    session_info: ResponseSession | None,
) -> ResponseSession | None:
    if session is None or session.mode != "create":
        return session_info
    if session_info is None:
        raise ProviderFailureError(
            provider="codex",
            message="Codex did not report a resumable session identifier",
        )
    return session_info
