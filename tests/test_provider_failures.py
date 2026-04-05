from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

import pytest

from subllm.errors import ProviderFailureError, ProviderTimeoutError
from subllm.providers.claude_code import ClaudeCodeProvider
from subllm.providers.codex import CodexProvider
from subllm.providers.gemini_cli import GeminiCLIProvider


class FakeStreamReader:
    def __init__(self, lines: list[str], *, delay: float = 0.0) -> None:
        self._lines = [line.encode() for line in lines]
        self._delay = delay

    async def readline(self) -> bytes:
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._lines:
            return b""
        return self._lines.pop(0)


class FakeStderr:
    def __init__(self, text: str) -> None:
        self._text = text.encode()
        self._read = False

    async def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._text


class FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str] | None = None,
        stderr_text: str = "",
        returncode: int = 0,
        communicate_delay: float = 0.0,
        readline_delay: float = 0.0,
    ) -> None:
        self.stdout = FakeStreamReader(stdout_lines or [], delay=readline_delay)
        self.stderr = FakeStderr(stderr_text)
        self.returncode = None if communicate_delay or readline_delay else returncode
        self._final_returncode = returncode
        self._communicate_delay = communicate_delay
        self.killed = False

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._communicate_delay:
            await asyncio.sleep(self._communicate_delay)
        self.returncode = self._final_returncode
        return b"", await self.stderr.read()

    async def wait(self) -> int:
        self.returncode = self._final_returncode
        return self._final_returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


@pytest.mark.asyncio
async def test_codex_completion_timeout_raises_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeProcess(communicate_delay=0.05)

    monkeypatch.setattr(
        "subllm.providers.codex.asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )
    provider = CodexProvider(cli_path="codex", command_timeout=0.01)

    with pytest.raises(ProviderTimeoutError):
        await provider.complete([{"role": "user", "content": "hello"}], "gpt-5.2")


@pytest.mark.asyncio
async def test_codex_stream_raises_on_nonzero_exit_after_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeProcess(
            stdout_lines=[
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "partial"},
                    }
                )
                + "\n"
            ],
            stderr_text="backend failed",
            returncode=1,
        )

    monkeypatch.setattr(
        "subllm.providers.codex.asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )
    provider = CodexProvider(cli_path="codex")
    stream = provider.stream([{"role": "user", "content": "hello"}], "gpt-5.2")

    first_chunk = await anext(stream)
    assert first_chunk.choices[0].delta.content == "partial"

    with pytest.raises(ProviderFailureError):
        await anext(stream)


@pytest.mark.asyncio
async def test_gemini_stream_raises_on_failed_result_after_partial_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeProcess(
            stdout_lines=[
                json.dumps({"type": "content", "value": "partial"}) + "\n",
                json.dumps({"type": "result", "status": "error"}) + "\n",
            ],
            stderr_text="tool prompt blocked",
            returncode=0,
        )

    monkeypatch.setattr(
        "subllm.providers.gemini_cli.asyncio.create_subprocess_exec",
        fake_create_subprocess_exec,
    )
    provider = GeminiCLIProvider(cli_path="gemini")
    stream = provider.stream([{"role": "user", "content": "hello"}], "gemini-3-flash-preview")

    first_chunk = await anext(stream)
    assert first_chunk.choices[0].delta.content == "partial"

    with pytest.raises(ProviderFailureError):
        await anext(stream)


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeAssistantMessage:
    content: list[FakeTextBlock] = field(default_factory=list)


@dataclass
class FakeResultMessage:
    is_error: bool
    result: str | None = None
    usage: dict[str, int] | None = None


class FakeClaudeClient:
    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield FakeResultMessage(is_error=True, result="permission denied")


@pytest.mark.asyncio
async def test_claude_stream_raises_on_result_error(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_ensure_client(*args, **kwargs):  # type: ignore[no-untyped-def]
        return FakeClaudeClient()

    monkeypatch.setattr("subllm.providers.claude_code.AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr("subllm.providers.claude_code.ResultMessage", FakeResultMessage)
    monkeypatch.setattr("subllm.providers.claude_code.TextBlock", FakeTextBlock)

    provider = ClaudeCodeProvider(cli_path="claude")
    monkeypatch.setattr(provider, "_ensure_sdk_client", fake_ensure_client)

    stream = provider.stream([{"role": "user", "content": "hello"}], "sonnet-4-5")

    with pytest.raises(ProviderFailureError):
        await anext(stream)
