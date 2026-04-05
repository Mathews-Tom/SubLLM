"""Eval harness for transcript-driven provider contract checks."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from unittest.mock import patch

from subllm.providers.base import Provider
from subllm.providers.claude_code import ClaudeCodeProvider
from subllm.providers.codex import CodexProvider
from subllm.providers.gemini_cli import GeminiCLIProvider
from subllm.types import ChatCompletionChunk, ChatCompletionResponse, ProviderMessage

CaseMode = Literal["completion", "stream"]
ProviderName = Literal["claude-code", "codex", "gemini"]


@dataclass(frozen=True)
class ContractExpectation:
    content: str | None = None
    usage: dict[str, int] | None = None
    chunk_contents: list[str] = field(default_factory=list)
    error_type: str | None = None
    error_message_substring: str | None = None


@dataclass(frozen=True)
class ContractCase:
    name: str
    provider: ProviderName
    mode: CaseMode
    model: str
    messages: list[ProviderMessage]
    stdout_lines: list[str] = field(default_factory=list)
    stderr_text: str = ""
    returncode: int = 0
    communicate_delay: float = 0.0
    readline_delay: float = 0.0
    sdk_events: list[dict[str, Any]] = field(default_factory=list)
    expectation: ContractExpectation = field(default_factory=ContractExpectation)

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> ContractCase:
        expectation = ContractExpectation(**payload.get("expectation", {}))
        return cls(
            name=payload["name"],
            provider=payload["provider"],
            mode=payload["mode"],
            model=payload["model"],
            messages=payload["messages"],
            stdout_lines=payload.get("stdout_lines", []),
            stderr_text=payload.get("stderr_text", ""),
            returncode=payload.get("returncode", 0),
            communicate_delay=payload.get("communicate_delay", 0.0),
            readline_delay=payload.get("readline_delay", 0.0),
            sdk_events=payload.get("sdk_events", []),
            expectation=expectation,
        )


@dataclass(frozen=True)
class ContractResult:
    name: str
    passed: bool
    provider: ProviderName
    mode: CaseMode
    detail: str


@dataclass(frozen=True)
class ContractSuiteResult:
    cases: list[ContractResult]

    @property
    def passed(self) -> int:
        return sum(1 for case in self.cases if case.passed)

    @property
    def failed(self) -> int:
        return len(self.cases) - self.passed

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "cases": [
                {
                    "name": case.name,
                    "provider": case.provider,
                    "mode": case.mode,
                    "passed": case.passed,
                    "detail": case.detail,
                }
                for case in self.cases
            ],
        }


class _FakeStreamReader:
    def __init__(self, lines: list[str], *, delay: float = 0.0) -> None:
        self._lines = [line.encode() for line in lines]
        self._delay = delay

    async def readline(self) -> bytes:
        if self._delay:
            await asyncio.sleep(self._delay)
        if not self._lines:
            return b""
        return self._lines.pop(0)


class _FakeStderr:
    def __init__(self, text: str) -> None:
        self._text = text.encode()
        self._read = False

    async def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._text


class _FakeProcess:
    def __init__(
        self,
        *,
        stdout_lines: list[str],
        stderr_text: str,
        returncode: int,
        communicate_delay: float = 0.0,
        readline_delay: float = 0.0,
    ) -> None:
        self.stdout = _FakeStreamReader(stdout_lines, delay=readline_delay)
        self.stderr = _FakeStderr(stderr_text)
        self.returncode = None if communicate_delay or readline_delay else returncode
        self._stdout = "".join(stdout_lines).encode()
        self._final_returncode = returncode
        self._communicate_delay = communicate_delay

    async def communicate(self) -> tuple[bytes, bytes]:
        if self._communicate_delay:
            await asyncio.sleep(self._communicate_delay)
        self.returncode = self._final_returncode
        return self._stdout, await self.stderr.read()

    async def wait(self) -> int:
        self.returncode = self._final_returncode
        return self._final_returncode

    def kill(self) -> None:
        self.returncode = -9


@dataclass
class _FakeTextBlock:
    text: str


@dataclass
class _FakeAssistantMessage:
    content: list[_FakeTextBlock] = field(default_factory=list)


@dataclass
class _FakeResultMessage:
    is_error: bool
    result: str | None = None
    usage: dict[str, int] | None = None


class _FakeClaudeClient:
    def __init__(self, events: list[dict[str, Any]]) -> None:
        self._events = events

    async def query(self, prompt: str) -> None:
        return None

    async def receive_response(self):  # type: ignore[no-untyped-def]
        for event in self._events:
            if event["type"] == "assistant":
                yield _FakeAssistantMessage(
                    content=[_FakeTextBlock(text=text) for text in event.get("texts", [])]
                )
            elif event["type"] == "result":
                yield _FakeResultMessage(
                    is_error=event.get("is_error", False),
                    result=event.get("result"),
                    usage=event.get("usage"),
                )

    async def disconnect(self) -> None:
        return None


def default_contract_fixture_dir() -> Path:
    return Path("tests/fixtures/provider_contracts")


def load_contract_cases(fixture_dir: str | Path) -> list[ContractCase]:
    root = Path(fixture_dir)
    return [
        ContractCase.from_json(json.loads(path.read_text()))
        for path in sorted(root.glob("*.json"))
    ]


async def run_contract_suite(fixture_dir: str | Path) -> ContractSuiteResult:
    cases = load_contract_cases(fixture_dir)
    results = [await run_contract_case(case) for case in cases]
    return ContractSuiteResult(cases=results)


async def run_contract_case(case: ContractCase) -> ContractResult:
    try:
        if case.provider == "claude-code":
            detail = await _run_claude_case(case)
        else:
            detail = await _run_subprocess_case(case)
        return ContractResult(
            name=case.name,
            passed=True,
            provider=case.provider,
            mode=case.mode,
            detail=detail,
        )
    except Exception as exc:
        return ContractResult(
            name=case.name,
            passed=False,
            provider=case.provider,
            mode=case.mode,
            detail=str(exc),
        )


async def _run_subprocess_case(case: ContractCase) -> str:
    provider, patch_target = _subprocess_provider(case.provider)

    async def fake_create_subprocess_exec(*args: Any, **kwargs: Any) -> _FakeProcess:
        return _FakeProcess(
            stdout_lines=case.stdout_lines,
            stderr_text=case.stderr_text,
            returncode=case.returncode,
            communicate_delay=case.communicate_delay,
            readline_delay=case.readline_delay,
        )

    with patch(patch_target, fake_create_subprocess_exec):
        if case.mode == "completion":
            response = await provider.complete(case.messages, case.model)
            _assert_completion(case, response)
            return "completion transcript matched"

        chunks, error = await _collect_stream(provider, case.messages, case.model)
        _assert_stream(case, chunks, error)
        return "stream transcript matched"


async def _run_claude_case(case: ContractCase) -> str:
    provider = ClaudeCodeProvider(cli_path="claude")

    async def fake_ensure_client(*args: Any, **kwargs: Any) -> _FakeClaudeClient:
        return _FakeClaudeClient(case.sdk_events)

    with (
        patch("subllm.providers.claude_code.AssistantMessage", _FakeAssistantMessage),
        patch("subllm.providers.claude_code.ResultMessage", _FakeResultMessage),
        patch("subllm.providers.claude_code.TextBlock", _FakeTextBlock),
    ):
        setattr(provider, "_ensure_sdk_client", fake_ensure_client)
        if case.mode == "completion":
            response = await provider.complete(case.messages, case.model)
            _assert_completion(case, response)
            return "claude completion transcript matched"

        chunks, error = await _collect_stream(provider, case.messages, case.model)
        _assert_stream(case, chunks, error)
        return "claude stream transcript matched"


async def _collect_stream(
    provider: Provider,
    messages: list[ProviderMessage],
    model: str,
) -> tuple[list[ChatCompletionChunk], Exception | None]:
    chunks: list[ChatCompletionChunk] = []
    error: Exception | None = None
    stream = provider.stream(messages, model)
    try:
        async for chunk in stream:
            chunks.append(chunk)
    except Exception as exc:
        error = exc
    return chunks, error


def _assert_completion(case: ContractCase, response: ChatCompletionResponse) -> None:
    content = response.choices[0].message.content
    if case.expectation.content is not None and content != case.expectation.content:
        raise AssertionError(
            f"{case.name}: expected content {case.expectation.content!r}, got {content!r}"
        )
    if case.expectation.usage is not None:
        assert response.usage is not None
        actual_usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        if actual_usage != case.expectation.usage:
            raise AssertionError(
                f"{case.name}: expected usage {case.expectation.usage}, got {actual_usage}"
            )


def _assert_stream(
    case: ContractCase,
    chunks: list[ChatCompletionChunk],
    error: Exception | None,
) -> None:
    chunk_contents = [
        chunk.choices[0].delta.content
        for chunk in chunks
        if chunk.choices[0].delta.content is not None
    ]
    if chunk_contents != case.expectation.chunk_contents:
        raise AssertionError(
            f"{case.name}: expected chunks {case.expectation.chunk_contents}, got {chunk_contents}"
        )

    expected_error = case.expectation.error_type
    if expected_error is None and error is not None:
        raise AssertionError(f"{case.name}: expected no stream error, got {error}")
    if expected_error is not None:
        if error is None:
            raise AssertionError(f"{case.name}: expected stream error {expected_error}")
        if error.__class__.__name__ != expected_error:
            raise AssertionError(
                f"{case.name}: expected {expected_error}, got {error.__class__.__name__}"
            )
        if (
            case.expectation.error_message_substring is not None
            and case.expectation.error_message_substring not in str(error)
        ):
            raise AssertionError(
                f"{case.name}: expected error containing "
                f"{case.expectation.error_message_substring!r}, got {error!s}"
            )


def _subprocess_provider(provider_name: ProviderName) -> tuple[Provider, str]:
    if provider_name == "codex":
        return CodexProvider(cli_path="codex"), "subllm.providers.codex.asyncio.create_subprocess_exec"
    if provider_name == "gemini":
        return (
            GeminiCLIProvider(cli_path="gemini"),
            "subllm.providers.gemini_cli.asyncio.create_subprocess_exec",
        )
    raise ValueError(f"Unsupported subprocess provider {provider_name!r}")
