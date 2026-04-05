from __future__ import annotations

import json

from fastapi.testclient import TestClient
import pytest

from subllm.cache import CacheConfig, ResponseCache
from subllm.errors import MalformedRequestError, ProviderFailureError, UnsupportedFeatureError
from subllm.providers.codex import CodexProvider
from subllm.router import Router
from subllm.server import create_app
from subllm.server_api.settings import ServerSettings
from subllm.types import (
    ChatCompletionResponse,
    Choice,
    CompletionRequest,
    Message,
    ResponseSession,
    SessionRequest,
)


def test_completion_request_accepts_explicit_session_create() -> None:
    request = CompletionRequest.from_mapping(
        {
            "model": "claude-code/sonnet-4-5",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"mode": "create"},
        }
    )

    assert request.session == SessionRequest(mode="create")


def test_completion_request_rejects_resume_without_session_id() -> None:
    with pytest.raises(MalformedRequestError) as exc_info:
        CompletionRequest.from_mapping(
            {
                "model": "claude-code/sonnet-4-5",
                "messages": [{"role": "user", "content": "hello"}],
                "session": {"mode": "resume"},
            }
        )

    assert "session.id" in str(exc_info.value)


def test_completion_request_rejects_create_with_session_id() -> None:
    with pytest.raises(MalformedRequestError) as exc_info:
        CompletionRequest.from_mapping(
            {
                "model": "claude-code/sonnet-4-5",
                "messages": [{"role": "user", "content": "hello"}],
                "session": {"mode": "create", "id": "abc"},
            }
        )

    assert "must not include session.id" in str(exc_info.value)


@pytest.mark.asyncio
async def test_router_rejects_explicit_sessions_for_unsupported_provider() -> None:
    router = Router()
    request = CompletionRequest.from_mapping(
        {
            "model": "gemini/gemini-3-flash-preview",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"mode": "create"},
        }
    )

    with pytest.raises(UnsupportedFeatureError) as exc_info:
        await router.complete_request(request)

    assert exc_info.value.param == "session"


@pytest.mark.asyncio
async def test_router_bypasses_response_cache_for_explicit_sessions() -> None:
    router = Router(response_cache=ResponseCache(CacheConfig(ttl_seconds=60, max_entries=4)))
    request = CompletionRequest.from_mapping(
        {
            "model": "codex/gpt-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"mode": "create"},
        }
    )

    assert (
        router._build_cache_key(
            provider_name="codex",
            request=request,
            resolved_prompt=None,
            system_prompt=request.compose_system_prompt(),
        )
        is None
    )


class _FakeCommunicatingProcess:
    def __init__(
        self, stdout_lines: list[str], *, returncode: int = 0, stderr_text: str = ""
    ) -> None:
        self.stdout = None
        self.stderr = _FakeReadAll(stderr_text)
        self.returncode = returncode
        self._stdout = "".join(stdout_lines).encode("utf-8")

    async def communicate(self) -> tuple[bytes, bytes]:
        return self._stdout, await self.stderr.read()

    async def wait(self) -> int:
        return self.returncode

    def kill(self) -> None:
        self.returncode = -9


class _FakeReadAll:
    def __init__(self, text: str) -> None:
        self._text = text.encode("utf-8")
        self._read = False

    async def read(self) -> bytes:
        if self._read:
            return b""
        self._read = True
        return self._text


@pytest.mark.asyncio
async def test_codex_create_session_returns_reported_thread_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeCommunicatingProcess(
            [
                json.dumps({"type": "thread.started", "thread_id": "thread-123"}) + "\n",
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                )
                + "\n",
                json.dumps(
                    {"type": "turn.completed", "usage": {"input_tokens": 1, "output_tokens": 1}}
                )
                + "\n",
            ]
        )

    monkeypatch.setattr(
        "subllm.providers.codex.asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )
    provider = CodexProvider(cli_path="codex")

    response = await provider.complete(
        [{"role": "user", "content": "hello"}],
        "gpt-5.2",
        session=SessionRequest(mode="create"),
    )

    assert response.session == ResponseSession(id="thread-123", mode="create")


def test_codex_resume_session_uses_resume_subcommand() -> None:
    provider = CodexProvider(cli_path="codex")

    args = provider._build_cli_args(
        "hello",
        "gpt-5.2",
        session=SessionRequest(mode="resume", id="thread-123"),
    )

    assert args[:5] == ["codex", "exec", "resume", "thread-123", "hello"]


def test_server_returns_explicit_session_metadata(monkeypatch: pytest.MonkeyPatch) -> None:
    async def fake_complete_request(self, request):  # type: ignore[no-untyped-def]
        return ChatCompletionResponse(
            model=request.model,
            choices=[Choice(message=Message(role="assistant", content="ok"), finish_reason="stop")],
            session=ResponseSession(id="thread-123", mode="create"),
        )

    monkeypatch.setattr("subllm.server_api.app.Router.complete_request", fake_complete_request)
    client = TestClient(create_app(ServerSettings()))

    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "codex/gpt-5.2",
            "messages": [{"role": "user", "content": "hello"}],
            "session": {"mode": "create"},
        },
    )

    assert response.status_code == 200
    assert response.json()["session"] == {"id": "thread-123", "mode": "create"}


@pytest.mark.asyncio
async def test_codex_create_session_requires_resumable_identifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_create_subprocess_exec(*args, **kwargs):  # type: ignore[no-untyped-def]
        return _FakeCommunicatingProcess(
            [
                json.dumps(
                    {
                        "type": "item.completed",
                        "item": {"type": "agent_message", "text": "ok"},
                    }
                )
                + "\n"
            ]
        )

    monkeypatch.setattr(
        "subllm.providers.codex.asyncio.create_subprocess_exec", fake_create_subprocess_exec
    )
    provider = CodexProvider(cli_path="codex")

    with pytest.raises(ProviderFailureError):
        await provider.complete(
            [{"role": "user", "content": "hello"}],
            "gpt-5.2",
            session=SessionRequest(mode="create"),
        )
