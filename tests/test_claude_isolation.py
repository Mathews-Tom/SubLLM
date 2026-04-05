from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

import pytest

from subllm.providers.claude_code import ClaudeCodeProvider
from subllm.types import SessionRequest


@dataclass
class FakeTextBlock:
    text: str


@dataclass
class FakeAssistantMessage:
    content: list[FakeTextBlock] = field(default_factory=list)


@dataclass
class FakeResultMessage:
    is_error: bool = False
    result: str | None = None
    usage: dict[str, int] | None = None
    session_id: str = "fake-session"


class FakeClaudeClient:
    def __init__(self) -> None:
        self.query_count = 0
        self.max_concurrent_queries = 0
        self._active_queries = 0
        self.session_ids: list[str] = []

    async def query(self, prompt: str, session_id: str = "default") -> None:
        self.query_count += 1
        self.session_ids.append(session_id)
        self._active_queries += 1
        self.max_concurrent_queries = max(self.max_concurrent_queries, self._active_queries)
        await asyncio.sleep(0.01)
        self._active_queries -= 1

    async def receive_response(self):  # type: ignore[no-untyped-def]
        yield FakeAssistantMessage(content=[FakeTextBlock(text="ok")])
        yield FakeResultMessage(usage={"input_tokens": 1, "output_tokens": 1})

    async def disconnect(self) -> None:
        return None


@pytest.mark.asyncio
async def test_claude_reuses_client_for_identical_config(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[FakeClaudeClient] = []

    async def fake_create_client(self, options):  # type: ignore[no-untyped-def]
        client = FakeClaudeClient()
        created_clients.append(client)
        return client

    provider = ClaudeCodeProvider(cli_path="claude")
    monkeypatch.setattr(provider, "_create_sdk_client", fake_create_client.__get__(provider))

    first = await provider._ensure_sdk_client("sonnet-4-5", system_prompt="same")
    second = await provider._ensure_sdk_client("sonnet-4-5", system_prompt="same")

    assert first is second
    assert len(created_clients) == 1


@pytest.mark.asyncio
async def test_claude_isolates_clients_by_config(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[FakeClaudeClient] = []

    async def fake_create_client(self, options):  # type: ignore[no-untyped-def]
        client = FakeClaudeClient()
        created_clients.append(client)
        return client

    provider = ClaudeCodeProvider(cli_path="claude")
    monkeypatch.setattr(provider, "_create_sdk_client", fake_create_client.__get__(provider))

    first = await provider._ensure_sdk_client("sonnet-4-5", system_prompt="alpha")
    second = await provider._ensure_sdk_client("sonnet-4-5", system_prompt="beta")

    assert first is not second
    assert len(created_clients) == 2


@pytest.mark.asyncio
async def test_claude_serializes_same_key_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    shared_client = FakeClaudeClient()

    async def fake_create_client(self, options):  # type: ignore[no-untyped-def]
        return shared_client

    monkeypatch.setattr("subllm.providers.claude_code.AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr("subllm.providers.claude_code.ResultMessage", FakeResultMessage)
    monkeypatch.setattr("subllm.providers.claude_code.TextBlock", FakeTextBlock)

    provider = ClaudeCodeProvider(cli_path="claude")
    monkeypatch.setattr(provider, "_create_sdk_client", fake_create_client.__get__(provider))

    await asyncio.gather(
        provider.complete([{"role": "user", "content": "one"}], "sonnet-4-5", system_prompt="same"),
        provider.complete([{"role": "user", "content": "two"}], "sonnet-4-5", system_prompt="same"),
    )

    assert shared_client.query_count == 2
    assert shared_client.max_concurrent_queries == 1


@pytest.mark.asyncio
async def test_claude_stateless_requests_use_distinct_sdk_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_client = FakeClaudeClient()

    async def fake_create_client(self, options):  # type: ignore[no-untyped-def]
        return shared_client

    monkeypatch.setattr("subllm.providers.claude_code.AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr("subllm.providers.claude_code.ResultMessage", FakeResultMessage)
    monkeypatch.setattr("subllm.providers.claude_code.TextBlock", FakeTextBlock)

    provider = ClaudeCodeProvider(cli_path="claude")
    monkeypatch.setattr(provider, "_create_sdk_client", fake_create_client.__get__(provider))

    await provider.complete([{"role": "user", "content": "one"}], "sonnet-4-5")
    await provider.complete([{"role": "user", "content": "two"}], "sonnet-4-5")

    assert len(shared_client.session_ids) == 2
    assert shared_client.session_ids[0] != shared_client.session_ids[1]


@pytest.mark.asyncio
async def test_claude_resume_session_uses_supplied_session_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_client = FakeClaudeClient()

    async def fake_create_client(self, options):  # type: ignore[no-untyped-def]
        return shared_client

    monkeypatch.setattr("subllm.providers.claude_code.AssistantMessage", FakeAssistantMessage)
    monkeypatch.setattr("subllm.providers.claude_code.ResultMessage", FakeResultMessage)
    monkeypatch.setattr("subllm.providers.claude_code.TextBlock", FakeTextBlock)

    provider = ClaudeCodeProvider(cli_path="claude")
    monkeypatch.setattr(provider, "_create_sdk_client", fake_create_client.__get__(provider))

    response = await provider.complete(
        [{"role": "user", "content": "resume"}],
        "sonnet-4-5",
        session=SessionRequest(mode="resume", id="session-123"),
    )

    assert shared_client.session_ids == ["session-123"]
    assert response.session is not None
    assert response.session.id == "fake-session"
    assert response.session.mode == "resume"
