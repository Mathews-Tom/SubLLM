"""OpenAI-compatible response types for SubLLM."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class Message:
    role: str = "assistant"
    content: str | None = None


@dataclass
class Delta:
    role: str | None = None
    content: str | None = None


@dataclass
class Choice:
    index: int = 0
    message: Message = field(default_factory=Message)
    finish_reason: str | None = None


@dataclass
class StreamChoice:
    index: int = 0
    delta: Delta = field(default_factory=Delta)
    finish_reason: str | None = None


@dataclass
class ChatCompletionResponse:
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[Choice] = field(default_factory=list)
    usage: Usage | None = None

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "message": {"role": c.message.role, "content": c.message.content},
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
            "usage": (
                {
                    "prompt_tokens": self.usage.prompt_tokens,
                    "completion_tokens": self.usage.completion_tokens,
                    "total_tokens": self.usage.total_tokens,
                }
                if self.usage
                else None
            ),
        }


@dataclass
class ChatCompletionChunk:
    id: str = field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: str = "chat.completion.chunk"
    created: int = field(default_factory=lambda: int(time.time()))
    model: str = ""
    choices: list[StreamChoice] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "object": self.object,
            "created": self.created,
            "model": self.model,
            "choices": [
                {
                    "index": c.index,
                    "delta": {"role": c.delta.role, "content": c.delta.content},
                    "finish_reason": c.finish_reason,
                }
                for c in self.choices
            ],
        }


@dataclass
class AuthStatus:
    provider: str
    authenticated: bool
    method: str | None = None  # "subscription", "api_key", "oauth_token", "google_oauth"
    error: str | None = None
