"""OpenAI-compatible response types for SubLLM."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, ClassVar, Literal, Mapping, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict, Field

from subllm.errors import MalformedRequestError, UnsupportedFeatureError


class RequestMessage(BaseModel):
    """Supported request message subset."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    role: Literal["system", "user", "assistant"]
    content: str


class ProviderMessage(TypedDict):
    role: Literal["user", "assistant"]
    content: str


class PromptReference(BaseModel):
    """Reference to a registered prompt version."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    variables: dict[str, str] = Field(default_factory=dict)


class ModelDescriptor(TypedDict):
    id: str
    provider: str


class CompletionRequest(BaseModel):
    """Internal request contract shared by the Python API and HTTP boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    supported_fields: ClassVar[frozenset[str]] = frozenset(
        {"model", "messages", "stream", "system_prompt", "max_tokens", "temperature", "prompt"}
    )

    model: str = Field(min_length=1)
    messages: list[RequestMessage] = Field(min_length=1)
    stream: bool = False
    system_prompt: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    prompt: PromptReference | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> CompletionRequest:
        unsupported_fields = sorted(set(data) - cls.supported_fields)
        if unsupported_fields:
            field_list = ", ".join(unsupported_fields)
            raise UnsupportedFeatureError(
                field="request",
                message=f"Unsupported request fields: {field_list}",
            )

        try:
            return cls.model_validate(data)
        except Exception as exc:
            raise MalformedRequestError.from_validation_error(exc) from exc

    @classmethod
    def from_inputs(
        cls,
        *,
        model: str,
        messages: Sequence[Mapping[str, Any]],
        stream: bool = False,
        system_prompt: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        prompt: Mapping[str, Any] | None = None,
    ) -> CompletionRequest:
        return cls.from_mapping(
            {
                "model": model,
                "messages": [dict(message) for message in messages],
                "stream": stream,
                "system_prompt": system_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "prompt": dict(prompt) if prompt is not None else None,
            }
        )

    @property
    def effective_system_prompt(self) -> str | None:
        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.extend(message.content for message in self.messages if message.role == "system")
        return "\n\n".join(parts) if parts else None

    def compose_system_prompt(self, prompt_text: str | None = None) -> str | None:
        parts: list[str] = []
        if prompt_text:
            parts.append(prompt_text)
        if self.effective_system_prompt:
            parts.append(self.effective_system_prompt)
        return "\n\n".join(parts) if parts else None

    @property
    def provider_messages(self) -> list[ProviderMessage]:
        return [
            {"role": message.role, "content": message.content}
            for message in self.messages
            if message.role != "system"
        ]


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

    def to_dict(self) -> dict[str, Any]:
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

    def to_dict(self) -> dict[str, Any]:
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
