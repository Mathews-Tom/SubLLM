"""OpenAI-compatible response types for SubLLM."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Literal, Mapping, Sequence, TypedDict

from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import NotRequired

from subllm.errors import MalformedRequestError, UnsupportedFeatureError


class RequestMessage(BaseModel):
    """Supported request message subset."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    role: Literal["system", "user", "assistant"]
    content: str | list[MessageContentPart]


class TextContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    type: Literal["text"]
    text: str


class ImageUrlSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    url: str = Field(min_length=1)


class ImageContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    type: Literal["image_url"]
    image_url: ImageUrlSpec


class FileContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    type: Literal["input_file"]
    file_path: str = Field(min_length=1)


MessageContentPart = Annotated[
    TextContentPart | ImageContentPart | FileContentPart,
    Field(discriminator="type"),
]


@dataclass(frozen=True)
class ResolvedImageInput:
    filename: str
    media_type: str
    file_path: str | None = None
    data: bytes | None = None


class ProviderMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str
    images: NotRequired[list[ResolvedImageInput]]


class PromptReference(BaseModel):
    """Reference to a registered prompt version."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    name: str = Field(min_length=1)
    version: str | None = Field(default=None, min_length=1)
    variables: dict[str, str] = Field(default_factory=dict)


class SessionRequest(BaseModel):
    """Explicit stateful execution request."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    mode: Literal["create", "resume"]
    id: str | None = Field(default=None, min_length=1)


class ModelDescriptor(TypedDict):
    id: str
    provider: str


class CompletionRequest(BaseModel):
    """Internal request contract shared by the Python API and HTTP boundary."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=False)

    supported_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "model",
            "messages",
            "stream",
            "system_prompt",
            "max_tokens",
            "temperature",
            "prompt",
            "session",
        }
    )

    model: str = Field(min_length=1)
    messages: list[RequestMessage] = Field(min_length=1)
    stream: bool = False
    system_prompt: str | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    prompt: PromptReference | None = None
    session: SessionRequest | None = None

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
            request = cls.model_validate(data)
        except Exception as exc:
            raise MalformedRequestError.from_validation_error(exc) from exc
        request._validate_session_shape()
        return request

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
        session: Mapping[str, Any] | None = None,
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
                "session": dict(session) if session is not None else None,
            }
        )

    def _validate_session_shape(self) -> None:
        if self.session is None:
            return
        if self.session.mode == "resume" and self.session.id is None:
            raise MalformedRequestError(
                "Session resume requests must include session.id",
                param="session.id",
            )
        if self.session.mode == "create" and self.session.id is not None:
            raise MalformedRequestError(
                "Session create requests must not include session.id",
                param="session.id",
            )

    @property
    def effective_system_prompt(self) -> str | None:
        from subllm.attachments import provider_message_from_request_message

        parts: list[str] = []
        if self.system_prompt:
            parts.append(self.system_prompt)
        parts.extend(
            provider_message_from_request_message(message)["content"]
            for message in self.messages
            if message.role == "system"
        )
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
        from subllm.attachments import provider_message_from_request_message

        return [
            provider_message_from_request_message(message)
            for message in self.messages
            if message.role != "system"
        ]


@dataclass
class Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class ResponseSession:
    id: str
    mode: Literal["create", "resume"]


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
    session: ResponseSession | None = None

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
            "session": (
                {
                    "id": self.session.id,
                    "mode": self.session.mode,
                }
                if self.session is not None
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
    session: ResponseSession | None = None

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
            "session": (
                {
                    "id": self.session.id,
                    "mode": self.session.mode,
                }
                if self.session is not None
                else None
            ),
        }


@dataclass
class AuthStatus:
    provider: str
    authenticated: bool
    method: str | None = None  # "subscription", "api_key", "oauth_token", "google_oauth"
    error: str | None = None
