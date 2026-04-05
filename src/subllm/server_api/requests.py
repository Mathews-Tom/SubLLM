"""Request parsing and validation helpers for the HTTP boundary."""

from __future__ import annotations

from typing import Any

from fastapi import Request

from subllm.errors import MalformedRequestError
from subllm.types import CompletionRequest


async def parse_json_object(request: Request) -> dict[str, Any]:
    try:
        body = await request.json()
    except Exception as exc:
        raise MalformedRequestError("Request body must be valid JSON") from exc

    if not isinstance(body, dict):
        raise MalformedRequestError("Request body must be a JSON object")

    return body


async def parse_chat_completion_request(request: Request) -> CompletionRequest:
    body = await parse_json_object(request)
    return CompletionRequest.from_mapping(body)
