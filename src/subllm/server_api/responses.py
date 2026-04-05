"""Response builders for the FastAPI boundary."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

from subllm.server_api.models import HealthResponse, ModelCard, ModelListResponse, ProviderHealth
from subllm.types import AuthStatus, ChatCompletionChunk, ModelDescriptor


def build_model_list(models: list[ModelDescriptor]) -> ModelListResponse:
    created = int(time.time())
    return ModelListResponse(
        data=[
            ModelCard(
                id=model["id"],
                created=created,
                owned_by=model["provider"],
            )
            for model in models
        ]
    )


def build_health_response(statuses: list[AuthStatus]) -> HealthResponse:
    return HealthResponse(
        status="ok",
        providers=[
            ProviderHealth(
                name=status.provider,
                authenticated=status.authenticated,
                method=status.method,
            )
            for status in statuses
        ],
    )


async def sse_chat_completion_stream(
    result: AsyncIterator[ChatCompletionChunk],
) -> AsyncIterator[str]:
    async for chunk in result:
        yield f"data: {json.dumps(chunk.to_dict())}\n\n"
    yield "data: [DONE]\n\n"
