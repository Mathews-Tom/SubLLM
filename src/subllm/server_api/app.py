"""FastAPI application factory."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

from subllm.cache import CacheConfig, ResponseCache, clear_cache_status, response_cache_headers
from subllm.router import Router
from subllm.server_api.errors import install_exception_handlers
from subllm.server_api.middleware import install_server_controls
from subllm.server_api.requests import parse_chat_completion_request
from subllm.server_api.responses import (
    build_health_response,
    build_model_list,
    sse_chat_completion_stream,
)
from subllm.server_api.settings import ServerSettings
from subllm.telemetry import (
    TelemetryConfig,
    configure_telemetry,
    response_context_headers,
    shutdown_telemetry,
)
from subllm.types import ChatCompletionResponse


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    active_settings = settings or ServerSettings.from_inputs()
    response_cache: ResponseCache | None = None
    if active_settings.response_cache_ttl_seconds is not None:
        response_cache = ResponseCache(
            CacheConfig(
                ttl_seconds=active_settings.response_cache_ttl_seconds,
                max_entries=active_settings.response_cache_max_entries,
            )
        )
    router = Router(response_cache=response_cache)

    @asynccontextmanager
    async def app_lifespan(_app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await router.close()
            shutdown_telemetry()

    app = FastAPI(title="SubLLM Proxy", version="0.2.0", lifespan=app_lifespan)
    configure_telemetry(
        TelemetryConfig(
            service_name=active_settings.trace_service_name,
            export_path=active_settings.trace_export_path,
        )
    )
    install_exception_handlers(app)
    install_server_controls(app, active_settings)

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        payload = build_model_list(router.list_models())
        return JSONResponse(payload.model_dump(), headers=response_context_headers())

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> Response:
        clear_cache_status()
        completion_request = await parse_chat_completion_request(request)

        if completion_request.stream:
            stream_result = router.stream_request(completion_request)
            return StreamingResponse(
                sse_chat_completion_stream(stream_result),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    **response_context_headers(),
                },
            )

        completion_result: ChatCompletionResponse = await router.complete_request(
            completion_request
        )
        return JSONResponse(
            completion_result.to_dict(),
            headers={**response_context_headers(), **response_cache_headers()},
        )

    @app.get("/health")
    async def health() -> JSONResponse:
        payload = build_health_response(await router.check_auth())
        return JSONResponse(payload.model_dump(), headers=response_context_headers())

    return app
