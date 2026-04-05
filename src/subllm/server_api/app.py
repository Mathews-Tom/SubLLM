"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.responses import Response

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
from subllm.types import ChatCompletionResponse


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(title="SubLLM Proxy", version="0.2.0")
    router = Router()
    active_settings = settings or ServerSettings.from_inputs()
    install_exception_handlers(app)
    install_server_controls(app, active_settings)

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        payload = build_model_list(router.list_models())
        return JSONResponse(payload.model_dump())

    @app.post("/v1/chat/completions", response_model=None)
    async def chat_completions(request: Request) -> Response:
        completion_request = await parse_chat_completion_request(request)

        if completion_request.stream:
            stream_result = router.stream_request(completion_request)
            return StreamingResponse(
                sse_chat_completion_stream(stream_result),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        completion_result: ChatCompletionResponse = await router.complete_request(
            completion_request
        )
        return JSONResponse(completion_result.to_dict())

    @app.get("/health")
    async def health() -> JSONResponse:
        payload = build_health_response(await router.check_auth())
        return JSONResponse(payload.model_dump())

    return app
