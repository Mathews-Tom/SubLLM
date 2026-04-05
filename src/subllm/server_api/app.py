"""FastAPI application factory."""

from __future__ import annotations

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from subllm.router import Router
from subllm.server_api.errors import install_exception_handlers
from subllm.server_api.requests import parse_chat_completion_request
from subllm.server_api.responses import (
    build_health_response,
    build_model_list,
    sse_chat_completion_stream,
)


def create_app() -> FastAPI:
    """Create FastAPI app with OpenAI-compatible endpoints."""
    app = FastAPI(title="SubLLM Proxy", version="0.2.0")
    router = Router()
    install_exception_handlers(app)

    @app.get("/v1/models")
    async def list_models() -> JSONResponse:
        payload = build_model_list(router.list_models())
        return JSONResponse(payload.model_dump())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        completion_request = await parse_chat_completion_request(request)

        if completion_request.stream:
            result = await router.completion(
                model=completion_request.model,
                messages=completion_request.provider_messages,
                stream=True,
                system_prompt=completion_request.effective_system_prompt,
                max_tokens=completion_request.max_tokens,
                temperature=completion_request.temperature,
            )
            return StreamingResponse(
                sse_chat_completion_stream(result),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )

        result = await router.completion(
            model=completion_request.model,
            messages=completion_request.provider_messages,
            system_prompt=completion_request.effective_system_prompt,
            max_tokens=completion_request.max_tokens,
            temperature=completion_request.temperature,
        )
        return JSONResponse(result.to_dict())

    @app.get("/health")
    async def health() -> JSONResponse:
        payload = build_health_response(await router.check_auth())
        return JSONResponse(payload.model_dump())

    return app
