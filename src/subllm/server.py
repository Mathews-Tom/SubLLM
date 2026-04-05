"""OpenAI-compatible HTTP server for SubLLM.

Run with: subllm serve --port 8080
Then use with ANY OpenAI-compatible client:

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")
    response = client.chat.completions.create(
        model="claude-code/sonnet-4-5",
        messages=[{"role": "user", "content": "hello"}],
    )
"""

import json
import time

from subllm.errors import MalformedRequestError, SubLLMError
from subllm.router import Router
from subllm.types import CompletionRequest


def create_app():
    """Create FastAPI app with OpenAI-compatible endpoints."""
    from fastapi import FastAPI, Request
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(title="SubLLM Proxy", version="0.2.0")
    router = Router()

    @app.get("/v1/models")
    async def list_models():
        models = router.list_models()
        return {
            "object": "list",
            "data": [
                {
                    "id": m["id"],
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": m["provider"],
                }
                for m in models
            ],
        }

    @app.exception_handler(SubLLMError)
    async def handle_subllm_error(_request: Request, exc: SubLLMError) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content=exc.to_response())

    @app.exception_handler(Exception)
    async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
        error = SubLLMError(
            str(exc) or "Internal server error",
            code="internal_error",
            error_type="server_error",
            status_code=500,
        )
        return JSONResponse(status_code=error.status_code, content=error.to_response())

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            body = await request.json()
        except Exception as exc:
            raise MalformedRequestError("Request body must be valid JSON") from exc

        if not isinstance(body, dict):
            raise MalformedRequestError("Request body must be a JSON object")

        completion_request = CompletionRequest.from_mapping(body)

        if completion_request.stream:
            result = await router.completion(
                model=completion_request.model,
                messages=completion_request.provider_messages,
                stream=True,
                system_prompt=completion_request.effective_system_prompt,
                max_tokens=completion_request.max_tokens,
                temperature=completion_request.temperature,
            )

            async def event_stream():
                async for chunk in result:
                    data = json.dumps(chunk.to_dict())
                    yield f"data: {data}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(
                event_stream(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        else:
            result = await router.completion(
                model=completion_request.model,
                messages=completion_request.provider_messages,
                system_prompt=completion_request.effective_system_prompt,
                max_tokens=completion_request.max_tokens,
                temperature=completion_request.temperature,
            )
            return JSONResponse(result.to_dict())

    @app.get("/health")
    async def health():
        statuses = await router.check_auth()
        return {
            "status": "ok",
            "providers": [
                {"name": s.provider, "authenticated": s.authenticated, "method": s.method}
                for s in statuses
            ],
        }

    return app
