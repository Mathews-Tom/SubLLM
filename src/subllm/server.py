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

from __future__ import annotations

import json
import time
from typing import TYPE_CHECKING

from subllm.router import Router

if TYPE_CHECKING:
    from fastapi import FastAPI


def create_app() -> FastAPI:
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

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        body = await request.json()

        model = body.get("model", "claude-code/sonnet-4-5")
        messages = body.get("messages", [])
        stream = body.get("stream", False)
        max_tokens = body.get("max_tokens")
        temperature = body.get("temperature")

        # Extract system prompt from messages
        system_prompt = None
        filtered_messages = []
        for msg in messages:
            if msg.get("role") == "system":
                system_prompt = msg.get("content", "")
            else:
                filtered_messages.append(msg)

        if stream:
            result = await router.completion(
                model=model, messages=filtered_messages, stream=True,
                system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
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
                model=model, messages=filtered_messages,
                system_prompt=system_prompt, max_tokens=max_tokens, temperature=temperature,
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
