"""FastAPI middleware for auth and request controls."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from subllm.errors import (
    AuthenticationError,
    RateLimitExceededError,
    RequestTimeoutError,
    RequestTooLargeError,
    SubLLMError,
)
from subllm.server_api.errors import build_error_response
from subllm.server_api.settings import ServerSettings


class RateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self._limit_per_minute = limit_per_minute
        self._requests: dict[str, deque[float]] = {}
        self._lock = asyncio.Lock()

    async def check(self, client_key: str) -> None:
        now = time.monotonic()
        cutoff = now - 60.0
        async with self._lock:
            bucket = self._requests.setdefault(client_key, deque())
            while bucket and bucket[0] < cutoff:
                bucket.popleft()
            if len(bucket) >= self._limit_per_minute:
                raise RateLimitExceededError(limit_per_minute=self._limit_per_minute)
            bucket.append(now)


def _authorization_token(request: Request) -> str | None:
    authorization = request.headers.get("authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


class RequestBodyLimitMiddleware:
    def __init__(self, app: ASGIApp, *, max_request_bytes: int) -> None:
        self.app = app
        self._max_request_bytes = max_request_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        total_bytes = 0

        async def guarded_receive() -> Message:
            nonlocal total_bytes
            message = await receive()
            if message["type"] == "http.request":
                total_bytes += len(message.get("body", b""))
                if total_bytes > self._max_request_bytes:
                    raise RequestTooLargeError(max_bytes=self._max_request_bytes)
            return message

        try:
            await self.app(scope, guarded_receive, send)
        except SubLLMError as exc:
            payload = build_error_response(exc)
            response = JSONResponse(status_code=exc.status_code, content=payload.model_dump())
            await response(scope, receive, send)


def install_server_controls(app: FastAPI, settings: ServerSettings) -> None:
    rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_request_bytes=settings.max_request_bytes,
    )

    @app.middleware("http")
    async def enforce_server_controls(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        def error_response(exc: SubLLMError) -> JSONResponse:
            payload = build_error_response(exc)
            return JSONResponse(status_code=exc.status_code, content=payload.model_dump())

        try:
            if settings.auth_token is not None:
                token = _authorization_token(request)
                if token != settings.auth_token:
                    raise AuthenticationError("Invalid or missing bearer token")

            client_host = request.client.host if request.client is not None else "unknown"
            await rate_limiter.check(client_host)

            return await asyncio.wait_for(
                call_next(request),
                timeout=settings.request_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return error_response(
                RequestTimeoutError(timeout_seconds=settings.request_timeout_seconds)
            )
        except SubLLMError as exc:
            return error_response(exc)
