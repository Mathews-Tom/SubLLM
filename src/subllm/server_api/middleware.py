"""FastAPI middleware for auth, request controls, tracing, and correlation."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from opentelemetry.trace.status import Status, StatusCode
from starlette.datastructures import Headers, MutableHeaders
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
from subllm.telemetry import (
    attach_request_context,
    bind_request_context,
    get_tracer,
    make_request_context,
    mark_span_failure,
    mark_span_success,
)

logger = logging.getLogger("subllm.server")


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


def _authorization_token(headers: Headers) -> str | None:
    authorization = headers.get("authorization")
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _request_context_from_headers(headers: Headers) -> tuple[str, str]:
    request_id = headers.get("x-request-id")
    correlation_id = headers.get("x-correlation-id")
    context = make_request_context(request_id=request_id, correlation_id=correlation_id)
    return context.request_id, context.correlation_id


def _error_response(exc: SubLLMError) -> JSONResponse:
    payload = build_error_response(exc)
    return JSONResponse(status_code=exc.status_code, content=payload.model_dump())


def _log_request(
    *,
    event: str,
    level: int,
    request_id: str,
    correlation_id: str,
    path: str,
    method: str,
    status_code: int,
    duration_ms: float,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "request_id": request_id,
        "correlation_id": correlation_id,
        "method": method,
        "path": path,
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
    }
    if extra_fields:
        payload.update(extra_fields)
    logger.log(level, "%s %s", event, json.dumps(payload, sort_keys=True))


class ServerBoundaryMiddleware:
    def __init__(
        self,
        app: ASGIApp,
        *,
        settings: ServerSettings,
        rate_limiter: RateLimiter,
    ) -> None:
        self.app = app
        self._settings = settings
        self._rate_limiter = rate_limiter

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id, correlation_id = _request_context_from_headers(headers)
        scope.setdefault("state", {})
        scope["state"]["request_id"] = request_id
        scope["state"]["correlation_id"] = correlation_id

        path = scope.get("path", "")
        method = scope.get("method", "")
        client = scope.get("client")
        client_host = client[0] if client is not None else "unknown"
        status_code = 500
        total_bytes = 0
        started = time.monotonic()

        async def guarded_receive() -> Message:
            nonlocal total_bytes
            message = await receive()
            if message["type"] == "http.request":
                total_bytes += len(message.get("body", b""))
                if total_bytes > self._settings.max_request_bytes:
                    raise RequestTooLargeError(max_bytes=self._settings.max_request_bytes)
            return message

        async def guarded_send(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
                mutable_headers = MutableHeaders(raw=message["headers"])
                mutable_headers["x-request-id"] = request_id
                mutable_headers["x-correlation-id"] = correlation_id
            await send(message)

        with (
            bind_request_context(
                make_request_context(request_id=request_id, correlation_id=correlation_id)
            ),
            get_tracer("subllm.server").start_as_current_span("subllm.http.request") as span,
        ):
            attach_request_context(span)
            span.set_attribute("http.request.method", method)
            span.set_attribute("url.path", path)
            span.set_attribute("server.address", client_host)

            try:
                if self._settings.auth_token is not None:
                    token = _authorization_token(headers)
                    if token != self._settings.auth_token:
                        raise AuthenticationError("Invalid or missing bearer token")

                await self._rate_limiter.check(client_host)
                await asyncio.wait_for(
                    self.app(scope, guarded_receive, guarded_send),
                    timeout=self._settings.request_timeout_seconds,
                )
                mark_span_success(span)
            except asyncio.TimeoutError:
                error = RequestTimeoutError(timeout_seconds=self._settings.request_timeout_seconds)
                mark_span_failure(span, error)
                span.set_attribute("http.response.status_code", error.status_code)
                response = _error_response(error)
                await response(scope, receive, guarded_send)
            except SubLLMError as exc:
                mark_span_failure(span, exc)
                span.set_attribute("http.response.status_code", exc.status_code)
                response = _error_response(exc)
                await response(scope, receive, guarded_send)
            except Exception as exc:
                mark_span_failure(span, exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
                raise
            finally:
                duration_ms = (time.monotonic() - started) * 1000.0
                span.set_attribute("http.response.status_code", status_code)
                span.set_attribute("subllm.request.duration_ms", duration_ms)
                _log_request(
                    event="request.completed",
                    level=logging.INFO,
                    request_id=request_id,
                    correlation_id=correlation_id,
                    path=path,
                    method=method,
                    status_code=status_code,
                    duration_ms=duration_ms,
                )


def install_server_controls(app: FastAPI, settings: ServerSettings) -> None:
    rate_limiter = RateLimiter(settings.rate_limit_per_minute)
    app.add_middleware(
        ServerBoundaryMiddleware,
        settings=settings,
        rate_limiter=rate_limiter,
    )
