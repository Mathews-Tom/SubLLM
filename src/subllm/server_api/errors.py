"""FastAPI exception handlers for the HTTP boundary."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import cast

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from subllm.errors import MalformedRequestError, SubLLMError
from subllm.server_api.models import ErrorBody, ErrorResponse

RequestHandler = Callable[[Request, Exception], JSONResponse | Awaitable[JSONResponse]]


def build_error_response(exc: SubLLMError) -> ErrorResponse:
    return ErrorResponse(
        error=ErrorBody(
            message=exc.message,
            type=exc.error_type,
            param=exc.param,
            code=exc.code,
        )
    )


async def handle_subllm_error(_request: Request, exc: SubLLMError) -> JSONResponse:
    response = build_error_response(exc)
    return JSONResponse(status_code=exc.status_code, content=response.model_dump())


async def handle_request_validation_error(
    _request: Request,
    exc: RequestValidationError,
) -> JSONResponse:
    error = MalformedRequestError.from_validation_error(exc)
    response = build_error_response(error)
    return JSONResponse(status_code=error.status_code, content=response.model_dump())


async def handle_unexpected_error(_request: Request, exc: Exception) -> JSONResponse:
    error = SubLLMError(
        str(exc) or "Internal server error",
        code="internal_error",
        error_type="server_error",
        status_code=500,
    )
    response = build_error_response(error)
    return JSONResponse(status_code=error.status_code, content=response.model_dump())


def install_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(SubLLMError, cast(RequestHandler, handle_subllm_error))
    app.add_exception_handler(
        RequestValidationError,
        cast(RequestHandler, handle_request_validation_error),
    )
    app.add_exception_handler(Exception, handle_unexpected_error)
