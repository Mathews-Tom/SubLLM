"""Structured error taxonomy for SubLLM request handling and provider failures."""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError


class SubLLMError(Exception):
    """Base exception with OpenAI-style error serialization."""

    def __init__(
        self,
        message: str,
        *,
        code: str,
        error_type: str = "invalid_request_error",
        status_code: int = 400,
        param: str | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.error_type = error_type
        self.status_code = status_code
        self.param = param

    def to_response(self) -> dict[str, Any]:
        return {
            "error": {
                "message": self.message,
                "type": self.error_type,
                "param": self.param,
                "code": self.code,
            }
        }


class MalformedRequestError(SubLLMError):
    """Raised when request payload validation fails."""

    def __init__(self, message: str, *, param: str | None = None) -> None:
        super().__init__(
            message,
            code="malformed_request",
            error_type="invalid_request_error",
            status_code=400,
            param=param,
        )

    @classmethod
    def from_validation_error(cls, exc: Exception) -> MalformedRequestError:
        if isinstance(exc, ValidationError):
            first_error = exc.errors()[0]
            location = ".".join(str(part) for part in first_error.get("loc", ()))
            message = first_error.get("msg", "Invalid request payload")
            return cls(message, param=location or None)
        return cls(str(exc))


class UnsupportedFeatureError(SubLLMError):
    """Raised when the caller uses an unsupported API field or capability."""

    def __init__(self, *, field: str, message: str) -> None:
        super().__init__(
            message,
            code="unsupported_feature",
            error_type="invalid_request_error",
            status_code=400,
            param=field,
        )
