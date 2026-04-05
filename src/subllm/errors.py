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


class UnknownModelError(SubLLMError):
    """Raised when a request references an unknown or ambiguous model."""

    def __init__(
        self, *, model: str, supported_models: list[str], detail: str | None = None
    ) -> None:
        supported = ", ".join(supported_models)
        message = detail or (f"Unsupported model '{model}'. Use one of: {supported}")
        super().__init__(
            message,
            code="model_not_found",
            error_type="invalid_request_error",
            status_code=400,
            param="model",
        )


class ProviderFailureError(SubLLMError):
    """Raised when a provider process or SDK session fails."""

    def __init__(self, *, provider: str, message: str) -> None:
        super().__init__(
            message,
            code="provider_failure",
            error_type="api_error",
            status_code=502,
            param=provider,
        )


class ProviderTimeoutError(SubLLMError):
    """Raised when a provider call exceeds the configured timeout."""

    def __init__(self, *, provider: str, operation: str, timeout_seconds: float) -> None:
        super().__init__(
            f"{provider} {operation} timed out after {timeout_seconds:.1f}s",
            code="provider_timeout",
            error_type="api_error",
            status_code=504,
            param=provider,
        )


class AuthenticationError(SubLLMError):
    """Raised when server auth requirements are not met."""

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(
            message,
            code="authentication_error",
            error_type="authentication_error",
            status_code=401,
        )


class RequestTooLargeError(SubLLMError):
    """Raised when an HTTP request body exceeds the configured limit."""

    def __init__(self, *, max_bytes: int) -> None:
        super().__init__(
            f"Request body exceeds the maximum allowed size of {max_bytes} bytes",
            code="request_too_large",
            error_type="invalid_request_error",
            status_code=413,
        )


class RateLimitExceededError(SubLLMError):
    """Raised when a client exceeds the configured request rate."""

    def __init__(self, *, limit_per_minute: int) -> None:
        super().__init__(
            f"Rate limit exceeded: {limit_per_minute} requests per minute",
            code="rate_limit_exceeded",
            error_type="rate_limit_error",
            status_code=429,
        )


class RequestTimeoutError(SubLLMError):
    """Raised when HTTP request handling exceeds the configured timeout."""

    def __init__(self, *, timeout_seconds: float) -> None:
        super().__init__(
            f"Request handling timed out after {timeout_seconds:.1f}s",
            code="request_timeout",
            error_type="api_error",
            status_code=504,
        )
