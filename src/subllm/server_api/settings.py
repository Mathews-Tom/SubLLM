"""Server runtime settings and safety guards."""

from __future__ import annotations

import os
from dataclasses import dataclass


LOCAL_HOSTS = {"127.0.0.1", "::1", "localhost"}


@dataclass(frozen=True)
class ServerSettings:
    auth_token: str | None = None
    max_request_bytes: int = 1_048_576
    request_timeout_seconds: float = 60.0
    rate_limit_per_minute: int = 120

    @classmethod
    def from_inputs(
        cls,
        *,
        auth_token: str | None = None,
        max_request_bytes: int | None = None,
        request_timeout_seconds: float | None = None,
        rate_limit_per_minute: int | None = None,
    ) -> ServerSettings:
        return cls(
            auth_token=auth_token or os.getenv("SUBLLM_SERVER_AUTH_TOKEN"),
            max_request_bytes=max_request_bytes
            or int(os.getenv("SUBLLM_SERVER_MAX_REQUEST_BYTES", "1048576")),
            request_timeout_seconds=request_timeout_seconds
            or float(os.getenv("SUBLLM_SERVER_REQUEST_TIMEOUT_SECONDS", "60")),
            rate_limit_per_minute=rate_limit_per_minute
            or int(os.getenv("SUBLLM_SERVER_RATE_LIMIT_PER_MINUTE", "120")),
        )


def requires_auth_for_host(host: str, settings: ServerSettings) -> bool:
    return host not in LOCAL_HOSTS and not settings.auth_token
