"""OpenTelemetry helpers and request correlation context."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence
from uuid import uuid4

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    SimpleSpanProcessor,
    SpanExportResult,
    SpanExporter,
)
from opentelemetry.trace import Span
from opentelemetry.trace.status import Status, StatusCode
from opentelemetry.util.types import Attributes

_request_id_var: ContextVar[str | None] = ContextVar("subllm_request_id", default=None)
_correlation_id_var: ContextVar[str | None] = ContextVar("subllm_correlation_id", default=None)


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    correlation_id: str


@dataclass(frozen=True)
class TelemetryConfig:
    service_name: str
    export_path: str | None = None


class JsonLineSpanExporter(SpanExporter):
    """Persist spans as JSON lines for local inspection and testing."""

    def __init__(self, export_path: str) -> None:
        self._path = Path(export_path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        lines = [json.dumps(_serialize_span(span), sort_keys=True) for span in spans]
        with self._lock:
            with self._path.open("a", encoding="utf-8") as handle:
                for line in lines:
                    handle.write(f"{line}\n")
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        return None


@dataclass
class _TelemetryRuntime:
    config: TelemetryConfig
    provider: TracerProvider
    processor: SimpleSpanProcessor
    exporter: JsonLineSpanExporter

    def shutdown(self) -> None:
        self.provider.shutdown()
        self.exporter.shutdown()


_runtime: _TelemetryRuntime | None = None


def configure_telemetry(config: TelemetryConfig | None) -> None:
    """Configure tracing for local export or disable it when no path is set."""

    global _runtime
    if config is None or config.export_path is None:
        if _runtime is not None:
            _runtime.shutdown()
            _runtime = None
        return

    if _runtime is not None and _runtime.config == config:
        return

    if _runtime is not None:
        _runtime.shutdown()

    provider = TracerProvider(
        resource=Resource.create({"service.name": config.service_name, "service.version": "0.5.0"})
    )
    exporter = JsonLineSpanExporter(config.export_path)
    processor = SimpleSpanProcessor(exporter)
    provider.add_span_processor(processor)
    _runtime = _TelemetryRuntime(
        config=config,
        provider=provider,
        processor=processor,
        exporter=exporter,
    )


def shutdown_telemetry() -> None:
    global _runtime
    if _runtime is not None:
        _runtime.shutdown()
        _runtime = None


def get_tracer(name: str) -> trace.Tracer:
    if _runtime is None:
        return trace.get_tracer(name)
    return _runtime.provider.get_tracer(name)


def make_request_context(
    *,
    request_id: str | None = None,
    correlation_id: str | None = None,
) -> RequestContext:
    resolved_request_id = request_id or str(uuid4())
    return RequestContext(
        request_id=resolved_request_id,
        correlation_id=correlation_id or resolved_request_id,
    )


@contextmanager
def bind_request_context(context: RequestContext) -> Iterator[None]:
    request_token: Token[str | None] = _request_id_var.set(context.request_id)
    correlation_token: Token[str | None] = _correlation_id_var.set(context.correlation_id)
    try:
        yield
    finally:
        _request_id_var.reset(request_token)
        _correlation_id_var.reset(correlation_token)


def current_request_context() -> RequestContext | None:
    request_id = _request_id_var.get()
    correlation_id = _correlation_id_var.get()
    if request_id is None or correlation_id is None:
        return None
    return RequestContext(request_id=request_id, correlation_id=correlation_id)


def response_context_headers() -> dict[str, str]:
    context = current_request_context()
    if context is None:
        return {}
    return {
        "x-request-id": context.request_id,
        "x-correlation-id": context.correlation_id,
    }


def attach_request_context(span: Span) -> None:
    context = current_request_context()
    if context is None:
        return
    span.set_attribute("subllm.request.id", context.request_id)
    span.set_attribute("subllm.correlation.id", context.correlation_id)


def mark_span_failure(span: Span, exc: Exception) -> None:
    span.record_exception(exc)
    description = str(exc) or exc.__class__.__name__
    span.set_status(Status(StatusCode.ERROR, description))
    span.set_attribute("error.type", exc.__class__.__name__)
    if hasattr(exc, "code"):
        span.set_attribute("subllm.error.code", getattr(exc, "code"))


def mark_span_success(span: Span) -> None:
    span.set_status(Status(StatusCode.OK))


def _serialize_span(span: Any) -> dict[str, Any]:
    return {
        "name": span.name,
        "trace_id": f"{span.context.trace_id:032x}",
        "span_id": f"{span.context.span_id:016x}",
        "parent_span_id": (f"{span.parent.span_id:016x}" if span.parent is not None else None),
        "start_time_unix_nano": span.start_time,
        "end_time_unix_nano": span.end_time,
        "attributes": _serialize_mapping(span.attributes),
        "events": [
            {
                "name": event.name,
                "timestamp_unix_nano": event.timestamp,
                "attributes": _serialize_mapping(event.attributes),
            }
            for event in span.events
        ],
        "status": {
            "status_code": span.status.status_code.name,
            "description": span.status.description,
        },
        "resource": _serialize_mapping(span.resource.attributes),
    }


def _serialize_mapping(attributes: Attributes | None) -> dict[str, Any]:
    if not attributes:
        return {}
    return {key: _serialize_value(value) for key, value in attributes.items()}


def _serialize_value(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, (list, tuple)):
        return [_serialize_value(item) for item in value]
    return str(value)
