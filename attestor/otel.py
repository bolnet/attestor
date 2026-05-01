"""OpenTelemetry bridge — opt-in spans for the recall pipeline.

Off by default. Enable with one of:

  OTEL_EXPORTER=console               # write spans to stderr (dev)
  OTEL_EXPORTER=otlp                  # write to OTEL_EXPORTER_OTLP_ENDPOINT
                                      # (default http://localhost:4318)

When enabled:
  - Every ``trace.recall_scope()`` opens an OTel span named "recall".
  - Every ``trace.event(name, **fields)`` adds an OTel event to the
    active span with ``recall_id`` / ``seq`` / ``parent_event_id`` as
    attributes (mapped from our existing contextvars).
  - The OTel span_id is added back into the JSONL trace event so
    operators can correlate JSONL ↔ OTel by span_id.

When disabled (default), this module is a no-op — no overhead on the
hot path beyond a single env-var lookup at import.

Audit-preservation:
  OTel spans CARRY the audit metadata (recall_id, seq) but are NOT the
  source of truth — JSONL remains canonical. OTel is one consumer of
  the audit trail, not the audit trail itself. This keeps OTel
  optional and the audit replay path unchanged.
"""

from __future__ import annotations

import os
from typing import Any


_ENABLED: bool = bool(os.environ.get("OTEL_EXPORTER"))
_TRACER: Any = None  # opentelemetry.trace.Tracer when enabled


def is_enabled() -> bool:
    return _ENABLED


def _init_tracer() -> Any:
    """Initialize the OTel tracer based on env. Idempotent — safe to
    call multiple times. Returns the tracer or None when disabled.
    """
    global _TRACER
    if not _ENABLED or _TRACER is not None:
        return _TRACER

    try:
        from opentelemetry import trace as ot
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import (
            BatchSpanProcessor,
            ConsoleSpanExporter,
            SimpleSpanProcessor,
        )

        exporter_kind = os.environ.get("OTEL_EXPORTER", "").lower()
        service_name = os.environ.get("OTEL_SERVICE_NAME", "attestor")

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if exporter_kind == "console":
            exporter = ConsoleSpanExporter()
            processor = SimpleSpanProcessor(exporter)
        elif exporter_kind == "otlp":
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter,
            )
            endpoint = os.environ.get(
                "OTEL_EXPORTER_OTLP_ENDPOINT", "http://localhost:4318",
            )
            exporter = OTLPSpanExporter(endpoint=f"{endpoint}/v1/traces")
            processor = BatchSpanProcessor(exporter)
        else:
            return None

        provider.add_span_processor(processor)
        ot.set_tracer_provider(provider)
        _TRACER = ot.get_tracer("attestor")
        return _TRACER

    except Exception:  # noqa: BLE001 — never break the hot path on init failure
        return None


def reset_for_test() -> None:
    """Test helper: re-read env, drop cached tracer. Used when tests
    toggle OTEL_EXPORTER between tests."""
    global _ENABLED, _TRACER
    _ENABLED = bool(os.environ.get("OTEL_EXPORTER"))
    _TRACER = None


def start_span(name: str, **attributes: Any) -> Any:
    """Return a span context manager. No-op when OTel is disabled.

    Usage:
        with otel.start_span("recall", recall_id=rid):
            ...
    """
    if not _ENABLED:
        return _NoopSpan()
    tracer = _init_tracer()
    if tracer is None:
        return _NoopSpan()
    span_ctx = tracer.start_as_current_span(name)
    return _SpanWrapper(span_ctx, attributes)


def add_event(name: str, **attributes: Any) -> None:
    """Add an event to the currently active span. No-op when OTel is
    disabled or no span is active.
    """
    if not _ENABLED:
        return
    tracer = _init_tracer()
    if tracer is None:
        return
    try:
        from opentelemetry import trace as ot
        span = ot.get_current_span()
        if span is None:
            return
        # Coerce attributes to OTel-compatible types.
        attrs = {}
        for k, v in attributes.items():
            if isinstance(v, (str, bool, int, float)):
                attrs[k] = v
            elif v is None:
                continue
            else:
                attrs[k] = str(v)[:500]
        span.add_event(name, attributes=attrs)
    except Exception:  # noqa: BLE001
        pass


def current_span_id() -> str | None:
    """Return the current span's span_id as a hex string, or None when
    no span is active. Used by ``trace.event`` to add a ``span_id``
    field to the JSONL payload so operators can correlate the two
    streams."""
    if not _ENABLED:
        return None
    try:
        from opentelemetry import trace as ot
        span = ot.get_current_span()
        if span is None:
            return None
        ctx = span.get_span_context()
        if not ctx.is_valid:
            return None
        return f"{ctx.span_id:016x}"
    except Exception:  # noqa: BLE001
        return None


class _NoopSpan:
    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def set_attribute(self, *args: Any, **kwargs: Any) -> None:
        return None


class _SpanWrapper:
    """Bridge between our context manager API and OTel's start_as_current_span."""
    def __init__(self, span_ctx: Any, initial_attrs: dict) -> None:
        self._span_ctx = span_ctx
        self._initial_attrs = initial_attrs
        self._span: Any = None

    def __enter__(self) -> _SpanWrapper:
        self._span = self._span_ctx.__enter__()
        for k, v in self._initial_attrs.items():
            if v is None:
                continue
            if isinstance(v, (str, bool, int, float)):
                self._span.set_attribute(k, v)
            else:
                self._span.set_attribute(k, str(v)[:500])
        return self

    def __exit__(self, *exc: Any) -> None:
        self._span_ctx.__exit__(*exc)

    def set_attribute(self, key: str, value: Any) -> None:
        if self._span is not None:
            self._span.set_attribute(key, value)
