# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Unified observability facade — recall trace + OTel + LLM cost trace.

Three modules previously formed the observability surface:

  - ``attestor.trace``      JSONL recall-trace events + nested scopes
  - ``attestor.otel``       OpenTelemetry spans (request-scoped)
  - ``attestor.llm_trace``  Per-LLM-call cost + latency capture

They each have distinct responsibilities, but callers regularly need a
mix of all three (e.g. "open a recall scope, emit an OTel span event,
log the embedded LLM call"). The previous setup forced every call site
to import three separate modules. This facade re-exports the canonical
names from one place; the underlying modules stay where they are so
existing imports keep working.

Use ``from attestor.observability import event, recall_scope, ...`` in
new code; the older ``from attestor import trace as _tr; _tr.event(...)``
form remains supported.
"""

from __future__ import annotations

# Recall-trace JSONL surface.
from attestor.trace import (
    event,
    event_scope,
    is_enabled,
    recall_scope,
    reset_for_test,
)

# OpenTelemetry-mirror surface.
from attestor import otel as _otel
otel_add_event = _otel.add_event
otel_current_span_id = _otel.current_span_id
otel_is_enabled = _otel.is_enabled
otel_start_span = _otel.start_span

# LLM cost / latency surface.
from attestor.config import chat_kwargs_for_role
from attestor.llm_trace import (
    emit_chat_trace,
    get_client_for_model,
    make_async_client,
    make_client,
    traced_create,
)

__all__ = [
    "chat_kwargs_for_role",
    "emit_chat_trace",
    "event",
    "event_scope",
    "get_client_for_model",
    "is_enabled",
    "make_async_client",
    "make_client",
    "otel_add_event",
    "otel_current_span_id",
    "otel_is_enabled",
    "otel_start_span",
    "recall_scope",
    "reset_for_test",
    "traced_create",
]
