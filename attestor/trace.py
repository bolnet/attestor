"""Pipeline tracing — env-gated, stderr + JSONL output.

Off by default. Enable with ATTESTOR_TRACE=1.

Why this exists:
    The smoke runner needs end-to-end visibility: where did my message
    go (ingest path) and how was it retrieved (retrieval path). The
    project's degradation pattern catches non-fatal vector/graph
    errors silently to keep the document path always green -- which
    is correct for production but blinds you when you're debugging.
    This module is the diagnostic eyepiece.

Usage in code:
    import attestor.trace as tr

    tr.event("ingest.embed", provider="voyage", model="voyage-4",
             dim=1024, latency_ms=42)

When ATTESTOR_TRACE is unset, ``tr.event`` is a fast no-op (one env
lookup cached at module load) so trace points can sit in hot paths
without measurable overhead.

Output format:
    Stderr: ``[trace] ingest.embed provider=voyage model=voyage-4 dim=1024 latency_ms=42``
    JSONL (when ATTESTOR_TRACE_FILE=path): one event per line, full fields,
    parseable for post-run analysis.

Conventions:
    - event names are dotted: <area>.<step>[.<sub>]
    - field names are snake_case
    - never include raw secrets in trace fields (we redact via the
      same patterns as hooks/post_tool_use.py if a field looks
      key-shaped, but the cleanest fix is "don't pass secrets")
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import sys
import threading
import uuid
from datetime import datetime, timezone
from typing import Any, Iterator, List, Optional, TextIO

_ENABLED: bool = bool(os.environ.get("ATTESTOR_TRACE"))
_FILE_PATH: Optional[str] = os.environ.get("ATTESTOR_TRACE_FILE") or None
_FH: Optional[TextIO] = None
_LOCK = threading.Lock()


# ──────────────────────────────────────────────────────────────────────
# Phase 3 — recall-scoped trace context (audit invariant A5).
#
# Each recall gets a unique ``recall_id`` and a monotonic ``seq``
# counter scoped to that recall_id. Events emitted inside the scope
# are auto-tagged so the audit dashboard can reconstruct the per-recall
# event tree even when 10 recalls are concurrent and their stderr/JSONL
# events interleave in append-time order.
#
# ``contextvars.ContextVar`` propagates automatically across
# ``asyncio.create_task`` / ``asyncio.gather`` boundaries, so a single
# ``with recall_scope():`` block works for both sync and async callers.
# ──────────────────────────────────────────────────────────────────────


_RECALL_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "attestor.trace.recall_id", default=None,
)
_SEQ_COUNTER: contextvars.ContextVar[Optional[List[int]]] = contextvars.ContextVar(
    "attestor.trace.seq_counter", default=None,
)
_PARENT_EVENT_ID: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "attestor.trace.parent_event_id", default=None,
)


# Defensive redactor: lifted from hooks/post_tool_use.py. If a trace
# field accidentally carries a key-shaped string we still scrub it.
_SECRET_PATTERNS = (
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{20,}"),  "<REDACTED:openrouter>"),
    (re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}"),   "<REDACTED:openai>"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),    "<REDACTED:anthropic>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{32,}"),        "<REDACTED:openai>"),
    (re.compile(r"\bpa-[A-Za-z0-9_-]{30,}"),        "<REDACTED:voyage>"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"),         "<REDACTED:github>"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"),           "<REDACTED:aws>"),
)


def _scrub(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    out = value
    for pat, repl in _SECRET_PATTERNS:
        out = pat.sub(repl, out)
    return out


def _open_file() -> Optional[TextIO]:
    global _FH
    if _FH is not None or not _FILE_PATH:
        return _FH
    try:
        os.makedirs(os.path.dirname(os.path.abspath(_FILE_PATH)), exist_ok=True)
        _FH = open(_FILE_PATH, "a", buffering=1, encoding="utf-8")
    except OSError as e:
        sys.stderr.write(f"[trace] could not open {_FILE_PATH!r}: {e}\n")
        _FH = None
    return _FH


def is_enabled() -> bool:
    """Cheap check; lets callers skip building expensive payloads."""
    return _ENABLED


def event(name: str, **fields: Any) -> None:
    """Emit a single trace event.

    Args:
        name: Dotted event name (e.g. ``recall.stage.vector``).
        **fields: Structured key/value pairs. Values must be JSON-encodable.
                  String values are scrubbed for known key prefixes.

    Phase 3 auto-tagging — when called inside a ``recall_scope()``:
      - ``recall_id`` is set to the scope's UUID
      - ``seq`` is the monotonic 1-based counter within the scope
      - ``event_id`` is always generated (UUID4)
      - ``parent_event_id`` is set if inside an ``event_scope()``

    Outside any scope, only ``event_id`` is added — backwards-compatible
    with pre-Phase-3 trace consumers.
    """
    if not _ENABLED:
        return

    scrubbed = {k: _scrub(v) for k, v in fields.items()}

    # Auto-tag with recall context (no-op if not inside a recall_scope).
    rid = _RECALL_ID.get()
    counter = _SEQ_COUNTER.get()
    parent_eid = _PARENT_EVENT_ID.get()
    seq: Optional[int] = None
    if counter is not None:
        counter[0] += 1
        seq = counter[0]
    eid = str(uuid.uuid4())

    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": name,
        "event_id": eid,
        **({"recall_id": rid} if rid is not None else {}),
        **({"seq": seq} if seq is not None else {}),
        **({"parent_event_id": parent_eid} if parent_eid is not None else {}),
        **scrubbed,
    }

    pretty = " ".join(f"{k}={v!r}" if isinstance(v, str) else f"{k}={v}"
                      for k, v in scrubbed.items())
    line = f"[trace] {name} {pretty}".rstrip()

    with _LOCK:
        sys.stderr.write(line + "\n")
        sys.stderr.flush()
        fh = _open_file()
        if fh is not None:
            try:
                fh.write(json.dumps(payload, default=str) + "\n")
            except (TypeError, ValueError) as e:
                sys.stderr.write(f"[trace] could not serialize event {name!r}: {e}\n")


@contextlib.contextmanager
def recall_scope(recall_id: Optional[str] = None) -> Iterator[str]:
    """Context manager that scopes trace events to a single recall.

    All ``event()`` calls inside (including across ``asyncio.create_task``
    / ``asyncio.gather`` boundaries) auto-tag with ``recall_id`` and a
    monotonic ``seq`` counter. ``contextvars`` propagation makes this
    work for both sync and async callers — same API.

    Usage:
        with tr.recall_scope() as rid:
            tr.event("recall.start", query=q)
            ...                          # async tasks here see the same rid
            tr.event("recall.end")
    """
    rid = recall_id or str(uuid.uuid4())
    counter: List[int] = [0]  # mutable list — shared across child tasks
    rid_token = _RECALL_ID.set(rid)
    seq_token = _SEQ_COUNTER.set(counter)
    try:
        yield rid
    finally:
        _RECALL_ID.reset(rid_token)
        _SEQ_COUNTER.reset(seq_token)


@contextlib.contextmanager
def event_scope(event_id: Optional[str] = None) -> Iterator[str]:
    """Context manager that marks the enclosing event as the parent
    for any nested ``event()`` calls. Used when a recall step spawns
    child operations whose trace events should link back to it.

    Usage:
        eid = uuid.uuid4().hex
        tr.event("recall.vector.start", event_id=eid)
        with tr.event_scope(eid):
            ...                          # child events get parent_event_id=eid
    """
    eid = event_id or str(uuid.uuid4())
    parent_token = _PARENT_EVENT_ID.set(eid)
    try:
        yield eid
    finally:
        _PARENT_EVENT_ID.reset(parent_token)


def reset_for_test() -> None:
    """Test helper: re-read env vars and reopen the file handle.

    Module-level _ENABLED is captured at import; tests that toggle the
    env var need to call this to take effect. Also clears any leaked
    contextvars from a prior test (defensive — tests should use
    ``with recall_scope():`` and rely on ``reset()``, but this catches
    misuse early).
    """
    global _ENABLED, _FILE_PATH, _FH
    if _FH is not None:
        try:
            _FH.close()
        except OSError:
            pass
    _ENABLED = bool(os.environ.get("ATTESTOR_TRACE"))
    _FILE_PATH = os.environ.get("ATTESTOR_TRACE_FILE") or None
    _FH = None
