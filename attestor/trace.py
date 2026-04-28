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

import json
import os
import re
import sys
import threading
from datetime import datetime, timezone
from typing import Any, Optional, TextIO

_ENABLED: bool = bool(os.environ.get("ATTESTOR_TRACE"))
_FILE_PATH: Optional[str] = os.environ.get("ATTESTOR_TRACE_FILE") or None
_FH: Optional[TextIO] = None
_LOCK = threading.Lock()


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
    """
    if not _ENABLED:
        return

    scrubbed = {k: _scrub(v) for k, v in fields.items()}
    payload = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "event": name,
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


def reset_for_test() -> None:
    """Test helper: re-read env vars and reopen the file handle.

    Module-level _ENABLED is captured at import; tests that toggle the
    env var need to call this to take effect.
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
