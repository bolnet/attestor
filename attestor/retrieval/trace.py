"""Per-query retrieval trace logger.

Writes one JSONL line per recall() call to ``logs/attestor_trace.jsonl``.
Disabled when env ``ATTESTOR_TRACE=0``.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any


_DEFAULT_PATH = "logs/attestor_trace.jsonl"
_lock = threading.Lock()
_fh = None
_resolved_path: Path | None = None


def _enabled() -> bool:
    return os.environ.get("ATTESTOR_TRACE", "1") not in {"0", "false", "False"}


def _get_fh():
    global _fh, _resolved_path
    if _fh is not None:
        return _fh
    path_str = os.environ.get("ATTESTOR_TRACE_PATH", _DEFAULT_PATH)
    path = Path(path_str)
    path.parent.mkdir(parents=True, exist_ok=True)
    _fh = path.open("a", encoding="utf-8")
    _resolved_path = path
    return _fh


def write(record: dict[str, Any]) -> None:
    """Append one trace record as a JSON line."""
    if not _enabled():
        return
    record = {"ts": time.time(), **record}
    line = json.dumps(record, default=str, ensure_ascii=False)
    with _lock:
        fh = _get_fh()
        fh.write(line + "\n")
        fh.flush()


def trace_path() -> Path | None:
    return _resolved_path
