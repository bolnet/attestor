# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Read-only explorer for the Attestor JSONL trace log.

The explorer indexes events by ``recall_id`` (set by
``attestor.trace.recall_scope``) so each recall's full event tree can
be served to the audit dashboard for inspection.

Design constraints:
    * Read-only — never mutate the JSONL log.
    * Cheap on cold path — 100k events index in <500ms on a laptop.
    * Backed by an in-memory cache keyed on ``(path, mtime, size)`` so
      a second call against the same log is O(1).
    * Tolerant of malformed lines — a corrupt JSONL line skips, never
      crashes the pipeline.
"""

from __future__ import annotations

import json
import os
import threading
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_iso_or_none(value: str | None) -> datetime | None:
    """Parse an ISO-8601 string to ``datetime`` or return ``None``.

    Tolerates ``YYYY-MM-DD`` (date-only), ``YYYY-MM-DDTHH:MM:SS`` (no tz),
    full ISO with offsets, and the ``Z`` shorthand for UTC. Used by the
    audit explorer's ``since`` filter so callers can pass any reasonable
    ISO form without breaking the comparison.
    """
    if value is None:
        return None
    s = value.strip()
    if not s:
        return None
    # ``Z`` suffix is RFC 3339 shorthand for UTC; fromisoformat accepts
    # it on Python 3.11+ but normalising keeps older interpreters happy.
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        # Date-only fallback.
        try:
            return datetime.fromisoformat(s[:10])
        except ValueError:
            return None

# ──────────────────────────────────────────────────────────────────────
# Public dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class RecallSummary:
    """One row in the audit dashboard's recall table."""

    recall_id: str
    started_at: str
    completed_at: str | None
    user_id: str | None
    namespace: str | None
    query: str
    elapsed_ms: float
    result_count: int
    lanes: tuple[str, ...]
    cost_usd: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_id": self.recall_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "user_id": self.user_id,
            "namespace": self.namespace,
            "query": self.query,
            "elapsed_ms": self.elapsed_ms,
            "result_count": self.result_count,
            "lanes": list(self.lanes),
            "cost_usd": self.cost_usd,
        }


@dataclass(frozen=True)
class RecallDetail:
    """Full per-recall view: summary fields + the complete event tree.

    We keep RecallDetail flat (not inheriting from RecallSummary) so the
    dataclass stays a frozen value type — frozen-dataclass inheritance
    with extra fields is a known foot-gun.
    """

    recall_id: str
    started_at: str
    completed_at: str | None
    user_id: str | None
    namespace: str | None
    query: str
    elapsed_ms: float
    result_count: int
    lanes: tuple[str, ...]
    cost_usd: float | None
    events: tuple[dict[str, Any], ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_id": self.recall_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "user_id": self.user_id,
            "namespace": self.namespace,
            "query": self.query,
            "elapsed_ms": self.elapsed_ms,
            "result_count": self.result_count,
            "lanes": list(self.lanes),
            "cost_usd": self.cost_usd,
            "events": list(self.events),
        }


# ──────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────


_DEFAULT_LOG_PATH = "~/.attestor/trace.jsonl"


def default_log_path() -> Path:
    """Resolve the default trace log path.

    Resolution order:
        1. ``ATTESTOR_TRACE_FILE`` env var.
        2. ``~/.attestor/trace.jsonl`` (matches the docs in
           ``attestor.trace``).
    """
    env = os.environ.get("ATTESTOR_TRACE_FILE")
    if env:
        return Path(env).expanduser()
    return Path(_DEFAULT_LOG_PATH).expanduser()


# ──────────────────────────────────────────────────────────────────────
# Internal index dataclass — built once per (path, mtime, size).
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Index:
    """In-memory index built from a JSONL trace log."""

    # All events tagged with a recall_id, grouped + ordered by seq+ts.
    events_by_recall: dict[str, list[dict[str, Any]]]
    # recall_ids ordered newest-first by the recall.start ts.
    ordered_recall_ids: tuple[str, ...]
    # Per-recall summary cache.
    summaries: dict[str, RecallSummary]
    # memory_id → list of (recall_id, ts) pairs.
    memory_to_recalls: dict[str, list[tuple[str, str]]]
    # File fingerprint for cache invalidation.
    fingerprint: tuple[str, float, int] = field(default=("", 0.0, 0))


# ──────────────────────────────────────────────────────────────────────
# TraceExplorer — primary class
# ──────────────────────────────────────────────────────────────────────


class TraceExplorer:
    """Read-only query interface for a JSONL trace log.

    Construct once per log path; the index is lazily built on first
    query and reused until the file's mtime/size changes.
    """

    def __init__(self, log_path: Path | str) -> None:
        self.path = Path(log_path).expanduser()
        self._lock = threading.Lock()
        self._index: _Index | None = None

    # ── Public surface ────────────────────────────────────────────

    def list_recalls(
        self,
        *,
        user_id: str | None = None,
        namespace: str | None = None,
        since: str | None = None,
        limit: int = 100,
    ) -> list[RecallSummary]:
        idx = self._ensure_index()
        # Parse ``since`` once. Using a raw string compare against
        # ISO-8601 ``started_at`` only works if both sides have the
        # same shape (date-only vs date+time vs offset). Parsing here
        # makes the comparison robust regardless of the caller's
        # format choice.
        since_dt = _parse_iso_or_none(since)
        out: list[RecallSummary] = []
        for rid in idx.ordered_recall_ids:
            summary = idx.summaries.get(rid)
            if summary is None:
                continue
            if user_id is not None and summary.user_id != user_id:
                continue
            if namespace is not None and summary.namespace != namespace:
                continue
            if since_dt is not None:
                start_dt = _parse_iso_or_none(summary.started_at)
                if start_dt is not None and start_dt < since_dt:
                    continue
            out.append(summary)
            if len(out) >= limit:
                break
        return out

    def get_recall(self, recall_id: str) -> RecallDetail | None:
        idx = self._ensure_index()
        events = idx.events_by_recall.get(recall_id)
        if not events:
            return None
        summary = idx.summaries.get(recall_id)
        if summary is None:
            return None
        return RecallDetail(
            recall_id=summary.recall_id,
            started_at=summary.started_at,
            completed_at=summary.completed_at,
            user_id=summary.user_id,
            namespace=summary.namespace,
            query=summary.query,
            elapsed_ms=summary.elapsed_ms,
            result_count=summary.result_count,
            lanes=summary.lanes,
            cost_usd=summary.cost_usd,
            events=tuple(events),
        )

    def memory_access_timeline(self, memory_id: str) -> list[dict[str, Any]]:
        idx = self._ensure_index()
        rows = idx.memory_to_recalls.get(memory_id, [])
        out: list[dict[str, Any]] = []
        for recall_id, ts in rows:
            summary = idx.summaries.get(recall_id)
            if summary is None:
                continue
            out.append({
                "recall_id": recall_id,
                "ts": ts,
                "query": summary.query,
                "user_id": summary.user_id,
                "namespace": summary.namespace,
            })
        out.sort(key=lambda r: r["ts"])
        return out

    def user_activity(
        self,
        user_id: str,
        *,
        since: str | None = None,
    ) -> dict[str, Any]:
        idx = self._ensure_index()
        recall_count = 0
        memories_accessed_total = 0
        distinct_memories: set[str] = set()
        lane_usage: dict[str, int] = defaultdict(int)

        since_dt = _parse_iso_or_none(since)
        for rid in idx.ordered_recall_ids:
            summary = idx.summaries.get(rid)
            if summary is None or summary.user_id != user_id:
                continue
            if since_dt is not None:
                start_dt = _parse_iso_or_none(summary.started_at)
                if start_dt is not None and start_dt < since_dt:
                    continue
            recall_count += 1
            for lane in summary.lanes:
                lane_usage[lane] += 1
            # Collect distinct memory ids from the recall.complete event.
            events = idx.events_by_recall.get(rid, [])
            for ev in events:
                if ev.get("event") in ("recall.complete", "recall.done"):
                    for mid in _extract_result_ids(ev):
                        distinct_memories.add(mid)
                        memories_accessed_total += 1
                    break

        return {
            "user_id": user_id,
            "recall_count": recall_count,
            "memories_accessed_total": memories_accessed_total,
            "memories_accessed_distinct": len(distinct_memories),
            "lane_usage": dict(lane_usage),
        }

    # ── Cache management ──────────────────────────────────────────

    def reset_cache(self) -> None:
        with self._lock:
            self._index = None

    def _ensure_index(self) -> _Index:
        with self._lock:
            fp = _file_fingerprint(self.path)
            if self._index is None or self._index.fingerprint != fp:
                self._index = _build_index(self.path, fp)
            return self._index


# ──────────────────────────────────────────────────────────────────────
# Internals — index construction + helpers
# ──────────────────────────────────────────────────────────────────────


def _file_fingerprint(path: Path) -> tuple[str, float, int]:
    try:
        stat = path.stat()
    except FileNotFoundError:
        return (str(path), 0.0, 0)
    return (str(path), stat.st_mtime, stat.st_size)


def _build_index(path: Path, fingerprint: tuple[str, float, int]) -> _Index:
    events_by_recall: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if not path.exists():
        return _Index(
            events_by_recall={},
            ordered_recall_ids=(),
            summaries={},
            memory_to_recalls={},
            fingerprint=fingerprint,
        )

    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                # Tolerant of corrupt lines — never crash the audit path.
                continue
            rid = ev.get("recall_id")
            if not rid:
                continue
            events_by_recall[rid].append(ev)

    # Stable per-recall ordering by seq (then ts) so lane order is
    # preserved even if the JSONL was concatenated out of order.
    for events in events_by_recall.values():
        events.sort(key=lambda e: (e.get("seq", 0), e.get("ts", "")))

    summaries: dict[str, RecallSummary] = {}
    memory_to_recalls: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for rid, events in events_by_recall.items():
        summary = _summarize(rid, events)
        if summary is None:
            continue
        summaries[rid] = summary
        # Index returned memory ids for the access timeline.
        for ev in events:
            if ev.get("event") in ("recall.complete", "recall.done"):
                ts = ev.get("ts") or summary.started_at
                for mid in _extract_result_ids(ev):
                    memory_to_recalls[mid].append((rid, ts))
                break

    # Newest-first ordering by start ts.
    ordered = sorted(
        summaries.keys(),
        key=lambda r: summaries[r].started_at,
        reverse=True,
    )
    return _Index(
        events_by_recall=dict(events_by_recall),
        ordered_recall_ids=tuple(ordered),
        summaries=summaries,
        memory_to_recalls=dict(memory_to_recalls),
        fingerprint=fingerprint,
    )


_LANE_PREFIX = "recall.lane."
_RERANKER_EVENT = "recall.stage.reranker"


def _summarize(
    recall_id: str,
    events: list[dict[str, Any]],
) -> RecallSummary | None:
    if not events:
        return None
    start = next(
        (e for e in events if e.get("event") == "recall.start"),
        None,
    )
    complete = next(
        (e for e in events if e.get("event") in ("recall.complete", "recall.done")),
        None,
    )
    # If we have neither start nor complete, fall back to the first event.
    anchor = start or events[0]

    lanes: list[str] = []
    seen_lanes: set[str] = set()
    for ev in events:
        name = ev.get("event", "")
        lane = _lane_from_event(name)
        if lane and lane not in seen_lanes:
            seen_lanes.add(lane)
            lanes.append(lane)

    user_id = _first(events, "user_id")
    namespace = _first(events, "namespace")
    query = (start or {}).get("query") or _first(events, "query") or ""

    started_at = (start or anchor).get("ts", "")
    completed_at = complete.get("ts") if complete else None
    elapsed_ms = float(complete.get("elapsed_ms", 0.0)) if complete else 0.0
    result_count = int(complete.get("result_count", 0)) if complete else 0
    cost_usd_raw = (complete or {}).get("cost_usd")
    cost_usd = float(cost_usd_raw) if cost_usd_raw is not None else None

    return RecallSummary(
        recall_id=recall_id,
        started_at=started_at,
        completed_at=completed_at,
        user_id=user_id,
        namespace=namespace,
        query=query,
        elapsed_ms=elapsed_ms,
        result_count=result_count,
        lanes=tuple(lanes),
        cost_usd=cost_usd,
    )


def _lane_from_event(name: str) -> str | None:
    """Extract a lane label from an event name.

    Recognized lanes:
        - ``recall.lane.<lane>``      → ``<lane>`` (vector / bm25 / graph / reranker)
        - ``recall.stage.reranker``   → ``reranker``
    """
    if name.startswith(_LANE_PREFIX):
        return name[len(_LANE_PREFIX):]
    if name == _RERANKER_EVENT:
        return "reranker"
    return None


def _first(events: Iterable[dict[str, Any]], key: str) -> Any:
    for ev in events:
        val = ev.get(key)
        if val is not None:
            return val
    return None


def _extract_result_ids(complete_event: dict[str, Any]) -> list[str]:
    """Pull memory ids from a recall.complete (or recall.done) event.

    Two payload shapes are supported because the codebase emits both:
        * ``result_ids: ["m-1", "m-2"]`` — flat list (Phase 3 audit format).
        * ``final_ids: [{"id": "m-1", ...}, ...]`` — rich format used by
          ``orchestrator/postprocess.py``.
    """
    raw = complete_event.get("result_ids")
    if isinstance(raw, list):
        return [str(x) for x in raw if isinstance(x, (str, int))]
    rich = complete_event.get("final_ids")
    if isinstance(rich, list):
        out: list[str] = []
        for entry in rich:
            if isinstance(entry, dict) and entry.get("id") is not None:
                out.append(str(entry["id"]))
            elif isinstance(entry, str):
                out.append(entry)
        return out
    return []


# ──────────────────────────────────────────────────────────────────────
# Module-level convenience surface (uses default log path).
# ──────────────────────────────────────────────────────────────────────


_default_explorer: TraceExplorer | None = None
_default_lock = threading.Lock()


def _resolve_default() -> TraceExplorer:
    global _default_explorer
    with _default_lock:
        path = default_log_path()
        # Re-resolve when the configured path changes.
        if _default_explorer is None or _default_explorer.path != path:
            _default_explorer = TraceExplorer(path)
        return _default_explorer


def reset_cache() -> None:
    """Drop the module-level explorer cache. Used by tests + on config reload."""
    global _default_explorer
    with _default_lock:
        _default_explorer = None


def list_recalls(
    *,
    user_id: str | None = None,
    namespace: str | None = None,
    since: str | None = None,
    limit: int = 100,
) -> list[RecallSummary]:
    return _resolve_default().list_recalls(
        user_id=user_id, namespace=namespace, since=since, limit=limit,
    )


def get_recall(recall_id: str) -> RecallDetail | None:
    return _resolve_default().get_recall(recall_id)


def memory_access_timeline(memory_id: str) -> list[dict[str, Any]]:
    return _resolve_default().memory_access_timeline(memory_id)


def user_activity(user_id: str, *, since: str | None = None) -> dict[str, Any]:
    return _resolve_default().user_activity(user_id, since=since)


# ``replace`` re-exported so external callers can derive variants of a
# RecallSummary without having to import ``dataclasses.replace`` directly.
__all__ = [
    "RecallDetail",
    "RecallSummary",
    "TraceExplorer",
    "default_log_path",
    "get_recall",
    "list_recalls",
    "memory_access_timeline",
    "replace",
    "reset_cache",
    "user_activity",
]
