"""Unit + integration tests for the audit trace explorer.

These tests are TDD-first: they exercise the not-yet-written
``attestor.audit.explorer`` module and the ``/audit/*`` REST surface
mounted on the existing Starlette ASGI app. They MUST fail before the
implementation lands.

The explorer is a read-only consumer of the JSONL trace log written by
``attestor.trace`` (see ``ATTESTOR_TRACE_FILE``). It groups events by
``recall_id`` (set by ``recall_scope()``) and surfaces summaries +
details for the audit dashboard.
"""

from __future__ import annotations

import importlib
import json
import time
from pathlib import Path
from typing import Any

import pytest


# ──────────────────────────────────────────────────────────────────────
# Trace log fixtures
# ──────────────────────────────────────────────────────────────────────


def _write_jsonl(path: Path, events: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ev in events:
            fh.write(json.dumps(ev) + "\n")


def _make_recall_events(
    recall_id: str,
    *,
    user_id: str | None = None,
    namespace: str = "default",
    query: str = "what was the meeting about?",
    lanes: tuple[str, ...] = ("vector", "bm25"),
    result_ids: tuple[str, ...] = ("m-1", "m-2"),
    started_at: str = "2026-05-01T12:00:00.000+00:00",
    completed_at: str = "2026-05-01T12:00:00.150+00:00",
    cost_usd: float = 0.00012,
    elapsed_ms: float | None = None,
) -> list[dict[str, Any]]:
    # Derive elapsed_ms from timestamps when not explicitly provided so
    # callers don't have to keep the two in sync by hand.
    if elapsed_ms is None:
        try:
            from datetime import datetime
            t0 = datetime.fromisoformat(started_at)
            t1 = datetime.fromisoformat(completed_at)
            elapsed_ms = (t1 - t0).total_seconds() * 1000.0
        except ValueError:
            elapsed_ms = 0.0
    base: dict[str, Any] = {
        "recall_id": recall_id,
        "namespace": namespace,
    }
    if user_id is not None:
        base["user_id"] = user_id

    events: list[dict[str, Any]] = [
        {**base, "ts": started_at, "event": "recall.start", "seq": 1,
         "query": query, "token_budget": 2000},
    ]
    seq = 2
    for lane in lanes:
        events.append({
            **base, "ts": started_at, "event": f"recall.lane.{lane}",
            "seq": seq, "candidates": 10, "latency_ms": 12.5,
        })
        seq += 1
    events.append({
        **base, "ts": completed_at, "event": "recall.complete",
        "seq": seq,
        "elapsed_ms": elapsed_ms, "cost_usd": cost_usd,
        "result_count": len(result_ids),
        "result_ids": list(result_ids),
    })
    return events


@pytest.fixture
def trace_log(tmp_path: Path) -> Path:
    """A populated JSONL trace log with three recalls."""
    log = tmp_path / "trace.jsonl"
    events: list[dict[str, Any]] = []
    events += _make_recall_events(
        "rec-001", user_id="alice", namespace="ns-a",
        query="when did we ship v4?",
        lanes=("vector", "bm25", "graph"),
        result_ids=("m-10", "m-11"),
        started_at="2026-05-01T08:00:00.000+00:00",
        completed_at="2026-05-01T08:00:00.120+00:00",
    )
    events += _make_recall_events(
        "rec-002", user_id="bob", namespace="ns-b",
        query="what's the deploy plan?",
        lanes=("vector",),
        result_ids=("m-20",),
        started_at="2026-05-01T09:30:00.000+00:00",
        completed_at="2026-05-01T09:30:00.090+00:00",
    )
    events += _make_recall_events(
        "rec-003", user_id="alice", namespace="ns-a",
        query="follow-up on v4 release",
        lanes=("vector", "bm25", "reranker"),
        result_ids=("m-10", "m-30", "m-31"),
        started_at="2026-05-01T11:15:00.000+00:00",
        completed_at="2026-05-01T11:15:00.205+00:00",
    )
    _write_jsonl(log, events)
    return log


@pytest.fixture
def empty_trace_log(tmp_path: Path) -> Path:
    log = tmp_path / "trace.jsonl"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.touch()
    return log


@pytest.fixture
def explorer(trace_log: Path):
    from attestor.audit import explorer as mod

    return mod.TraceExplorer(trace_log)


@pytest.fixture
def empty_explorer(empty_trace_log: Path):
    from attestor.audit import explorer as mod

    return mod.TraceExplorer(empty_trace_log)


# ──────────────────────────────────────────────────────────────────────
# Module-level smoke
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_module_exports_public_symbols():
    mod = importlib.import_module("attestor.audit.explorer")
    for sym in ("RecallSummary", "RecallDetail", "TraceExplorer",
                "list_recalls", "get_recall",
                "memory_access_timeline", "user_activity"):
        assert hasattr(mod, sym), f"missing {sym}"


# ──────────────────────────────────────────────────────────────────────
# list_recalls
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_list_recalls_empty_log(empty_explorer):
    out = empty_explorer.list_recalls()
    assert out == []


@pytest.mark.unit
def test_list_recalls_groups_by_recall_id(explorer):
    out = explorer.list_recalls()
    assert {r.recall_id for r in out} == {"rec-001", "rec-002", "rec-003"}
    # Newest-first ordering.
    assert out[0].recall_id == "rec-003"
    assert out[-1].recall_id == "rec-001"


@pytest.mark.unit
def test_list_recalls_summary_shape(explorer):
    out = explorer.list_recalls()
    summary = next(r for r in out if r.recall_id == "rec-001")
    assert summary.user_id == "alice"
    assert summary.namespace == "ns-a"
    assert summary.query == "when did we ship v4?"
    assert summary.result_count == 2
    assert summary.elapsed_ms == 120.0
    # Lanes are deduped + tuple-typed.
    assert set(summary.lanes) == {"vector", "bm25", "graph"}
    assert isinstance(summary.lanes, tuple)


@pytest.mark.unit
def test_list_recalls_filters_by_user(explorer):
    out = explorer.list_recalls(user_id="alice")
    assert {r.recall_id for r in out} == {"rec-001", "rec-003"}


@pytest.mark.unit
def test_list_recalls_filters_by_namespace(explorer):
    out = explorer.list_recalls(namespace="ns-b")
    assert [r.recall_id for r in out] == ["rec-002"]


@pytest.mark.unit
def test_list_recalls_filters_by_since(explorer):
    out = explorer.list_recalls(since="2026-05-01T10:00:00+00:00")
    # Only rec-003 starts after 10:00.
    assert [r.recall_id for r in out] == ["rec-003"]


@pytest.mark.unit
def test_list_recalls_limit_honored(explorer):
    out = explorer.list_recalls(limit=2)
    assert len(out) == 2
    # Newest-first → rec-003, rec-002.
    assert [r.recall_id for r in out] == ["rec-003", "rec-002"]


# ──────────────────────────────────────────────────────────────────────
# get_recall
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_get_recall_returns_detail(explorer):
    detail = explorer.get_recall("rec-001")
    assert detail is not None
    assert detail.recall_id == "rec-001"
    assert detail.query == "when did we ship v4?"
    # Detail extends summary AND carries the full event tree.
    assert len(detail.events) == 5  # start + 3 lanes + complete
    assert detail.events[0]["event"] == "recall.start"
    assert detail.events[-1]["event"] == "recall.complete"
    # Events are tuple (frozen) — not list.
    assert isinstance(detail.events, tuple)


@pytest.mark.unit
def test_get_recall_missing_id_returns_none(explorer):
    assert explorer.get_recall("does-not-exist") is None


# ──────────────────────────────────────────────────────────────────────
# memory_access_timeline
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_memory_access_timeline(explorer):
    # m-10 was returned by rec-001 AND rec-003.
    tl = explorer.memory_access_timeline("m-10")
    assert len(tl) == 2
    recall_ids = {row["recall_id"] for row in tl}
    assert recall_ids == {"rec-001", "rec-003"}
    # Rows are sorted ascending by ts (oldest first).
    assert tl[0]["ts"] <= tl[1]["ts"]
    # Each row has the query and the recall context.
    for row in tl:
        assert "query" in row
        assert "user_id" in row
        assert "namespace" in row


@pytest.mark.unit
def test_memory_access_timeline_unseen_id(explorer):
    assert explorer.memory_access_timeline("never-recalled") == []


# ──────────────────────────────────────────────────────────────────────
# user_activity
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_user_activity_aggregates(explorer):
    stats = explorer.user_activity("alice")
    assert stats["user_id"] == "alice"
    assert stats["recall_count"] == 2
    # Memories accessed across both recalls (m-10 dedup'd? no — total accesses).
    assert stats["memories_accessed_total"] == 5  # 2 + 3
    # Distinct memory ids across user's recalls = m-10, m-11, m-30, m-31.
    assert stats["memories_accessed_distinct"] == 4
    # Lane-usage histogram.
    histo = stats["lane_usage"]
    assert histo["vector"] == 2
    assert histo["bm25"] == 2
    assert histo["graph"] == 1
    assert histo["reranker"] == 1


@pytest.mark.unit
def test_user_activity_since_filter(explorer):
    stats = explorer.user_activity("alice", since="2026-05-01T10:00:00+00:00")
    assert stats["recall_count"] == 1


@pytest.mark.unit
def test_user_activity_unknown_user(explorer):
    stats = explorer.user_activity("nobody")
    assert stats["recall_count"] == 0
    assert stats["memories_accessed_total"] == 0


# ──────────────────────────────────────────────────────────────────────
# Module-level convenience functions (point at default log path)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_module_functions_use_env_path(monkeypatch, trace_log):
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(trace_log))
    # Force a re-read in the explorer cache.
    from attestor.audit import explorer as mod
    mod.reset_cache()

    out = mod.list_recalls()
    assert {r.recall_id for r in out} == {"rec-001", "rec-002", "rec-003"}

    detail = mod.get_recall("rec-002")
    assert detail is not None
    assert detail.user_id == "bob"

    tl = mod.memory_access_timeline("m-20")
    assert len(tl) == 1

    stats = mod.user_activity("bob")
    assert stats["recall_count"] == 1


# ──────────────────────────────────────────────────────────────────────
# Indexing perf — must handle 100k events under 500ms on first call,
# and subsequent calls hit the cache.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_index_performance_100k_events(tmp_path):
    log = tmp_path / "big.jsonl"
    events: list[dict[str, Any]] = []
    # ~17k recalls * 6 events each ≈ 100k events.
    for i in range(17_000):
        events += _make_recall_events(
            f"rec-{i:05d}", user_id=f"user-{i % 50}", namespace="ns",
            lanes=("vector", "bm25", "graph", "reranker"),
            result_ids=(f"m-{i}-1", f"m-{i}-2", f"m-{i}-3"),
        )
    _write_jsonl(log, events)
    assert sum(1 for _ in log.open()) >= 100_000

    from attestor.audit import explorer as mod
    exp = mod.TraceExplorer(log)
    t0 = time.monotonic()
    summaries = exp.list_recalls(limit=10)
    elapsed = time.monotonic() - t0
    # 500ms budget for the cold path.
    assert elapsed < 0.5, f"index too slow: {elapsed:.3f}s"
    assert len(summaries) == 10


# ──────────────────────────────────────────────────────────────────────
# REST endpoints — wired into existing api.py
# ──────────────────────────────────────────────────────────────────────


@pytest.fixture
def api_client(monkeypatch, trace_log):
    """Starlette TestClient with audit endpoints active and SOLO mode (no auth)."""
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(trace_log))
    monkeypatch.setenv("ATTESTOR_AUDIT_ENABLED", "1")
    monkeypatch.delenv("ATTESTOR_MODE", raising=False)
    # Reset the audit cache so the new file is read.
    from attestor.audit import explorer as mod
    mod.reset_cache()

    # Reload api.py to rebuild routes with audit toggled on.
    import attestor.api as api_mod
    importlib.reload(api_mod)

    from starlette.testclient import TestClient
    return TestClient(api_mod.app)


@pytest.mark.unit
def test_endpoint_list_recalls_returns_json(api_client):
    r = api_client.get("/audit/recalls")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/json")
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["data"], list)
    assert len(body["data"]) == 3


@pytest.mark.unit
def test_endpoint_list_recalls_filter(api_client):
    r = api_client.get("/audit/recalls", params={"user_id": "alice"})
    assert r.status_code == 200
    body = r.json()
    assert {item["recall_id"] for item in body["data"]} == {"rec-001", "rec-003"}


@pytest.mark.unit
def test_endpoint_recall_detail(api_client):
    r = api_client.get("/audit/recall/rec-001")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["data"]["recall_id"] == "rec-001"
    assert len(body["data"]["events"]) == 5


@pytest.mark.unit
def test_endpoint_recall_detail_404(api_client):
    r = api_client.get("/audit/recall/does-not-exist")
    assert r.status_code == 404
    assert r.json()["ok"] is False


@pytest.mark.unit
def test_endpoint_memory_access(api_client):
    r = api_client.get("/audit/memory/m-10/access")
    assert r.status_code == 200
    body = r.json()
    assert len(body["data"]) == 2


@pytest.mark.unit
def test_endpoint_user_activity(api_client):
    r = api_client.get("/audit/user/alice/activity")
    assert r.status_code == 200
    body = r.json()
    assert body["data"]["recall_count"] == 2


@pytest.mark.unit
def test_endpoint_disabled_when_audit_off(monkeypatch, trace_log):
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(trace_log))
    monkeypatch.delenv("ATTESTOR_AUDIT_ENABLED", raising=False)
    monkeypatch.delenv("ATTESTOR_MODE", raising=False)

    import attestor.api as api_mod
    importlib.reload(api_mod)
    from starlette.testclient import TestClient
    client = TestClient(api_mod.app)
    r = client.get("/audit/recalls")
    # With audit disabled the route does not exist — Starlette returns 404.
    assert r.status_code == 404


@pytest.mark.unit
def test_endpoint_auth_required_in_hosted_mode(monkeypatch, trace_log):
    """In HOSTED mode, /audit/* requires a bearer token like /recall does."""
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(trace_log))
    monkeypatch.setenv("ATTESTOR_AUDIT_ENABLED", "1")
    monkeypatch.setenv("ATTESTOR_MODE", "hosted")
    # Public key set so the middleware doesn't 500 with "auth not configured".
    monkeypatch.setenv("ATTESTOR_JWT_PUBLIC_KEY", "dummy-key-for-route-check")

    import attestor.api as api_mod
    importlib.reload(api_mod)
    from starlette.testclient import TestClient
    client = TestClient(api_mod.app)
    r = client.get("/audit/recalls")
    # No bearer header → 401.
    assert r.status_code == 401


# ──────────────────────────────────────────────────────────────────────
# Static dashboard
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_dashboard_html_served(api_client):
    r = api_client.get("/audit.html")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "Attestor" in r.text
    assert "Audit" in r.text


@pytest.mark.unit
def test_dashboard_calls_endpoints(api_client):
    """The dashboard's JS must reference the documented endpoints."""
    r = api_client.get("/audit.html")
    assert r.status_code == 200
    body = r.text
    # We expect fetch('/audit/...') calls in the inline JS or referenced JS file.
    js_path = "/audit/recalls"
    if js_path not in body:
        # If the dashboard separates its JS into a file, the link tag must
        # reference it AND the file itself must call /audit/recalls.
        assert "audit.js" in body
        r2 = api_client.get("/ui/static/audit.js")
        assert r2.status_code == 200
        assert "/audit/recalls" in r2.text
    else:
        assert "/audit/recall/" in body
        assert "/audit/user/" in body
