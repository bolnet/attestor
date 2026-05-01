"""Integration tests for the temporal pre-filter wired into the orchestrator.

Pure regex unit behavior is covered by ``test_temporal_prefilter.py``.
This file checks that the orchestrator opens Step 0 only when
``temporal_prefilter_cfg.enabled`` is True, that the computed
``TimeWindow`` flows through to the vector lane via the existing
``time_window`` kwarg, and that a caller-supplied ``time_window``
always wins.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from attestor.config import TemporalPrefilterCfg
from attestor.retrieval.temporal_query import TimeWindow


# ──────────────────────────────────────────────────────────────────────
# Test doubles — same shape as test_multi_query_orchestrator.py
# ──────────────────────────────────────────────────────────────────────


class FakeVectorStore:
    """Records the time_window kwarg passed on each search call."""

    def __init__(self, hits_per_query: dict[str, list[dict]] | None = None):
        self.hits_per_query = hits_per_query or {}
        self.calls: list[dict[str, Any]] = []

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        namespace: str | None = None,
        as_of: Any = None,
        time_window: Any = None,
        **kwargs: Any,
    ) -> list[dict]:
        self.calls.append({
            "query": query,
            "limit": limit,
            "namespace": namespace,
            "as_of": as_of,
            "time_window": time_window,
        })
        return list(self.hits_per_query.get(query, []))


class FakeStore:
    """Minimal DocumentStore stub returning real Memory objects."""

    def __init__(self) -> None:
        self._mems: dict[str, Any] = {}

    def add_memory(self, memory_id: str, content: str = "x") -> None:
        from attestor.models import Memory

        self._mems[memory_id] = Memory(id=memory_id, content=content)

    def get(self, memory_id: str):
        return self._mems.get(memory_id)


# ──────────────────────────────────────────────────────────────────────
# Default behavior — disabled
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_disabled_by_default_no_time_window_override() -> None:
    """Out of the box (no cfg wired) the orchestrator must not touch
    time_window — caller's None stays None on the vector call."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"yesterday I shipped": [
        {"memory_id": "a", "distance": 0.1},
    ]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    assert orch.temporal_prefilter_cfg is None

    orch.recall("yesterday I shipped")

    assert len(vec.calls) == 1
    assert vec.calls[0]["time_window"] is None


@pytest.mark.unit
def test_cfg_present_but_disabled_no_override() -> None:
    """A TemporalPrefilterCfg(enabled=False) must behave the same as
    cfg=None — even on a question that contains a relative phrase."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"yesterday I shipped": [
        {"memory_id": "a", "distance": 0.1},
    ]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.temporal_prefilter_cfg = TemporalPrefilterCfg(enabled=False)

    orch.recall("yesterday I shipped")

    assert vec.calls[0]["time_window"] is None


# ──────────────────────────────────────────────────────────────────────
# Enabled — phrase present
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_enabled_with_phrase_passes_time_window_to_vector_lane() -> None:
    """enabled=True + question with phrase → vector_store.search receives
    a TimeWindow centered on the implied target."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"what did I think yesterday": [
        {"memory_id": "a", "distance": 0.1},
    ]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.temporal_prefilter_cfg = TemporalPrefilterCfg(
        enabled=True, tolerance_days=3,
    )

    orch.recall("what did I think yesterday")

    assert len(vec.calls) == 1
    tw = vec.calls[0]["time_window"]
    assert isinstance(tw, TimeWindow), (
        f"expected TimeWindow on vector lane, got {type(tw).__name__}"
    )
    # Yesterday ± 3 days → a 6-day span.
    assert tw.start is not None and tw.end is not None
    span_days = (tw.end - tw.start).total_seconds() / 86400
    assert abs(span_days - 6.0) < 0.01, f"expected 6-day span, got {span_days}"


@pytest.mark.unit
def test_enabled_no_phrase_no_override() -> None:
    """enabled=True but the question has no relative phrase → no
    time_window override (None passes through to the lane)."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"what is my favorite color": [
        {"memory_id": "a", "distance": 0.1},
    ]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.temporal_prefilter_cfg = TemporalPrefilterCfg(enabled=True)

    orch.recall("what is my favorite color")

    assert vec.calls[0]["time_window"] is None


# ──────────────────────────────────────────────────────────────────────
# Caller wins
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_caller_supplied_time_window_is_preserved() -> None:
    """When the caller passes an explicit time_window, the pre-filter
    must NOT override it — even if the question would have triggered
    a different window."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"yesterday's commit": [
        {"memory_id": "a", "distance": 0.1},
    ]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.temporal_prefilter_cfg = TemporalPrefilterCfg(enabled=True)

    # Caller's window is far in the past — clearly distinguishable
    # from anything yesterday-based.
    caller_window = TimeWindow(
        start=datetime(2020, 1, 1, tzinfo=timezone.utc),
        end=datetime(2020, 1, 31, tzinfo=timezone.utc),
        interpretation="caller-supplied",
    )

    orch.recall("yesterday's commit", time_window=caller_window)

    tw = vec.calls[0]["time_window"]
    assert tw is caller_window, (
        f"caller's TimeWindow should pass through unchanged; got {tw!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# Tolerance flows through
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_custom_tolerance_flows_through_to_window() -> None:
    """tolerance_days=10 on the cfg → 20-day window span on the vector
    lane (target ± 10 days)."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"yesterday": [
        {"memory_id": "a", "distance": 0.1},
    ]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.temporal_prefilter_cfg = TemporalPrefilterCfg(
        enabled=True, tolerance_days=10,
    )

    orch.recall("yesterday")

    tw = vec.calls[0]["time_window"]
    assert tw is not None and tw.start is not None and tw.end is not None
    span_days = (tw.end - tw.start).total_seconds() / 86400
    assert abs(span_days - 20.0) < 0.01, (
        f"expected 20-day span at tolerance=10, got {span_days}"
    )


# ──────────────────────────────────────────────────────────────────────
# YAML loader
# ──────────────────────────────────────────────────────────────────────


_BASE_YAML = """
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: {{ username: neo4j, password: pw }}
    database: neo4j
  embedder:
    provider: voyage
    model: voyage-4
    dimensions: 1024
  models:
    answerer: x
    judge: x
    extraction: x
    distill: x
    verifier: x
    planner: x
    benchmark_default: x
  llm:
    provider: openrouter
  budget: 4000
  parallel: 2
  retrieval:
    vector_top_k: 50
{extra}
"""


@pytest.mark.unit
def test_yaml_loader_parses_temporal_prefilter_block(tmp_path) -> None:
    """retrieval.temporal_prefilter.* in YAML flows through to
    RetrievalCfg.temporal_prefilter."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(_BASE_YAML.format(extra="""
    temporal_prefilter:
      enabled: true
      tolerance_days: 7
"""))

    stack = load_stack(yaml_path)
    tp = stack.retrieval.temporal_prefilter
    assert tp.enabled is True
    assert tp.tolerance_days == 7


@pytest.mark.unit
def test_yaml_loader_defaults_temporal_prefilter_when_missing(tmp_path) -> None:
    """When retrieval.temporal_prefilter is omitted, the default cfg
    (enabled=False, tolerance_days=3) applies."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(_BASE_YAML.format(extra=""))

    stack = load_stack(yaml_path)
    tp = stack.retrieval.temporal_prefilter
    assert tp.enabled is False
    assert tp.tolerance_days == 3
