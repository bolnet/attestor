"""Integration tests for the multi-query lane wired into the orchestrator.

The pure rewriter + RRF logic is covered by ``test_multi_query.py``.
This file checks that the orchestrator opens the lane only when
``multi_query_cfg.enabled`` is True, and that the lane's RRF-merged
output flows into the existing pipeline (BM25/graph/MMR/budget fit)
without behavior drift.

Lanes that hit the LLM are mocked — we verify call shape + counts,
not real model output.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from attestor.config import MultiQueryCfg
from attestor.retrieval.multi_query import RewriteResult


class FakeVectorStore:
    """Records calls so tests can assert fan-out behavior."""

    def __init__(self, hits_per_query: Dict[str, List[Dict]]):
        self.hits_per_query = hits_per_query
        self.calls: List[str] = []

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        namespace: Optional[str] = None,
        **kwargs: Any,
    ) -> List[Dict]:
        self.calls.append(query)
        return list(self.hits_per_query.get(query, []))


class FakeStore:
    """Minimal DocumentStore stub that returns real Memory objects
    keyed by id. We use the real dataclass (not a MagicMock) because
    the scorer does numeric ops on confidence + access_count + dates,
    and MagicMock fields fail those operations."""

    def __init__(self):
        self._mems: Dict[str, Any] = {}

    def add_memory(self, memory_id: str, content: str = "x") -> None:
        from attestor.models import Memory

        # Real dataclass — defaults give us a coherent Memory for the
        # scorer's confidence/access_count math.
        m = Memory(id=memory_id, content=content)
        self._mems[memory_id] = m

    def get(self, memory_id: str):
        return self._mems.get(memory_id)


@pytest.mark.unit
def test_multi_query_disabled_runs_single_vector_call() -> None:
    """multi_query_cfg.enabled=False → exactly one vector_store.search."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"who is alice": [{"memory_id": "a", "distance": 0.1}]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    # Default — multi_query_cfg is None on construction
    assert orch.multi_query_cfg is None

    orch.recall("who is alice")
    assert vec.calls == ["who is alice"], (
        f"expected single-query path, got {vec.calls}"
    )


@pytest.mark.unit
def test_multi_query_enabled_fans_out_to_n_plus_1_calls(monkeypatch) -> None:
    """multi_query_cfg.enabled=True → 1 + n vector_store.search calls,
    one per (original + paraphrase)."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval import multi_query as mq

    store = FakeStore()
    store.add_memory("a")
    store.add_memory("b")
    vec = FakeVectorStore({
        "who is alice":      [{"memory_id": "a", "distance": 0.1}],
        "alice's identity":  [{"memory_id": "a", "distance": 0.15},
                               {"memory_id": "b", "distance": 0.2}],
        "tell me about her": [{"memory_id": "b", "distance": 0.18}],
    })

    # Stub the rewriter — we test wire-up, not LLM behavior.
    monkeypatch.setattr(
        mq, "rewrite_query",
        lambda q, n=3, model=None, api_key=None, timeout=30.0: RewriteResult(
            original=q, paraphrases=["alice's identity", "tell me about her"],
        ),
    )

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.multi_query_cfg = MultiQueryCfg(enabled=True, n=2)

    orch.recall("who is alice")

    # Exactly 3 fan-out calls: original + 2 paraphrases.
    assert vec.calls == [
        "who is alice", "alice's identity", "tell me about her",
    ], f"expected fan-out, got {vec.calls}"


@pytest.mark.unit
def test_multi_query_consensus_hit_outranks_lone_top(monkeypatch) -> None:
    """A memory that appears in ALL fan-out lanes should outrank a memory
    that's #1 in only one lane — proves RRF actually drives the merge."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval import multi_query as mq

    store = FakeStore()
    store.add_memory("consensus")
    store.add_memory("lone_top")
    for i in range(5):
        store.add_memory(f"x{i}")

    vec = FakeVectorStore({
        # Lane 0 (original): lone_top at #1, consensus way down at #6
        "q": [
            {"memory_id": "lone_top", "distance": 0.05},
            {"memory_id": "x0", "distance": 0.1},
            {"memory_id": "x1", "distance": 0.15},
            {"memory_id": "x2", "distance": 0.2},
            {"memory_id": "x3", "distance": 0.25},
            {"memory_id": "consensus", "distance": 0.3},
        ],
        # Lane 1: consensus at #2
        "q2": [
            {"memory_id": "x4", "distance": 0.1},
            {"memory_id": "consensus", "distance": 0.15},
        ],
        # Lane 2: consensus at #1
        "q3": [
            {"memory_id": "consensus", "distance": 0.05},
        ],
    })

    monkeypatch.setattr(
        mq, "rewrite_query",
        lambda q, n=3, model=None, api_key=None, timeout=30.0: RewriteResult(
            original=q, paraphrases=["q2", "q3"],
        ),
    )

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.multi_query_cfg = MultiQueryCfg(enabled=True, n=2)
    orch.enable_mmr = False  # isolate the multi-query effect

    results = orch.recall("q")
    ids = [r.memory.id for r in results]
    assert "consensus" in ids, f"consensus missing from results: {ids}"
    assert "lone_top" in ids, f"lone_top missing from results: {ids}"
    # consensus must rank above lone_top (RRF lifted it via the 3 lanes)
    assert ids.index("consensus") < ids.index("lone_top"), (
        f"RRF didn't lift consensus above lone_top; order = {ids}"
    )


@pytest.mark.unit
def test_multi_query_disabled_when_cfg_object_present_but_off() -> None:
    """A MultiQueryCfg with enabled=False must NOT fan out — same
    behavior as cfg=None."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"x": [{"memory_id": "a", "distance": 0.1}]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.multi_query_cfg = MultiQueryCfg(enabled=False, n=3)

    orch.recall("x")
    assert vec.calls == ["x"], (
        f"enabled=False should be single-query, got {vec.calls}"
    )


@pytest.mark.unit
def test_yaml_loader_parses_multi_query_block(tmp_path) -> None:
    """configs/attestor.yaml's retrieval.multi_query.* block must
    flow through to RetrievalCfg.multi_query."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(
        """
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
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
  retrieval:
    vector_top_k: 99
    multi_query:
      enabled: true
      n: 5
      rewriter_model: openai/custom
      rewriter_reasoning_effort: medium
      merge: union
""",
    )

    stack = load_stack(yaml_path)
    mq = stack.retrieval.multi_query
    assert mq.enabled is True
    assert mq.n == 5
    assert mq.rewriter_model == "openai/custom"
    assert mq.rewriter_reasoning_effort == "medium"
    assert mq.merge == "union"
    assert stack.retrieval.vector_top_k == 99


@pytest.mark.unit
def test_yaml_loader_defaults_multi_query_to_disabled(tmp_path) -> None:
    """When retrieval.multi_query is omitted, MultiQueryCfg defaults
    apply (enabled=False, n=3, merge=rrf)."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(
        """
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
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
  retrieval:
    vector_top_k: 50
""",
    )

    stack = load_stack(yaml_path)
    mq = stack.retrieval.multi_query
    assert mq.enabled is False
    assert mq.n == 3
    assert mq.merge == "rrf"
    assert mq.rewriter_model is None
