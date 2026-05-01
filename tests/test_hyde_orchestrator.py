"""Integration tests for the HyDE lane wired into the orchestrator.

Mirror of `tests/test_multi_query_orchestrator.py` — same FakeStore /
FakeVectorStore stubs, same wire-up assertions. Adds the
HyDE-vs-multi_query mutual-exclusion check.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import pytest

from attestor.config import HydeCfg, MultiQueryCfg
from attestor.retrieval.hyde import HydeResult


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
    """Real Memory dataclass instances keyed by id."""

    def __init__(self):
        self._mems: Dict[str, Any] = {}

    def add_memory(self, memory_id: str, content: str = "x") -> None:
        from attestor.models import Memory
        self._mems[memory_id] = Memory(id=memory_id, content=content)

    def get(self, memory_id: str):
        return self._mems.get(memory_id)


# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_hyde_disabled_runs_single_vector_call() -> None:
    """hyde_cfg.enabled=False (default) → exactly one vector_store.search."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({"x": [{"memory_id": "a", "distance": 0.1}]})

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    assert orch.hyde_cfg is None  # default

    orch.recall("x")
    assert vec.calls == ["x"]


@pytest.mark.unit
def test_hyde_enabled_fans_out_to_two_lanes(monkeypatch) -> None:
    """hyde_cfg.enabled=True + mocked generator → 2 vector_search calls."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval import hyde as _h

    store = FakeStore()
    store.add_memory("a")
    store.add_memory("b")
    vec = FakeVectorStore({
        "x":          [{"memory_id": "a", "distance": 0.1}],
        "hypo answer": [{"memory_id": "b", "distance": 0.2}],
    })

    monkeypatch.setattr(
        _h, "generate_hypothetical_answer",
        lambda q, model=None, api_key=None, timeout=30.0: HydeResult(
            original_question=q, hypothetical_answer="hypo answer",
        ),
    )

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.hyde_cfg = HydeCfg(enabled=True)

    orch.recall("x")
    assert vec.calls == ["x", "hypo answer"]


@pytest.mark.unit
def test_hyde_and_multi_query_both_enabled_prefers_multi_query(
    monkeypatch, caplog,
) -> None:
    """Mutual exclusion: when both flags are True, multi_query wins
    and a warning is logged. HyDE's generator MUST NOT run."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval import multi_query as _mq
    from attestor.retrieval import hyde as _h
    from attestor.retrieval.multi_query import RewriteResult

    store = FakeStore()
    store.add_memory("a")
    vec = FakeVectorStore({
        "x":     [{"memory_id": "a", "distance": 0.1}],
        "x v1":  [{"memory_id": "a", "distance": 0.1}],
    })

    # Stub both rewriters; we'll assert which one fires.
    mq_called = []
    hyde_called = []

    def stub_mq_rewrite(q, n=3, model=None, api_key=None, timeout=30.0):
        mq_called.append(q)
        return RewriteResult(original=q, paraphrases=["x v1"])

    def stub_hyde_gen(q, model=None, api_key=None, timeout=30.0):
        hyde_called.append(q)
        return HydeResult(original_question=q, hypothetical_answer="hypo")

    monkeypatch.setattr(_mq, "rewrite_query", stub_mq_rewrite)
    monkeypatch.setattr(_h, "generate_hypothetical_answer", stub_hyde_gen)

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.multi_query_cfg = MultiQueryCfg(enabled=True, n=1)
    orch.hyde_cfg = HydeCfg(enabled=True)

    with caplog.at_level("WARNING"):
        orch.recall("x")

    assert mq_called, "multi_query rewriter should have fired"
    assert not hyde_called, "HyDE generator must NOT fire when multi_query wins"
    assert any(
        "multi_query and retrieval.hyde both enabled" in rec.message
        for rec in caplog.records
    ), "expected mutual-exclusion warning in log"


@pytest.mark.unit
def test_hyde_consensus_hit_outranks_lone_top(monkeypatch) -> None:
    """A memory in both lanes should outrank a lane-1 hit thanks to
    RRF fusion (mirror of multi_query's analogous test)."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval import hyde as _h

    store = FakeStore()
    store.add_memory("consensus")
    store.add_memory("lone_top")
    for i in range(3):
        store.add_memory(f"x{i}")

    vec = FakeVectorStore({
        "q": [
            {"memory_id": "lone_top", "distance": 0.05},
            {"memory_id": "x0", "distance": 0.1},
            {"memory_id": "x1", "distance": 0.15},
            {"memory_id": "x2", "distance": 0.2},
            {"memory_id": "consensus", "distance": 0.25},
        ],
        "hypo": [
            {"memory_id": "consensus", "distance": 0.05},
        ],
    })

    monkeypatch.setattr(
        _h, "generate_hypothetical_answer",
        lambda q, model=None, api_key=None, timeout=30.0: HydeResult(
            original_question=q, hypothetical_answer="hypo",
        ),
    )

    orch = RetrievalOrchestrator(store=store, vector_store=vec)
    orch.hyde_cfg = HydeCfg(enabled=True)
    orch.enable_mmr = False  # isolate the HyDE/RRF effect

    results = orch.recall("q")
    ids = [r.memory.id for r in results]
    assert "consensus" in ids
    assert "lone_top" in ids
    assert ids.index("consensus") < ids.index("lone_top"), (
        f"RRF should lift consensus above lone_top; got {ids}"
    )


@pytest.mark.unit
def test_yaml_loader_parses_hyde_block(tmp_path) -> None:
    """retrieval.hyde block flows through to RetrievalCfg.hyde."""
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
    hyde:
      enabled: true
      generator_model: openai/custom
      generator_reasoning_effort: medium
      merge: union
""",
    )
    stack = load_stack(yaml_path)
    h = stack.retrieval.hyde
    assert h.enabled is True
    assert h.generator_model == "openai/custom"
    assert h.generator_reasoning_effort == "medium"
    assert h.merge == "union"


@pytest.mark.unit
def test_yaml_loader_defaults_hyde_to_disabled(tmp_path) -> None:
    """When retrieval.hyde is omitted, defaults apply (enabled=False,
    merge=rrf, generator_model=None)."""
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
""",
    )
    stack = load_stack(yaml_path)
    h = stack.retrieval.hyde
    assert h.enabled is False
    assert h.generator_model is None
    assert h.merge == "rrf"
