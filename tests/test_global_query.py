"""Multi-hop / global query lane — unit tests.

The 6-step recall pipeline (vector → BM25 → RRF → reranker → graph BFS →
MMR → fit) is tuned for LOCAL queries: "what did I say about X?",
"what's my pre-approval amount?". It silently fails on GLOBAL queries
that span many memories and need cluster-level synthesis:

  - "summarize my interactions with Wells Fargo across 6 months"
  - "what are my recurring concerns this quarter?"
  - "what trips have I taken over 8 months?"

Microsoft GraphRAG, HippoRAG, and Anthropic's contextual retrieval all
solve this by detecting global queries, running community detection on
the entity graph, summarizing each community, and returning cluster-
level facts. This module ports that pattern onto Attestor's existing
Postgres + Pinecone + Neo4j stack — Neo4j GDS already ships with
``gds.leiden``, so all we add is the wiring + the LLM summary step.

Tests follow the TDD discipline mandated by the user's coding-style
rules — write failing tests first, then minimal implementation. Every
test is independent of live Postgres / Neo4j / LLM connections; the
graph store and LLM client are mocked.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any
from unittest.mock import MagicMock

import pytest

from attestor.config.models import GlobalQueryCfg
from attestor.models import Memory, RetrievalResult
from attestor.retrieval.global_query import (
    GlobalQueryResult,
    is_global_query,
    run_global_query,
)


# ── Helpers / fakes ────────────────────────────────────────────────────


class _FakeGraphStore:
    """Minimal Neo4jBackend stand-in.

    Implements the three methods the global-query lane reaches for:

      - ``get_entities_for_query(query, namespace, ...)`` — returns the
        candidate entity names found in the user's graph that match the
        question (in the real impl this is a ``MATCH (e:Entity) WHERE
        e.display_name =~ $regex`` call). We expose a simple list here.
      - ``get_subgraph(entity, depth, namespace, user_id=...)`` — returns
        ``{"nodes": [...], "edges": [...]}`` per the Neo4jBackend
        contract.
      - ``leiden_communities(node_keys, namespace=..., user_id=...)`` —
        returns ``{community_id: [entity_keys]}``. The real impl wraps
        ``gds.leiden.stream``.

    Tests inject the responses they want via constructor kwargs.
    """

    def __init__(
        self,
        *,
        candidate_entities: list[str] | None = None,
        subgraph: dict[str, Any] | None = None,
        communities: dict[int, list[str]] | None = None,
        raise_on_subgraph: Exception | None = None,
        raise_on_leiden: Exception | None = None,
    ) -> None:
        self._candidate_entities = candidate_entities or []
        self._subgraph = subgraph or {"nodes": [], "edges": []}
        self._communities = communities or {}
        self._raise_on_subgraph = raise_on_subgraph
        self._raise_on_leiden = raise_on_leiden
        self.calls: list[tuple[str, tuple, dict]] = []

    def get_entities_for_query(
        self, query: str, *, namespace: str | None = None,
        user_id: str | None = None, limit: int = 50,
    ) -> list[str]:
        self.calls.append(("get_entities_for_query",
                           (query,), {"namespace": namespace,
                                      "user_id": user_id, "limit": limit}))
        return list(self._candidate_entities)

    def get_subgraph(
        self, entity: str, depth: int = 2,
        namespace: str | None = None, user_id: str | None = None,
    ) -> dict[str, Any]:
        self.calls.append(("get_subgraph", (entity, depth),
                           {"namespace": namespace, "user_id": user_id}))
        if self._raise_on_subgraph:
            raise self._raise_on_subgraph
        return self._subgraph

    def leiden_communities(
        self, node_keys: list[str], *,
        namespace: str | None = None, user_id: str | None = None,
    ) -> dict[int, list[str]]:
        self.calls.append(("leiden_communities", tuple(node_keys),
                           {"namespace": namespace, "user_id": user_id}))
        if self._raise_on_leiden:
            raise self._raise_on_leiden
        return dict(self._communities)


class _FakeDocStore:
    """Stand-in for PostgresBackend with the two methods we use.

    The global-query lane reaches into the doc store to pull the
    underlying memories that back each community's entities so the LLM
    can write a summary grounded in real facts.
    """

    def __init__(self, memories_for_entity: dict[str, list[Memory]]) -> None:
        self._index = memories_for_entity

    def list_memories_for_entities(
        self, entities: list[str], *,
        namespace: str | None = None, user_id: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        out: list[Memory] = []
        seen: set[str] = set()
        for ent in entities:
            for m in self._index.get(ent, []):
                if m.id in seen:
                    continue
                seen.add(m.id)
                out.append(m)
                if len(out) >= limit:
                    return out
        return out


def _summarizer_factory(prefix: str = "Summary"):
    """Build a deterministic summarizer that records every call.

    Returns ``(fn, calls)`` where ``calls`` is a mutable list of
    invocation dicts. The injected fn returns
    ``f"{prefix} of cluster {cluster_id}"``.
    """
    calls: list[dict[str, Any]] = []

    def _fn(query: str, *, cluster_id: int, memories: list[Memory],
           model: str) -> str:
        calls.append({"query": query, "cluster_id": cluster_id,
                      "memory_ids": [m.id for m in memories],
                      "model": model})
        return f"{prefix} of cluster {cluster_id}"

    return _fn, calls


# ── Classifier ─────────────────────────────────────────────────────────


@pytest.mark.unit
def test_classifier_heuristic_global() -> None:
    """Regex picks up summarize / across / over time without LLM."""
    assert is_global_query(
        "summarize my interactions with Wells Fargo over the last 6 months"
    ) is True


@pytest.mark.unit
def test_classifier_heuristic_local() -> None:
    """Specific factoid questions stay on the local lane."""
    assert is_global_query(
        "what is my Wells Fargo pre-approval amount?"
    ) is False


@pytest.mark.unit
def test_classifier_recurring_keywords() -> None:
    """``recurring`` is a global cue — return True."""
    assert is_global_query("what are my recurring concerns?") is True


@pytest.mark.unit
def test_classifier_themes_keyword() -> None:
    """``themes`` and ``patterns`` are also cues."""
    assert is_global_query("what themes show up in my journals?") is True
    assert is_global_query("any patterns in my coding habits?") is True


@pytest.mark.unit
def test_classifier_llm_tiebreaker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ambiguous queries route to the optional LLM tiebreaker.

    "what about Wells Fargo" has no global cue and no factoid form, so
    when an LLM model is supplied the classifier asks it. We mock the
    LLM call to return ``"global"`` and assert the result is True.
    """
    captured: dict[str, Any] = {}

    def _fake_call(query: str, *, model: str) -> str:
        captured["query"] = query
        captured["model"] = model
        return "global"

    monkeypatch.setattr(
        "attestor.retrieval.global_query._classify_via_llm",
        _fake_call,
    )

    out = is_global_query(
        "tell me about Wells Fargo", model="anthropic/claude-haiku-4.5",
    )
    assert out is True
    assert captured["model"] == "anthropic/claude-haiku-4.5"
    assert captured["query"] == "tell me about Wells Fargo"


@pytest.mark.unit
def test_classifier_no_model_skips_llm(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When ``model`` is None and heuristics miss, the classifier
    returns False — never silently calls the LLM."""

    called = {"yes": False}

    def _boom(*a, **kw) -> str:
        called["yes"] = True
        return "global"

    monkeypatch.setattr(
        "attestor.retrieval.global_query._classify_via_llm",
        _boom,
    )

    out = is_global_query("tell me about Wells Fargo", model=None)
    assert out is False
    assert called["yes"] is False


# ── run_global_query ──────────────────────────────────────────────────


@pytest.fixture
def trip_memories() -> list[Memory]:
    """Twelve "trip" memories spread across 8 months."""
    months = [
        "2025-09-04", "2025-10-12", "2025-11-02", "2025-11-29",
        "2025-12-15", "2026-01-08", "2026-01-25", "2026-02-13",
        "2026-02-28", "2026-03-09", "2026-03-22", "2026-04-15",
    ]
    cities = [
        "Tokyo", "Lisbon", "Bogota", "Paris", "Cairo", "Singapore",
        "Reykjavik", "Bangkok", "Berlin", "Cape Town", "Mumbai", "Quito",
    ]
    return [
        Memory(
            id=f"mem-{i:02d}",
            content=f"Took a trip to {city} on {date}",
            entity=city, category="trip",
            event_date=date, valid_from=f"{date}T00:00:00+00:00",
        )
        for i, (date, city) in enumerate(zip(months, cities, strict=True))
    ]


def _build_run_kwargs(
    *, graph_store, doc_store, summarizer=None,
    user_id: str = "user-42", namespace: str = "personal",
    max_clusters: int = 8, llm_model: str = "haiku-4.5",
):
    return dict(
        user_id=user_id, namespace=namespace,
        graph_store=graph_store, doc_store=doc_store,
        llm_model=llm_model, max_clusters=max_clusters,
        summarizer=summarizer,
    )


@pytest.mark.unit
def test_run_global_query_returns_summaries(
    trip_memories: list[Memory],
) -> None:
    """End-to-end: 12 trips → 3 communities → 3 cluster summaries."""
    cities = [m.entity for m in trip_memories if m.entity]
    graph = _FakeGraphStore(
        candidate_entities=cities,
        subgraph={
            "nodes": [{"key": c.lower(), "name": c, "type": "location"}
                      for c in cities],
            "edges": [],
        },
        communities={
            0: [cities[0].lower(), cities[1].lower(), cities[2].lower(),
                cities[3].lower()],
            1: [cities[4].lower(), cities[5].lower(), cities[6].lower(),
                cities[7].lower()],
            2: [cities[8].lower(), cities[9].lower(), cities[10].lower(),
                cities[11].lower()],
        },
    )
    doc = _FakeDocStore({c: [m] for c, m in zip(cities, trip_memories,
                                                strict=True)})

    summarizer, calls = _summarizer_factory("Summary")
    out = run_global_query(
        "summarize trips I took over the last 8 months",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=summarizer),
    )
    assert isinstance(out, GlobalQueryResult)
    assert out.cluster_count == 3
    assert len(out.summaries) == 3
    # Summarizer ran exactly once per cluster.
    assert len(calls) == 3
    assert {c["cluster_id"] for c in calls} == {0, 1, 2}
    # cluster_entities exposed for downstream observability.
    assert len(out.cluster_entities) == 3
    assert sum(len(es) for es in out.cluster_entities) == 12


@pytest.mark.unit
def test_run_global_query_respects_max_clusters() -> None:
    """``max_clusters=2`` caps the summarizer at 2 calls."""
    graph = _FakeGraphStore(
        candidate_entities=["a", "b", "c", "d", "e", "f", "g", "h"],
        subgraph={"nodes": [{"key": x, "name": x.upper(), "type": "x"}
                            for x in "abcdefgh"], "edges": []},
        # Five communities; the cap should keep only the two largest.
        communities={
            0: ["a", "b", "c"],   # 3 nodes
            1: ["d", "e"],        # 2 nodes
            2: ["f"],             # 1 node
            3: ["g"],             # 1 node
            4: ["h"],             # 1 node
        },
    )
    doc = _FakeDocStore({})  # empty doc store is fine for this assert.

    summarizer, calls = _summarizer_factory()
    out = run_global_query(
        "summarize letters",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=summarizer, max_clusters=2),
    )
    assert out.cluster_count == 2
    assert len(out.summaries) == 2
    assert len(calls) == 2


@pytest.mark.unit
def test_run_global_query_falls_back_when_graph_down() -> None:
    """If the graph raises, return empty result + log warning — never
    crash recall."""
    graph = _FakeGraphStore(
        candidate_entities=["x"],
        raise_on_subgraph=RuntimeError("neo4j down"),
    )
    doc = _FakeDocStore({})

    out = run_global_query(
        "summarize anything",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=_summarizer_factory()[0]),
    )
    assert isinstance(out, GlobalQueryResult)
    assert out.cluster_count == 0
    assert out.summaries == ()


@pytest.mark.unit
def test_run_global_query_falls_back_when_no_entities() -> None:
    """No candidate entities → empty result, no LLM call."""
    graph = _FakeGraphStore(candidate_entities=[])
    doc = _FakeDocStore({})

    summarizer, calls = _summarizer_factory()
    out = run_global_query(
        "summarize my walks",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=summarizer),
    )
    assert out.cluster_count == 0
    assert calls == []


@pytest.mark.unit
def test_run_global_query_namespace_isolation() -> None:
    """Every graph call must carry the supplied namespace."""
    graph = _FakeGraphStore(
        candidate_entities=["acme"],
        subgraph={
            "nodes": [{"key": "acme", "name": "Acme", "type": "company"}],
            "edges": [],
        },
        communities={0: ["acme"]},
    )
    doc = _FakeDocStore({"Acme": [Memory(id="m1", content="Acme uses Postgres",
                                          entity="Acme")]})
    run_global_query(
        "summarize Acme work",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=_summarizer_factory()[0],
                            namespace="tenant-7"),
    )
    # Every captured call carries namespace="tenant-7".
    namespaces = [kw.get("namespace") for _, _, kw in graph.calls]
    assert all(ns == "tenant-7" for ns in namespaces), namespaces


@pytest.mark.unit
def test_run_global_query_user_isolation_v4() -> None:
    """user_id round-trips into every graph call."""
    graph = _FakeGraphStore(
        candidate_entities=["ent"],
        subgraph={"nodes": [{"key": "ent", "name": "Ent", "type": "x"}],
                  "edges": []},
        communities={0: ["ent"]},
    )
    doc = _FakeDocStore({"Ent": [Memory(id="m", content="x")]})
    run_global_query(
        "summarize x",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=_summarizer_factory()[0],
                            user_id="USER-42"),
    )
    user_ids = [kw.get("user_id") for _, _, kw in graph.calls]
    assert all(uid == "USER-42" for uid in user_ids), user_ids


@pytest.mark.unit
def test_cost_estimate_returned() -> None:
    """``cost_estimate_usd`` is a positive float when summaries ran."""
    graph = _FakeGraphStore(
        candidate_entities=["a", "b"],
        subgraph={"nodes": [{"key": x, "name": x.upper(), "type": "x"}
                            for x in "ab"], "edges": []},
        communities={0: ["a"], 1: ["b"]},
    )
    doc = _FakeDocStore({"A": [Memory(id="m1", content="x")],
                          "B": [Memory(id="m2", content="y")]})
    out = run_global_query(
        "summarize",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=_summarizer_factory()[0]),
    )
    assert out.cost_estimate_usd >= 0.0
    # 2 community summaries, no classifier call inside run_global_query
    # (classifier is upstream). Cost should be non-zero.
    assert out.cost_estimate_usd > 0.0


# ── AgentMemory.recall() integration ───────────────────────────────────


@pytest.fixture
def stub_memory(monkeypatch: pytest.MonkeyPatch):
    """Build an AgentMemory-like stub that exercises the recall branch.

    We don't spin up Postgres here — instead we instantiate the bare
    routing layer and pass mocks. The branch under test is purely:

        if self._global_cfg.enabled and is_global_query(query, ...):
            return self._run_global_recall(query, ...)
        # else: existing 6-step pipeline

    so we can exercise it without booting backends.
    """
    from attestor.core.agent_memory import AgentMemory

    inst = AgentMemory.__new__(AgentMemory)
    inst._store = MagicMock()
    inst._store._v4 = False
    inst._vector_store = MagicMock()
    inst._graph = _FakeGraphStore(
        candidate_entities=["x"],
        subgraph={"nodes": [{"key": "x", "name": "X", "type": "t"}],
                  "edges": []},
        communities={0: ["x"]},
    )
    inst._retrieval = MagicMock()
    inst._retrieval.recall.return_value = []
    inst._ops_log = []
    inst.config = MagicMock()
    inst.config.default_token_budget = 1000
    return inst


@pytest.mark.unit
def test_recall_routes_global_to_global_lane(
    stub_memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When global_query is enabled AND the query is global, the lane
    runs and the local 6-step pipeline is bypassed."""
    cfg = GlobalQueryCfg(enabled=True, classifier_model=None,
                         summary_model="haiku-4.5")
    stub_memory._global_cfg = cfg

    invoked = {"global": False, "local": False}

    def _fake_global(query, **kw):
        invoked["global"] = True
        return [
            RetrievalResult(
                memory=Memory(id="cluster-0", content="Summary 0",
                              category="global_summary",
                              metadata={"_cluster_id": 0}),
                score=1.0, match_source="global_query",
            )
        ]

    def _fake_local(query, *a, **k):
        invoked["local"] = True
        return []

    monkeypatch.setattr(stub_memory, "_run_global_recall", _fake_global)
    stub_memory._retrieval.recall.side_effect = _fake_local

    results = stub_memory.recall(
        "summarize my interactions with X over the last 6 months",
    )
    assert invoked["global"] is True
    assert invoked["local"] is False
    assert len(results) == 1
    assert results[0].memory.category == "global_summary"


@pytest.mark.unit
def test_recall_routes_local_to_existing_pipeline(
    stub_memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A local question stays on the 6-step pipeline."""
    cfg = GlobalQueryCfg(enabled=True, classifier_model=None,
                         summary_model="haiku-4.5")
    stub_memory._global_cfg = cfg

    invoked = {"global": False, "local": False}

    def _fake_global(query, **kw):
        invoked["global"] = True
        return []

    def _fake_local(query, *a, **k):
        invoked["local"] = True
        return [
            RetrievalResult(
                memory=Memory(id="m1", content="local"),
                score=0.5, match_source="vector",
            )
        ]

    monkeypatch.setattr(stub_memory, "_run_global_recall", _fake_global)
    stub_memory._retrieval.recall.side_effect = _fake_local

    results = stub_memory.recall(
        "what is my Wells Fargo pre-approval amount?",
    )
    assert invoked["local"] is True
    assert invoked["global"] is False
    assert len(results) == 1


@pytest.mark.unit
def test_global_disabled_falls_through_to_local(
    stub_memory, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``enabled=False`` is a hard kill switch — local wins regardless
    of how global the question looks."""
    cfg = GlobalQueryCfg(enabled=False)
    stub_memory._global_cfg = cfg

    invoked = {"global": False, "local": False}

    def _fake_global(query, **kw):
        invoked["global"] = True
        return [RetrievalResult(memory=Memory(id="g"), score=1.0,
                                match_source="global_query")]

    def _fake_local(query, *a, **k):
        invoked["local"] = True
        return []

    monkeypatch.setattr(stub_memory, "_run_global_recall", _fake_global)
    stub_memory._retrieval.recall.side_effect = _fake_local

    stub_memory.recall(
        "summarize my interactions with everyone over time",
    )
    assert invoked["global"] is False
    assert invoked["local"] is True


# ── LME-style integration ─────────────────────────────────────────────


@pytest.mark.unit
def test_lme_global_query_8_month_trip_history(
    trip_memories: list[Memory],
) -> None:
    """LME-style: 12 trip memories across 8 months → cluster summary."""
    cities = [m.entity for m in trip_memories if m.entity]
    # All 12 cities cluster into one community for this test — the
    # answerer's question is "what trips" not "where are they grouped".
    graph = _FakeGraphStore(
        candidate_entities=cities,
        subgraph={
            "nodes": [{"key": c.lower(), "name": c, "type": "location"}
                      for c in cities],
            "edges": [],
        },
        communities={0: [c.lower() for c in cities]},
    )
    doc = _FakeDocStore({c: [m] for c, m in zip(cities, trip_memories,
                                                strict=True)})

    captured_memories: list[list[Memory]] = []

    def _summarizer(query, *, cluster_id, memories, model) -> str:
        captured_memories.append(list(memories))
        return (
            f"You took {len(memories)} trips covering "
            f"{', '.join(m.entity for m in memories[:3])} and more."
        )

    out = run_global_query(
        "what trips have I taken over the last 8 months?",
        **_build_run_kwargs(graph_store=graph, doc_store=doc,
                            summarizer=_summarizer),
    )
    # One cluster, one summary.
    assert out.cluster_count == 1
    assert len(out.summaries) == 1
    # The summary references the count and at least one city.
    summary = out.summaries[0]
    assert "12 trips" in summary
    # Summary received all 12 memories.
    assert len(captured_memories[0]) == 12
