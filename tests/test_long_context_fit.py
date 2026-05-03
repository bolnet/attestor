"""Tests for the long-context fit mode (alternate Step 6 packing).

Long-context mode swaps the standard greedy ``fit_to_budget`` packing
for a high-cap pack that returns the top-K reranked candidates verbatim
(no MMR diversity penalty). Designed for downstream answerers running on
1M+ context models (Claude Sonnet 4.6 / Opus 4.x / Gemini 2 Pro) where
near-duplicate cuts actively hurt quality.

Default behavior must be byte-identical to the pre-feature pipeline:
``long_context=False`` is the contract; tests assert it explicitly.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from attestor.models import Memory, RetrievalResult


# ──────────────────────────────────────────────────────────────────────
# Stubs (mirror the pattern in tests/test_hyde_orchestrator.py — no
# Postgres / Pinecone / Neo4j touched).
# ──────────────────────────────────────────────────────────────────────


class _FakeStore:
    def __init__(self) -> None:
        self._mems: dict[str, Memory] = {}

    def add_memory(self, memory_id: str, content: str = "x") -> None:
        self._mems[memory_id] = Memory(id=memory_id, content=content)

    def get(self, memory_id: str) -> Memory | None:
        return self._mems.get(memory_id)


class _FakeVectorStore:
    def __init__(self, hits: list[dict[str, Any]]) -> None:
        self._hits = hits

    def search(
        self,
        query: str,
        *,
        limit: int = 50,
        namespace: str | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        return list(self._hits)


def _make_result(
    content: str, score: float, memory_id: str | None = None
) -> RetrievalResult:
    mem = Memory(content=content, **({"id": memory_id} if memory_id else {}))
    return RetrievalResult(memory=mem, score=score, match_source="vector")


# ──────────────────────────────────────────────────────────────────────
# Direct unit tests on _step6_pack / _long_context_pack
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_default_mode_uses_greedy_fit() -> None:
    """long_context=False → result is identical to fit_to_budget."""
    from attestor.retrieval.orchestrator.postprocess import _step6_pack
    from attestor.retrieval.scorer import fit_to_budget

    results = [_make_result(f"memory {i} content here", 1.0 - i * 0.05)
               for i in range(10)]
    expected = fit_to_budget(results, token_budget=20)
    actual = _step6_pack(
        results,
        token_budget=20,
        long_context_mode=False,
    )
    assert [r.memory.id for r in actual] == [r.memory.id for r in expected]


@pytest.mark.unit
def test_long_context_pack_respects_max_tokens() -> None:
    """Cumulative token count never exceeds long_context_max_tokens."""
    from attestor.retrieval.orchestrator.postprocess import _step6_pack
    from attestor.utils.tokens import estimate_tokens

    # Each content is ~13 tokens (10 words * 1.3).
    results = [
        _make_result(" ".join([f"w{i}{j}" for j in range(10)]), 1.0 - i * 0.01)
        for i in range(50)
    ]
    cap = 50  # tokens
    packed = _step6_pack(
        results,
        token_budget=999_999,  # ignored in long-context mode
        long_context_mode=True,
        long_context_max_tokens=cap,
    )
    total = sum(estimate_tokens(r.memory.content) for r in packed)
    assert total <= cap


@pytest.mark.unit
def test_long_context_pack_orders_by_score() -> None:
    """Top-scored memories surface first when packing for long-context."""
    from attestor.retrieval.orchestrator.postprocess import _step6_pack

    results = [
        _make_result("low score doc", 0.10, memory_id="low"),
        _make_result("highest score doc", 0.99, memory_id="hi"),
        _make_result("middle score doc", 0.50, memory_id="mid"),
    ]
    packed = _step6_pack(
        results,
        token_budget=100,
        long_context_mode=True,
        long_context_max_tokens=200_000,
    )
    assert [r.memory.id for r in packed] == ["hi", "mid", "low"]


@pytest.mark.unit
def test_long_context_returns_more_results_than_default() -> None:
    """Same input — long-context mode includes ≥ memories than default."""
    from attestor.retrieval.orchestrator.postprocess import _step6_pack

    # 30 candidates, each ~13 tokens; default budget=50 fits maybe 3,
    # long-context cap=200_000 fits all 30.
    results = [
        _make_result(" ".join([f"w{i}{j}" for j in range(10)]), 1.0 - i * 0.01,
                     memory_id=f"m{i}")
        for i in range(30)
    ]
    default = _step6_pack(
        results, token_budget=50, long_context_mode=False,
    )
    long_ctx = _step6_pack(
        results,
        token_budget=50,
        long_context_mode=True,
        long_context_max_tokens=200_000,
    )
    assert len(long_ctx) >= len(default)
    assert len(long_ctx) == 30


@pytest.mark.unit
def test_long_context_with_zero_candidates() -> None:
    """Empty input → empty output (both modes)."""
    from attestor.retrieval.orchestrator.postprocess import _step6_pack

    assert _step6_pack([], token_budget=2000, long_context_mode=False) == []
    assert _step6_pack(
        [], token_budget=2000, long_context_mode=True,
        long_context_max_tokens=200_000,
    ) == []


@pytest.mark.unit
def test_long_context_token_count_uses_existing_helper() -> None:
    """Long-context pack must call the same ``estimate_tokens`` helper
    as ``fit_to_budget``; otherwise the cap-vs-budget semantics drift."""
    from attestor.retrieval.orchestrator import postprocess as pp

    results = [_make_result(f"doc {i}", 1.0 - i * 0.1) for i in range(5)]
    with patch.object(pp, "estimate_tokens", wraps=pp.estimate_tokens) as spy:
        pp._step6_pack(
            results,
            token_budget=2000,
            long_context_mode=True,
            long_context_max_tokens=200_000,
        )
        assert spy.call_count >= 1


@pytest.mark.unit
def test_token_overflow_truncates_last_candidate() -> None:
    """When adding a candidate would exceed the cap, drop it whole —
    never include a partial memory."""
    from attestor.retrieval.orchestrator.postprocess import _step6_pack
    from attestor.utils.tokens import estimate_tokens

    # Each ~13 tokens.
    big = " ".join([f"w{i}" for i in range(10)])
    results = [
        _make_result(big, 0.9, memory_id="a"),
        _make_result(big, 0.8, memory_id="b"),
        _make_result(big, 0.7, memory_id="c"),
    ]
    # cap=20 → only the first ~13-token doc fits; the second would overflow.
    packed = _step6_pack(
        results,
        token_budget=999,
        long_context_mode=True,
        long_context_max_tokens=20,
    )
    total = sum(estimate_tokens(r.memory.content) for r in packed)
    assert total <= 20
    assert len(packed) == 1
    assert packed[0].memory.id == "a"


@pytest.mark.unit
def test_long_context_default_cap_200k() -> None:
    """When ``long_context_max_tokens`` is omitted, default = 200_000."""
    import inspect

    from attestor.retrieval.orchestrator.postprocess import _step6_pack

    sig = inspect.signature(_step6_pack)
    assert sig.parameters["long_context_max_tokens"].default == 200_000


@pytest.mark.unit
def test_long_context_max_tokens_override() -> None:
    """Caller-supplied cap is honored over the default."""
    from attestor.retrieval.orchestrator.postprocess import _step6_pack
    from attestor.utils.tokens import estimate_tokens

    results = [_make_result(" ".join([f"w{i}" for i in range(10)]),
                            1.0 - i * 0.01, memory_id=f"m{i}")
               for i in range(20)]
    # Cap to ~26 tokens — only two ~13-token docs fit.
    packed = _step6_pack(
        results,
        token_budget=999,
        long_context_mode=True,
        long_context_max_tokens=26,
    )
    total = sum(estimate_tokens(r.memory.content) for r in packed)
    assert total <= 26
    assert len(packed) == 2


# ──────────────────────────────────────────────────────────────────────
# Orchestrator-level: long_context kwarg threading
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_long_context_mode_skips_mmr() -> None:
    """When ``long_context=True``, the MMR rerank is NOT invoked."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval import scorer as _scorer

    store = _FakeStore()
    for i in range(5):
        store.add_memory(f"m{i}", content=f"unique content {i}")
    vec = _FakeVectorStore([
        {"memory_id": f"m{i}", "distance": 0.1 + i * 0.05}
        for i in range(5)
    ])
    orch = RetrievalOrchestrator(store=store, vector_store=vec)

    with patch.object(_scorer, "mmr_rerank", wraps=_scorer.mmr_rerank) as spy:
        orch.recall("query", long_context=True)
        spy.assert_not_called()


@pytest.mark.unit
def test_recall_kwarg_threads_through() -> None:
    """``orch.recall(..., long_context=True)`` invokes _step6_pack with
    long_context_mode=True."""
    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval.orchestrator import postprocess as pp

    store = _FakeStore()
    store.add_memory("m1")
    vec = _FakeVectorStore([{"memory_id": "m1", "distance": 0.1}])
    orch = RetrievalOrchestrator(store=store, vector_store=vec)

    with patch.object(pp, "_step6_pack", wraps=pp._step6_pack) as spy:
        orch.recall("q", long_context=True, long_context_max_tokens=12_345)
        assert spy.call_count == 1
        kwargs = spy.call_args.kwargs
        assert kwargs.get("long_context_mode") is True
        assert kwargs.get("long_context_max_tokens") == 12_345


@pytest.mark.unit
def test_recall_async_supports_long_context() -> None:
    """The async path threads ``long_context`` through to Step 6."""
    import asyncio

    from attestor.retrieval.orchestrator import RetrievalOrchestrator
    from attestor.retrieval.orchestrator import postprocess as pp

    store = _FakeStore()
    store.add_memory("m1")
    vec = _FakeVectorStore([{"memory_id": "m1", "distance": 0.1}])
    orch = RetrievalOrchestrator(store=store, vector_store=vec)

    with patch.object(pp, "_step6_pack", wraps=pp._step6_pack) as spy:
        asyncio.run(orch.recall_async("q", long_context=True))
        assert spy.call_count == 1
        assert spy.call_args.kwargs.get("long_context_mode") is True
