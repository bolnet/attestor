"""Unit tests for the multi-query retrieval module.

The rewriter LLM call is mocked — we verify the parsing + RRF merge
deterministically. End-to-end testing happens via the LME smoke once
the orchestrator is wired.
"""

from __future__ import annotations


import pytest

from attestor.retrieval.multi_query import (
    RRF_K,
    RewriteResult,
    _parse_rewrites,
    reciprocal_rank_fusion,
    union_merge,
)


# ── _parse_rewrites ───────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_rewrites_plain_json_array() -> None:
    text = '["where does she live", "her current address", "address now"]'
    out = _parse_rewrites(text, n=3)
    assert out == ["where does she live", "her current address", "address now"]


@pytest.mark.unit
def test_parse_rewrites_strips_fenced_code_block() -> None:
    text = '```json\n["q1", "q2"]\n```'
    assert _parse_rewrites(text, n=3) == ["q1", "q2"]


@pytest.mark.unit
def test_parse_rewrites_finds_array_in_noisy_text() -> None:
    text = "Here are the rewrites:\n[\"a\", \"b\", \"c\"]\nLet me know!"
    assert _parse_rewrites(text, n=3) == ["a", "b", "c"]


@pytest.mark.unit
def test_parse_rewrites_caps_at_n() -> None:
    text = '["a","b","c","d","e"]'
    assert _parse_rewrites(text, n=3) == ["a", "b", "c"]


@pytest.mark.unit
def test_parse_rewrites_dedupes() -> None:
    text = '["same", "same", "different"]'
    assert _parse_rewrites(text, n=3) == ["same", "different"]


@pytest.mark.unit
def test_parse_rewrites_returns_empty_on_garbage() -> None:
    assert _parse_rewrites("not json at all", n=3) == []
    assert _parse_rewrites("", n=3) == []
    assert _parse_rewrites('{"not": "an array"}', n=3) == []


# ── RewriteResult.queries ─────────────────────────────────────────────


@pytest.mark.unit
def test_rewrite_result_queries_includes_original_first() -> None:
    r = RewriteResult(original="who is she", paraphrases=["her name", "the name"])
    assert r.queries == ["who is she", "her name", "the name"]


@pytest.mark.unit
def test_rewrite_result_no_paraphrases_returns_just_original() -> None:
    r = RewriteResult(original="x")
    assert r.queries == ["x"]


# ── reciprocal_rank_fusion ────────────────────────────────────────────


@pytest.mark.unit
def test_rrf_single_lane_preserves_order() -> None:
    lane = [{"memory_id": "a"}, {"memory_id": "b"}, {"memory_id": "c"}]
    merged = reciprocal_rank_fusion([lane])
    assert [h["memory_id"] for h in merged] == ["a", "b", "c"]


@pytest.mark.unit
def test_rrf_empty_input_returns_empty() -> None:
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[]]) == []


@pytest.mark.unit
def test_rrf_consensus_beats_one_off_top_rank() -> None:
    """A memory at rank 5 in BOTH lanes should beat a memory at rank 1
    in only one lane — the whole point of RRF."""
    lane_a = [
        {"memory_id": "lone_top"},  # rank 1 in lane A only
        {"memory_id": "x"},
        {"memory_id": "x2"},
        {"memory_id": "x3"},
        {"memory_id": "consensus"},  # rank 5 in lane A
    ]
    lane_b = [
        {"memory_id": "y"},
        {"memory_id": "y2"},
        {"memory_id": "y3"},
        {"memory_id": "y4"},
        {"memory_id": "consensus"},  # rank 5 in lane B
    ]
    merged = reciprocal_rank_fusion([lane_a, lane_b])
    ids = [h["memory_id"] for h in merged]
    consensus_pos = ids.index("consensus")
    lone_pos = ids.index("lone_top")
    assert consensus_pos < lone_pos, (
        "consensus at rank 5 in both lanes should outrank lone_top at "
        f"rank 1 in one lane only; got order {ids}"
    )


@pytest.mark.unit
def test_rrf_score_uses_60_constant() -> None:
    lane = [{"memory_id": "a"}]
    merged = reciprocal_rank_fusion([lane], k=RRF_K)
    assert merged[0]["rrf_score"] == pytest.approx(1.0 / (RRF_K + 1), abs=1e-6)


@pytest.mark.unit
def test_rrf_annotates_per_lane_ranks() -> None:
    """Each merged hit should record its rank in every lane it appeared in."""
    lane_a = [{"memory_id": "x"}, {"memory_id": "y"}]
    lane_b = [{"memory_id": "y"}]
    merged = reciprocal_rank_fusion([lane_a, lane_b])
    by_id = {h["memory_id"]: h for h in merged}
    assert by_id["x"]["per_lane_ranks"] == [1]
    assert by_id["y"]["per_lane_ranks"] == [2, 1]


@pytest.mark.unit
def test_rrf_skips_hits_without_key() -> None:
    """Defensive — malformed hit dicts shouldn't crash the merger."""
    lane = [{"memory_id": "a"}, {"no_id": True}, {"memory_id": "b"}]
    merged = reciprocal_rank_fusion([lane])
    ids = [h["memory_id"] for h in merged]
    assert ids == ["a", "b"]


@pytest.mark.unit
def test_rrf_does_not_mutate_input() -> None:
    """Caller's hit dicts should be untouched — we copy before annotating."""
    lane = [{"memory_id": "a"}, {"memory_id": "b"}]
    original = [dict(h) for h in lane]
    reciprocal_rank_fusion([lane])
    assert lane == original


@pytest.mark.unit
def test_rrf_deterministic_across_runs() -> None:
    """Same input → same output. The orchestrator depends on this."""
    lane_a = [{"memory_id": str(i)} for i in range(10)]
    lane_b = [{"memory_id": str(i)} for i in range(5, 15)]
    a = reciprocal_rank_fusion([lane_a, lane_b])
    b = reciprocal_rank_fusion([lane_a, lane_b])
    assert [h["memory_id"] for h in a] == [h["memory_id"] for h in b]


# ── union_merge ───────────────────────────────────────────────────────


@pytest.mark.unit
def test_union_preserves_first_seen_order() -> None:
    lane_a = [{"memory_id": "x"}, {"memory_id": "y"}]
    lane_b = [{"memory_id": "y"}, {"memory_id": "z"}]
    merged = union_merge([lane_a, lane_b])
    assert [h["memory_id"] for h in merged] == ["x", "y", "z"]


@pytest.mark.unit
def test_union_dedupes_across_lanes() -> None:
    lane_a = [{"memory_id": "x"}]
    lane_b = [{"memory_id": "x"}]
    merged = union_merge([lane_a, lane_b])
    assert len(merged) == 1


# ── multi_query_search end-to-end (mocked) ────────────────────────────


@pytest.mark.unit
def test_multi_query_search_runs_n_lanes(monkeypatch) -> None:
    """multi_query_search should invoke the vector_search callable once
    per (original + paraphrase) query."""
    from attestor.retrieval import multi_query as mq

    # Stub the rewriter so we don't hit OpenRouter.
    monkeypatch.setattr(
        mq, "rewrite_query",
        lambda q, n=3, model=None, api_key=None, timeout=30.0: RewriteResult(
            original=q, paraphrases=[f"{q} v1", f"{q} v2"],
        ),
    )

    calls = []

    def fake_search(q: str):
        calls.append(q)
        # Return a small lane unique to each query.
        return [{"memory_id": f"{q}-hit-1"}, {"memory_id": "shared"}]

    queries, merged = mq.multi_query_search(
        "who is the cto",
        vector_search=fake_search,
        n=2,
    )

    assert calls == ["who is the cto", "who is the cto v1", "who is the cto v2"]
    assert "shared" in {h["memory_id"] for h in merged}
    # "shared" appears in all 3 lanes → highest RRF score
    assert merged[0]["memory_id"] == "shared"


@pytest.mark.unit
def test_multi_query_search_handles_lane_failure(monkeypatch) -> None:
    """If one lane raises, the other lanes' results should still merge."""
    from attestor.retrieval import multi_query as mq

    monkeypatch.setattr(
        mq, "rewrite_query",
        lambda q, n=3, model=None, api_key=None, timeout=30.0: RewriteResult(
            original=q, paraphrases=["v1"],
        ),
    )

    def flaky_search(q: str):
        if "v1" in q:
            raise RuntimeError("simulated vector store outage")
        return [{"memory_id": "ok"}]

    queries, merged = mq.multi_query_search(
        "x", vector_search=flaky_search, n=1,
    )
    # Original query succeeded, paraphrase failed — merged should contain "ok"
    assert {h["memory_id"] for h in merged} == {"ok"}


@pytest.mark.unit
def test_multi_query_search_union_strategy(monkeypatch) -> None:
    """When merge='union', no rrf_score is added — order is first-seen."""
    from attestor.retrieval import multi_query as mq

    monkeypatch.setattr(
        mq, "rewrite_query",
        lambda q, n=3, model=None, api_key=None, timeout=30.0: RewriteResult(
            original=q, paraphrases=["v1"],
        ),
    )

    def fake_search(q: str):
        return [{"memory_id": q}]

    queries, merged = mq.multi_query_search(
        "x", vector_search=fake_search, n=1, merge="union",
    )
    assert "rrf_score" not in merged[0]
    # Original "x" came first, paraphrase "v1" second.
    assert [h["memory_id"] for h in merged] == ["x", "v1"]
