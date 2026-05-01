"""Unit tests for the HyDE retrieval module.

Mirror of `tests/test_multi_query.py` — same shape (mocked LLM,
deterministic merge tests, lane-failure tolerance).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from attestor.retrieval.hyde import (
    HydeResult,
    generate_hypothetical_answer,
    hyde_search,
)


# ── HydeResult.queries ────────────────────────────────────────────────


@pytest.mark.unit
def test_hyde_result_queries_includes_original_first() -> None:
    r = HydeResult(
        original_question="who is the cto",
        hypothetical_answer="The CTO is Bob Patel.",
    )
    assert r.queries == ["who is the cto", "The CTO is Bob Patel."]


@pytest.mark.unit
def test_hyde_result_empty_hypothetical_returns_just_original() -> None:
    """Degraded path: when generation failed, queries is just [original]
    so the caller naturally falls back to single-query."""
    r = HydeResult(original_question="x", hypothetical_answer="")
    assert r.queries == ["x"]


@pytest.mark.unit
def test_hyde_result_whitespace_only_hypothetical_treated_as_empty() -> None:
    r = HydeResult(original_question="x", hypothetical_answer="   \n  ")
    assert r.queries == ["x"]


# ── generate_hypothetical_answer ──────────────────────────────────────


@pytest.mark.unit
def test_generate_no_api_key_returns_degraded(monkeypatch) -> None:
    """No API key → return original-only result, never raise."""
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    r = generate_hypothetical_answer("who is alice")
    assert r.original_question == "who is alice"
    assert r.hypothetical_answer == ""


@pytest.mark.unit
def test_generate_empty_question_returns_degraded() -> None:
    r = generate_hypothetical_answer("")
    assert r.hypothetical_answer == ""


@pytest.mark.unit
def test_generate_with_mocked_llm_returns_hypothetical(monkeypatch) -> None:
    """OpenAI is imported locally inside generate_hypothetical_answer
    so we patch the source module (`openai.OpenAI`), not the alias."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content="Bob Patel is the CTO.")),
    ]
    fake_response.id = "test-id"
    fake_response.model = "test/model"
    fake_response.usage = {
        "prompt_tokens": 5, "completion_tokens": 8, "total_tokens": 13,
    }

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("openai.OpenAI", return_value=fake_client):
        r = generate_hypothetical_answer(
            "who is the cto", model="test/m", api_key="fake",
        )

    assert "Bob Patel" in r.hypothetical_answer
    assert r.original_question == "who is the cto"


@pytest.mark.unit
def test_generate_strips_label_prefix(monkeypatch) -> None:
    """Some models echo the 'Hypothetical answer:' label — strip it."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")

    fake_response = MagicMock()
    fake_response.choices = [MagicMock(message=MagicMock(
        content="Hypothetical answer: Bob is CTO.",
    ))]
    fake_response.id = "x"
    fake_response.model = "test/m"
    fake_response.usage = {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("openai.OpenAI", return_value=fake_client):
        r = generate_hypothetical_answer("q?", model="test/m", api_key="fake")

    assert not r.hypothetical_answer.lower().startswith("hypothetical answer:")
    assert "Bob is CTO" in r.hypothetical_answer


@pytest.mark.unit
def test_generate_handles_llm_failure(monkeypatch) -> None:
    """LLM call raises → return degraded result, never escape."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")

    fake_client = MagicMock()
    fake_client.chat.completions.create.side_effect = RuntimeError("network down")

    with patch("openai.OpenAI", return_value=fake_client):
        r = generate_hypothetical_answer("q?", model="test/m", api_key="fake")

    assert r.hypothetical_answer == ""
    assert r.original_question == "q?"


# ── hyde_search end-to-end (mocked) ───────────────────────────────────


@pytest.mark.unit
def test_hyde_search_runs_two_lanes_when_generation_succeeds(monkeypatch) -> None:
    """generate succeeds → vector_search called once per (original, hypothetical)."""
    from attestor.retrieval import hyde as _h

    monkeypatch.setattr(
        _h, "generate_hypothetical_answer",
        lambda q, model=None, api_key=None, timeout=30.0: HydeResult(
            original_question=q,
            hypothetical_answer="Bob Patel is the new CTO.",
        ),
    )

    calls: list[str] = []

    def fake_search(q: str):
        calls.append(q)
        return [{"memory_id": f"{q[:5]}-hit", "distance": 0.1}]

    queries, merged = hyde_search(
        "who is the cto",
        vector_search=fake_search,
    )

    assert queries == ["who is the cto", "Bob Patel is the new CTO."]
    assert calls == queries
    # Two distinct memories merged, each annotated with rrf_score
    assert all("rrf_score" in h for h in merged)


@pytest.mark.unit
def test_hyde_search_falls_back_to_single_lane_on_degraded_gen(
    monkeypatch,
) -> None:
    """When generator returns empty hypothetical, only the original
    runs through vector_search."""
    from attestor.retrieval import hyde as _h

    monkeypatch.setattr(
        _h, "generate_hypothetical_answer",
        lambda q, model=None, api_key=None, timeout=30.0: HydeResult(
            original_question=q, hypothetical_answer="",
        ),
    )

    calls: list[str] = []

    def fake_search(q: str):
        calls.append(q)
        return [{"memory_id": "a"}]

    queries, merged = hyde_search("x", vector_search=fake_search)
    assert queries == ["x"]
    assert calls == ["x"]
    assert len(merged) == 1


@pytest.mark.unit
def test_hyde_search_handles_lane_failure(monkeypatch) -> None:
    """If one lane raises, the other lane's results still merge."""
    from attestor.retrieval import hyde as _h

    monkeypatch.setattr(
        _h, "generate_hypothetical_answer",
        lambda q, model=None, api_key=None, timeout=30.0: HydeResult(
            original_question=q,
            hypothetical_answer="hypo answer",
        ),
    )

    def flaky_search(q: str):
        if q == "x":
            return [{"memory_id": "ok"}]
        raise RuntimeError("hypothetical lane down")

    queries, merged = hyde_search("x", vector_search=flaky_search)
    assert {h["memory_id"] for h in merged} == {"ok"}


@pytest.mark.unit
def test_hyde_search_union_merge_strategy(monkeypatch) -> None:
    """When merge='union', no rrf_score annotation; first-seen order."""
    from attestor.retrieval import hyde as _h

    monkeypatch.setattr(
        _h, "generate_hypothetical_answer",
        lambda q, model=None, api_key=None, timeout=30.0: HydeResult(
            original_question=q, hypothetical_answer="hypo",
        ),
    )

    def fake_search(q: str):
        return [{"memory_id": q}]

    queries, merged = hyde_search(
        "x", vector_search=fake_search, merge="union",
    )
    assert "rrf_score" not in merged[0]
    assert [h["memory_id"] for h in merged] == ["x", "hypo"]


@pytest.mark.unit
def test_hyde_search_consensus_outranks_lone_top(monkeypatch) -> None:
    """A memory found in BOTH lanes should outrank one found in only
    one lane — proves RRF actually drives the merge ordering, same
    invariant as test_rrf_consensus_beats_one_off_top_rank in
    test_multi_query.py."""
    from attestor.retrieval import hyde as _h

    monkeypatch.setattr(
        _h, "generate_hypothetical_answer",
        lambda q, model=None, api_key=None, timeout=30.0: HydeResult(
            original_question=q, hypothetical_answer="hypo",
        ),
    )

    def fake_search(q: str):
        if q == "x":
            return [
                {"memory_id": "lone_top"},
                {"memory_id": "consensus"},
            ]
        else:
            # hypo lane: consensus at rank 1, no lone_top
            return [{"memory_id": "consensus"}, {"memory_id": "y"}]

    queries, merged = hyde_search("x", vector_search=fake_search)
    ids = [h["memory_id"] for h in merged]
    assert ids.index("consensus") < ids.index("lone_top"), (
        f"RRF should lift consensus above lone_top; got {ids}"
    )
