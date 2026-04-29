"""Unit tests for the self-consistency K-sample answerer (Phase 3 PR-B).

The answerer LLM client is mocked — we verify the K-sample loop,
fingerprint-based majority vote, judge_pick election, and the
defensive fallbacks deterministically. End-to-end testing happens via
the LME smoke once the module is wired into ``answer_question``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, List
from unittest.mock import MagicMock

import pytest


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


from attestor.longmemeval_consistency import (  # noqa: E402
    ConsistencyResult,
    _fingerprint,
    answer_with_self_consistency,
)


# ── Helpers ───────────────────────────────────────────────────────────


def _mk_response(text: str) -> Any:
    """Build a minimal OpenAI-shaped chat-completion response object."""
    response = MagicMock()
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    response.choices = [choice]
    response.model = "test/model"
    response.id = "gen-test"
    response.usage = None
    return response


def _client_returning(answers: List[str]) -> MagicMock:
    """Build a MagicMock OpenAI client whose chat.completions.create
    returns successive answers from ``answers`` on each call."""
    client = MagicMock()
    client.chat.completions.create.side_effect = [_mk_response(a) for a in answers]
    return client


# ── _fingerprint ──────────────────────────────────────────────────────


@pytest.mark.unit
def test_fingerprint_collapses_case_punct_and_whitespace() -> None:
    """Three answers that differ only in capitalization, trailing
    punctuation, and internal whitespace must hash to the same key."""
    assert _fingerprint("Bob Patel.") == _fingerprint("bob patel")
    assert _fingerprint("Bob Patel.") == _fingerprint("Bob Patel")
    assert _fingerprint("  Bob   Patel  ") == _fingerprint("Bob Patel")


@pytest.mark.unit
def test_fingerprint_distinct_for_distinct_answers() -> None:
    assert _fingerprint("Alice") != _fingerprint("Bob")
    assert _fingerprint("42") != _fingerprint("43")


@pytest.mark.unit
def test_fingerprint_empty_string_is_stable() -> None:
    assert _fingerprint("") == _fingerprint("   ")
    assert _fingerprint("") == ""


# ── answer_with_self_consistency — majority vote ─────────────────────


@pytest.mark.unit
def test_majority_vote_unanimous() -> None:
    client = _client_returning(["Bob", "Bob", "Bob", "Bob", "Bob"])
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "who?"}],
        k=5,
        temperature=0.7,
        voter="majority",
    )
    assert result.chosen == "Bob"
    assert result.voter == "majority"
    assert len(result.samples) == 5
    assert sum(result.vote_breakdown.values()) == 5


@pytest.mark.unit
def test_majority_vote_3_1_1_distribution() -> None:
    """Three samples agree on 'Alice', two outliers split — 'Alice' wins."""
    client = _client_returning(["Alice", "Alice", "Alice", "Bob", "Carol"])
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
        voter="majority",
    )
    assert result.chosen == "Alice"
    assert result.vote_breakdown[_fingerprint("Alice")] == 3


@pytest.mark.unit
def test_majority_vote_tie_picks_first_sample() -> None:
    """2-2-1 tie: two distinct answers share the top count. Stable
    tiebreak picks the first sample's answer."""
    client = _client_returning(["Bob", "Alice", "Alice", "Bob", "Carol"])
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
        voter="majority",
    )
    # Bob and Alice both have 2 votes; first sample is "Bob" → Bob wins.
    assert result.chosen == "Bob"


@pytest.mark.unit
def test_majority_vote_normalizes_fingerprints() -> None:
    """Answers that differ only in casing/punctuation merge into one
    bucket and win with their combined count."""
    client = _client_returning(["Bob Patel.", "bob patel", "Bob Patel", "Alice", "Alice"])
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
        voter="majority",
    )
    # Three Bob variants merge to 3 votes → beats Alice's 2.
    # Chosen is the first-seen surface form.
    assert result.chosen == "Bob Patel."
    assert result.vote_breakdown[_fingerprint("Bob Patel")] == 3


# ── answer_with_self_consistency — defensive paths ───────────────────


@pytest.mark.unit
def test_k_zero_returns_empty_string() -> None:
    client = MagicMock()
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=0,
        temperature=0.7,
    )
    assert result.chosen == ""
    assert result.samples == []
    # No LLM calls should fire when k=0.
    client.chat.completions.create.assert_not_called()


@pytest.mark.unit
def test_all_samples_empty_returns_empty() -> None:
    client = _client_returning(["", "  ", "\n", "", ""])
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
    )
    assert result.chosen == ""


@pytest.mark.unit
def test_one_nonempty_among_empties_is_chosen() -> None:
    """If 4 of 5 samples are empty, the lone non-empty one wins."""
    client = _client_returning(["", "Alice", "", "", ""])
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
    )
    assert result.chosen == "Alice"


@pytest.mark.unit
def test_sampler_failure_skipped_remaining_samples_continue() -> None:
    """A single sample raising should not abort the K-sample loop —
    the survivors still vote."""
    client = MagicMock()

    def flaky_create(**kwargs: Any) -> Any:
        n = client.chat.completions.create.call_count
        if n == 2:
            raise RuntimeError("transient outage")
        return _mk_response("Alice")

    client.chat.completions.create.side_effect = flaky_create
    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
    )
    # Four "Alice" samples, one failure → "Alice" still wins.
    assert result.chosen == "Alice"
    assert len(result.samples) == 4


# ── answer_with_self_consistency — judge_pick ────────────────────────


@pytest.mark.unit
def test_judge_pick_calls_judge_and_returns_choice() -> None:
    answerer_client = _client_returning(["Alice", "Bob", "Carol", "Dan", "Eve"])
    judge_client = MagicMock()
    judge_client.chat.completions.create.return_value = _mk_response("3")

    result = answer_with_self_consistency(
        client=answerer_client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
        voter="judge_pick",
        judge_client=judge_client,
        judge_model="judge/test",
    )
    # Judge picked sample index 3 → "Dan"
    assert result.chosen == "Dan"
    assert result.voter == "judge_pick"
    judge_client.chat.completions.create.assert_called_once()


@pytest.mark.unit
def test_judge_pick_falls_back_to_majority_on_judge_failure() -> None:
    answerer_client = _client_returning(["Alice", "Alice", "Alice", "Bob", "Carol"])
    judge_client = MagicMock()
    judge_client.chat.completions.create.side_effect = RuntimeError("judge offline")

    result = answer_with_self_consistency(
        client=answerer_client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
        voter="judge_pick",
        judge_client=judge_client,
        judge_model="judge/test",
    )
    # Falls back to majority: 3x Alice wins.
    assert result.chosen == "Alice"
    assert result.voter == "majority"  # fallback voter is recorded


@pytest.mark.unit
def test_judge_pick_without_judge_client_uses_answerer_client() -> None:
    """When judge_client is None, we reuse the answerer client (and
    then the K+1th call is the judge call)."""
    # K=3 sample answers + 1 judge response.
    client = _client_returning(["Alice", "Bob", "Carol", "1"])

    result = answer_with_self_consistency(
        client=client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=3,
        temperature=0.7,
        voter="judge_pick",
        judge_client=None,
        judge_model="judge/test",
    )
    # Judge picked index 1 → "Bob"
    assert result.chosen == "Bob"
    assert result.voter == "judge_pick"


@pytest.mark.unit
def test_judge_pick_invalid_index_falls_back_to_majority() -> None:
    """Judge returning a non-numeric or out-of-range answer should
    fall back to majority instead of crashing."""
    answerer_client = _client_returning(["Alice", "Alice", "Alice", "Bob", "Carol"])
    judge_client = MagicMock()
    judge_client.chat.completions.create.return_value = _mk_response("not a number")

    result = answer_with_self_consistency(
        client=answerer_client,
        model="test/model",
        messages=[{"role": "user", "content": "q"}],
        k=5,
        temperature=0.7,
        voter="judge_pick",
        judge_client=judge_client,
        judge_model="judge/test",
    )
    assert result.chosen == "Alice"
    assert result.voter == "majority"


# ── ConsistencyResult shape ──────────────────────────────────────────


@pytest.mark.unit
def test_consistency_result_is_frozen_dataclass() -> None:
    """ConsistencyResult is immutable per coding-style.md."""
    result = ConsistencyResult(
        samples=["a", "b"], chosen="a", voter="majority", vote_breakdown={"a": 2},
    )
    with pytest.raises((AttributeError, Exception)):  # frozen dataclass error
        result.chosen = "x"  # type: ignore[misc]


@pytest.mark.unit
def test_invalid_voter_raises() -> None:
    """An unrecognized voter strategy is a programming error — surface
    it loudly rather than silently falling through."""
    client = _client_returning(["a", "b", "c"])
    with pytest.raises(ValueError, match="voter"):
        answer_with_self_consistency(
            client=client,
            model="test/model",
            messages=[{"role": "user", "content": "q"}],
            k=3,
            temperature=0.7,
            voter="not_a_real_strategy",
        )
