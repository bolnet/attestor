"""Unit tests for attestor.longmemeval_critique.

Pure unit tests — every LLM call is mocked. End-to-end behavior with
the real LongMemEval flow is exercised via the bench harness, not here.

Coverage shape mirrors `tests/test_self_consistency.py` (the closest
sibling, shipped in PR #97).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from attestor.longmemeval_critique import (
    _parse_fix,
    _parse_reason,
    _parse_verdict,
    _question_from_messages,
    answer_with_critique_revise,
)


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


def _make_client(responses: list[str]) -> Any:
    """Build a stub OpenAI-compatible client whose chat.completions.create
    returns each response string in order. Records every call for
    introspection."""
    seq = iter(responses)

    def _create(**kwargs):
        try:
            content = next(seq)
        except StopIteration:
            raise AssertionError(
                f"client received unexpected extra call: {kwargs.get('model')}"
            )
        msg = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=msg)
        # Shape is what `traced_create` and the existing call sites expect.
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=20, total_tokens=30,
        )
        return SimpleNamespace(
            id="resp-test",
            choices=[choice],
            usage=usage,
            model=kwargs.get("model", "test/model"),
        )

    completions = SimpleNamespace(create=MagicMock(side_effect=_create))
    chat = SimpleNamespace(completions=completions)
    client = SimpleNamespace(chat=chat)
    client.chat.completions.create.responses = responses
    return client


def _msg(role: str, content: str) -> dict[str, Any]:
    return {"role": role, "content": content}


# ──────────────────────────────────────────────────────────────────────
# _parse_verdict
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_verdict_pass() -> None:
    assert _parse_verdict("VERDICT: pass\nREASON: looks good") == "pass"


@pytest.mark.unit
def test_parse_verdict_revise() -> None:
    assert _parse_verdict("VERDICT: revise\nREASON: wrong year") == "revise"


@pytest.mark.unit
def test_parse_verdict_case_insensitive() -> None:
    assert _parse_verdict("verdict: PASS") == "pass"
    assert _parse_verdict("VERDICT: REVISE") == "revise"
    assert _parse_verdict("Verdict: Revise") == "revise"


@pytest.mark.unit
def test_parse_verdict_defaults_to_pass_when_unparsable() -> None:
    """If we can't tell what the critic said, never make things worse —
    default to pass (no revision)."""
    assert _parse_verdict("hmm, looks ok-ish") == "pass"
    assert _parse_verdict("") == "pass"
    assert _parse_verdict("VERDICT: maybe") == "pass"


@pytest.mark.unit
def test_parse_verdict_handles_extra_whitespace() -> None:
    assert _parse_verdict("   VERDICT:   revise   ") == "revise"
    assert _parse_verdict("\n\nVERDICT: pass\n") == "pass"


# ──────────────────────────────────────────────────────────────────────
# _parse_reason / _parse_fix
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_reason_extracts_value() -> None:
    text = "VERDICT: revise\nREASON: wrong year, not 2021"
    assert "wrong year" in _parse_reason(text).lower()


@pytest.mark.unit
def test_parse_reason_returns_empty_when_missing() -> None:
    assert _parse_reason("VERDICT: pass") == ""


@pytest.mark.unit
def test_parse_fix_extracts_value() -> None:
    text = "VERDICT: revise\nREASON: wrong\nFIX: should be 2022"
    assert "2022" in _parse_fix(text)


@pytest.mark.unit
def test_parse_fix_returns_empty_when_absent() -> None:
    assert _parse_fix("VERDICT: pass\nREASON: ok") == ""


# ──────────────────────────────────────────────────────────────────────
# _question_from_messages
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_question_from_messages_picks_last_user() -> None:
    messages = [
        _msg("system", "You are an answerer."),
        _msg("user", "first question?"),
        _msg("assistant", "previous answer"),
        _msg("user", "what's the actual question"),
    ]
    assert _question_from_messages(messages) == "what's the actual question"


@pytest.mark.unit
def test_question_from_messages_falls_back_to_assistant() -> None:
    """When no user message exists, the helper falls back to the most
    recent non-system content rather than returning empty — better
    than silently dropping context for the critic."""
    messages = [_msg("system", "answerer."), _msg("assistant", "hello")]
    assert _question_from_messages(messages) == "hello"


@pytest.mark.unit
def test_question_from_messages_handles_empty_list() -> None:
    """Empty messages → a sentinel string so the critic prompt template
    doesn't crash on empty interpolation."""
    assert _question_from_messages([]) == "(no question available)"


# ──────────────────────────────────────────────────────────────────────
# answer_with_critique_revise — end-to-end (mocked)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_critique_pass_returns_initial_unchanged() -> None:
    """VERDICT=pass → no revise call, final == initial."""
    client = _make_client([
        "Initial answer: it was Bob.",
        "VERDICT: pass\nREASON: matches the context exactly.",
    ])
    result = answer_with_critique_revise(
        client=client,
        model="test/answerer",
        messages=[_msg("user", "who was the CTO?")],
        critic_client=client,
        critic_model="test/critic",
        context="Bob became CTO on Monday.",
    )
    assert result.initial_answer == "Initial answer: it was Bob."
    assert result.final_answer == result.initial_answer
    assert result.revised is False
    assert result.n_revisions == 0
    assert "matches the context" in result.critique
    # 2 calls only — no revise
    assert client.chat.completions.create.call_count == 2


@pytest.mark.unit
def test_critique_revise_replaces_initial() -> None:
    """VERDICT=revise → reviser fires; final_answer = revised text."""
    client = _make_client([
        "Initial answer: Alice.",                                    # initial
        "VERDICT: revise\nREASON: wrong person\nFIX: should be Bob.", # critique
        "Corrected answer: Bob took over as CTO on Monday.",          # revise
    ])
    result = answer_with_critique_revise(
        client=client,
        model="test/answerer",
        messages=[_msg("user", "who was the new CTO?")],
        critic_client=client,
        critic_model="test/critic",
        revise_client=client,
        revise_model="test/reviser",
        context="Bob became CTO on Monday.",
    )
    assert result.initial_answer == "Initial answer: Alice."
    assert "Bob" in result.final_answer
    assert result.revised is True
    assert result.n_revisions == 1
    assert client.chat.completions.create.call_count == 3


@pytest.mark.unit
def test_critique_malformed_output_defaults_to_pass() -> None:
    """Garbled critique → no revise; degrades safely to initial."""
    client = _make_client([
        "Initial answer: it was Bob.",
        "lorem ipsum dolor sit amet — no verdict line",
    ])
    result = answer_with_critique_revise(
        client=client,
        model="test/answerer",
        messages=[_msg("user", "q?")],
        critic_client=client,
        critic_model="test/critic",
        context="ctx",
    )
    assert result.final_answer == result.initial_answer
    assert result.revised is False
    assert result.n_revisions == 0


@pytest.mark.unit
def test_critique_handles_critic_failure() -> None:
    """If the critic LLM raises, return CritiqueResult with initial as
    final and revised=False — never escape the exception."""
    answerer_client = _make_client(["Initial answer: it was Bob."])
    critic_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(side_effect=RuntimeError("critic down")),
            ),
        ),
    )
    result = answer_with_critique_revise(
        client=answerer_client,
        model="test/answerer",
        messages=[_msg("user", "q?")],
        critic_client=critic_client,
        critic_model="test/critic",
        context="ctx",
    )
    assert result.final_answer == "Initial answer: it was Bob."
    assert result.revised is False
    assert result.n_revisions == 0


@pytest.mark.unit
def test_critique_handles_reviser_failure() -> None:
    """If the reviser LLM raises during revision, fall back to initial.
    No exception escapes; n_revisions stays 0."""
    initial_client = _make_client([
        "Initial answer: Alice.",
        "VERDICT: revise\nREASON: wrong\nFIX: Bob",
    ])
    reviser_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(side_effect=RuntimeError("reviser down")),
            ),
        ),
    )
    result = answer_with_critique_revise(
        client=initial_client,
        model="test/answerer",
        messages=[_msg("user", "q?")],
        critic_client=initial_client,
        critic_model="test/critic",
        revise_client=reviser_client,
        revise_model="test/reviser",
        context="ctx",
    )
    assert result.initial_answer == "Initial answer: Alice."
    assert result.final_answer == result.initial_answer
    assert result.revised is False
    assert result.n_revisions == 0


@pytest.mark.unit
def test_critique_handles_initial_answer_failure() -> None:
    """If the initial answer call itself raises, return an empty
    CritiqueResult — never let it escape."""
    bad_client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(side_effect=RuntimeError("answerer down")),
            ),
        ),
    )
    result = answer_with_critique_revise(
        client=bad_client,
        model="test/answerer",
        messages=[_msg("user", "q?")],
        critic_client=bad_client,
        critic_model="test/critic",
        context="ctx",
    )
    assert result.initial_answer == ""
    assert result.final_answer == ""
    assert result.revised is False
    assert result.n_revisions == 0


@pytest.mark.unit
def test_critique_max_revisions_capped_at_one() -> None:
    """max_revisions > 1 is silently clamped to 1 in the call site
    (the YAML loader is the public-facing rejection path; this test
    asserts the defensive clamp catches anyone bypassing the loader)."""
    client = _make_client([
        "Initial.",
        "VERDICT: revise\nREASON: wrong\nFIX: better",
        "Revised once.",
    ])
    result = answer_with_critique_revise(
        client=client,
        model="test/answerer",
        messages=[_msg("user", "q?")],
        critic_client=client,
        critic_model="test/critic",
        max_revisions=999,   # caller bypassed the loader; we must clamp
        context="ctx",
    )
    # Exactly one revision happened, not 999
    assert result.n_revisions == 1
    assert client.chat.completions.create.call_count == 3


@pytest.mark.unit
def test_critique_uses_distinct_critic_and_revise_models() -> None:
    """Verify the model arg actually flows through to each call."""
    seen_models: list[str] = []

    def _create(**kwargs):
        seen_models.append(kwargs["model"])
        # Always verdict=revise so we exercise all three calls
        if len(seen_models) == 1:
            content = "Initial."
        elif len(seen_models) == 2:
            content = "VERDICT: revise\nREASON: wrong\nFIX: x"
        else:
            content = "Revised."
        msg = SimpleNamespace(content=content)
        usage = SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2,
        )
        return SimpleNamespace(
            id="r", choices=[SimpleNamespace(message=msg)],
            usage=usage, model=kwargs["model"],
        )

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(
                create=MagicMock(side_effect=_create),
            ),
        ),
    )
    answer_with_critique_revise(
        client=client,
        model="test/answerer",
        messages=[_msg("user", "q?")],
        critic_client=client,
        critic_model="test/critic",
        revise_client=client,
        revise_model="test/reviser",
        context="ctx",
    )
    assert seen_models == ["test/answerer", "test/critic", "test/reviser"]
