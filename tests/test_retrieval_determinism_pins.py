"""Regression pins for retrieval-side determinism + scoring invariants.

Each pin in this file locks in BEHAVIOR THAT IS CORRECT TODAY but is
unprotected against silent drift in a future refactor. Companion file to
``tests/test_temporal_supersession_gaps.py`` (which guards the
supersession/temporal layer); these guards focus on the recall hot path.

Pins covered:

  Pin 1 — HyDE generator pins ``temperature=0`` on every LLM call (sync
          + async). Audit invariant A7: deterministic HyDE recall depends
          on temp=0; a refactor that drops the kwarg silently breaks
          LME-S reproducibility.
  Pin 2 — HyDE prompt is event-descriptive (not answer-shaped). The
          specific phrasing yielded the +9pp recall lift on LME-S
          temporal; a quiet copy edit silently regresses the lever.
  Pin 3 — ``_blend_score`` formula is
          ``vector_weight*vec + graph_weight*max(0, bonus)`` with an
          additive ``+ graph_unreachable_penalty`` when ``hop is None``.
          A sign flip or a dropped penalty branch ships green today.
  Pin 4 — ``fit_to_budget`` is greedy, NOT abort-on-first-oversize. A
          long memory in the middle of the ranked list must skip past,
          not terminate the loop and drop later-but-smaller hits.

If any pin fails, the failure indicates a real regression in production
behavior — STOP and report rather than relax the assertion.
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest

from attestor.models import Memory, RetrievalResult
from attestor.retrieval import hyde as hyde_module
from attestor.retrieval.hyde import (
    HydeResult,
    _GENERATOR_PROMPT,
    generate_hypothetical_answer,
    generate_hypothetical_answer_async,
)
from attestor.retrieval.orchestrator.config import RetrievalRuntimeConfig
from attestor.retrieval.orchestrator.helpers import _OrchestratorHelpersMixin
from attestor.retrieval.scorer import fit_to_budget


# ──────────────────────────────────────────────────────────────────────
# Pin 1 — HyDE generator pins temperature=0 on every LLM call.
# Sync + async paths both must pass temperature=0.0 to the underlying
# chat.completions.create. Audit invariant A7.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_pin1_hyde_sync_generator_pins_temperature_zero(monkeypatch) -> None:
    """``generate_hypothetical_answer`` must pass ``temperature=0.0`` to
    the LLM client. Captures the kwargs the sync path sends and asserts
    on the temperature field directly — an assertion that survives a
    refactor that renames the wrapper but keeps the kwarg.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content="Fake hypothetical."))
    ]
    fake_response.id = "x"
    fake_response.model = "test/m"
    fake_response.usage = {
        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
    }

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    with patch("openai.OpenAI", return_value=fake_client):
        generate_hypothetical_answer(
            "who is the cto?", model="test/m", api_key="fake-test-key",
        )

    assert fake_client.chat.completions.create.called, (
        "expected the LLM client to be called once"
    )
    kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert "temperature" in kwargs, (
        "HyDE sync generator dropped the temperature kwarg — "
        "audit invariant A7 (deterministic HyDE) requires temperature=0"
    )
    assert kwargs["temperature"] == 0.0, (
        f"HyDE sync generator must pin temperature=0.0; got "
        f"{kwargs['temperature']!r}"
    )


@pytest.mark.unit
def test_pin1_hyde_async_generator_pins_temperature_zero(monkeypatch) -> None:
    """``generate_hypothetical_answer_async`` must pass ``temperature=0.0``
    to the AsyncOpenAI client. Async amplifies non-determinism risk
    because gathered lanes can observe different hypotheticals across
    runs if T > 0; this pin makes the silent regression loud.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "fake-test-key")

    fake_response = MagicMock()
    fake_response.choices = [
        MagicMock(message=MagicMock(content="Fake hypothetical."))
    ]
    fake_response.id = "x"
    fake_response.model = "test/m"
    fake_response.usage = {
        "prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2,
    }

    captured: dict = {}

    async def fake_create(**kwargs):
        captured.update(kwargs)
        return fake_response

    fake_async_client = MagicMock()
    fake_async_client.chat.completions.create = fake_create

    # Use a model id WITHOUT a ``provider/`` prefix so the async path
    # falls through to ``pool.default_strategy()`` instead of the
    # "unknown provider → degraded" branch (which would silently bail
    # before hitting the LLM and bypass this pin entirely).
    asyncio.run(
        generate_hypothetical_answer_async(
            "who is the cto?",
            model="gpt-4",
            api_key="fake-test-key",
            client=fake_async_client,
        )
    )

    assert "temperature" in captured, (
        "HyDE async generator dropped the temperature kwarg — "
        "audit invariant A7 (deterministic HyDE) requires temperature=0"
    )
    assert captured["temperature"] == 0.0, (
        f"HyDE async generator must pin temperature=0.0; got "
        f"{captured['temperature']!r}"
    )


# ──────────────────────────────────────────────────────────────────────
# Pin 2 — HyDE prompt is event-descriptive (not answer-shaped).
# The exact prompt produced the +9pp recall lift on LME-S temporal; a
# "helpful" rewrite that turns it into an answer-prediction prompt
# silently regresses the lever. Pin specific distinctive phrases that
# such a rewrite would lose.
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize(
    "phrase",
    [
        # Frames the task as event description, not question answering.
        # Capitalisation matters — the all-caps form is the signal that
        # tells the model "match the source-turn surface form".
        "ORIGINALLY MENTIONED",
        # The semantic-asymmetry rationale — central to the lever.
        "Question-shape and answer-shape don't embed close to source-shape",
        # Length and tone constraint that keeps embeddings tight.
        "1-2 sentences, first-person, declarative, conversational",
        # Anti-pattern guard — bans answer-shape tokens like "X weeks ago".
        "do NOT say",
    ],
)
def test_pin2_hyde_prompt_is_event_descriptive(phrase: str) -> None:
    """The HyDE generator prompt must keep its event-descriptive framing.

    Each phrase encodes a load-bearing piece of the prompt design. A
    well-meaning copy edit that reframes the task as "answer the
    question" or drops the asymmetry rationale will lose the LME-S
    temporal lift.
    """
    assert phrase in _GENERATOR_PROMPT, (
        f"HyDE generator prompt lost the marker {phrase!r} — this "
        "phrase is part of the event-descriptive framing that produces "
        "the +9pp LME-S temporal recall lift. Restore the phrase or "
        "update this pin (and re-bench LME-S temporal first)."
    )


@pytest.mark.unit
def test_pin2_hyde_prompt_includes_question_substitution_token() -> None:
    """The prompt template must keep the ``{question}`` substitution
    token so ``_GENERATOR_PROMPT.format(question=...)`` keeps working.
    """
    assert "{question}" in _GENERATOR_PROMPT, (
        "HyDE generator prompt must contain the {question} placeholder"
    )


@pytest.mark.unit
def test_pin2_hyde_prompt_keeps_examples() -> None:
    """The few-shot examples are part of the prompt that drives the
    +9pp lift. Pin the section header so a future "trim the examples"
    refactor surfaces here before LME-S regresses.
    """
    assert "Examples:" in _GENERATOR_PROMPT, (
        "HyDE generator prompt lost the few-shot Examples block — the "
        "examples are load-bearing for the +9pp LME-S temporal lift"
    )


# ──────────────────────────────────────────────────────────────────────
# Pin 3 — ``_blend_score`` formula.
#
# Formula (helpers.py:147-166):
#   vec_norm = clamp(vector_sim, 0, 1)
#   bonus    = graph_unreachable_penalty if hop is None else
#              graph_affinity_bonus.get(hop, 0.0)
#   final    = vector_weight * vec_norm + graph_weight * max(0, bonus)
#   if hop is None: final += graph_unreachable_penalty
#
# Defaults: vector_weight=0.7, graph_weight=0.3,
#           bonus={0:0.30, 1:0.20, 2:0.10}, unreachable=-0.05.
# ──────────────────────────────────────────────────────────────────────


class _Blender(_OrchestratorHelpersMixin):
    """Bare carrier for ``_blend_score`` so we can test the helper
    without standing up the full orchestrator (which requires a live
    Postgres). Only ``self.config`` is read by ``_blend_score``.
    """

    def __init__(self, config: RetrievalRuntimeConfig) -> None:
        self.config = config


@pytest.mark.unit
@pytest.mark.parametrize(
    "vector_sim,hop,expected_final,expected_bonus",
    [
        # hop=0 (entity itself): final = 0.7*1.0 + 0.3*0.30 = 0.79
        (1.0, 0, 0.79, 0.30),
        # hop=1: final = 0.7*0.5 + 0.3*0.20 = 0.41
        (0.5, 1, 0.41, 0.20),
        # hop=2: final = 0.7*0.8 + 0.3*0.10 = 0.59
        (0.8, 2, 0.59, 0.10),
        # hop unmapped (e.g. 99): bonus=0.0, final = 0.7*0.6 = 0.42
        (0.6, 99, 0.42, 0.0),
        # hop=None: bonus=-0.05; final = 0.7*1.0 + 0 + (-0.05) = 0.65
        (1.0, None, 0.65, -0.05),
        # vector_sim out-of-range (clamped to 1.0) at hop=0:
        # final = 0.7*1.0 + 0.3*0.30 = 0.79
        (1.5, 0, 0.79, 0.30),
        # negative vector_sim clamped to 0.0 at hop=0:
        # final = 0.7*0 + 0.3*0.30 = 0.09
        (-0.2, 0, 0.09, 0.30),
    ],
)
def test_pin3_blend_score_formula(
    vector_sim: float,
    hop: int | None,
    expected_final: float,
    expected_bonus: float,
) -> None:
    """``_blend_score`` must compute the exact documented formula across
    representative (vector_sim, hop) coordinates. A sign flip on the
    unreachable-penalty addition or a dropped ``max(0, bonus)`` clamp
    will surface here.
    """
    cfg = RetrievalRuntimeConfig()  # defaults match historical literals
    blender = _Blender(cfg)
    final, bonus = blender._blend_score(vector_sim, hop)
    assert final == pytest.approx(expected_final, abs=1e-9), (
        f"blend score regression: vector_sim={vector_sim} hop={hop} "
        f"expected final={expected_final}; got {final}"
    )
    assert bonus == pytest.approx(expected_bonus, abs=1e-9), (
        f"blend bonus regression: vector_sim={vector_sim} hop={hop} "
        f"expected bonus={expected_bonus}; got {bonus}"
    )


@pytest.mark.unit
def test_pin3_blend_score_unreachable_penalty_is_additive() -> None:
    """When ``hop is None``, the formula adds ``graph_unreachable_penalty``
    to ``final``. With a custom config we can isolate the additive branch
    from the ``max(0, bonus)`` clamp and confirm both fire.
    """
    cfg = RetrievalRuntimeConfig(
        vector_weight=1.0,
        graph_weight=0.0,
        graph_unreachable_penalty=-0.10,
    )
    blender = _Blender(cfg)
    # vec_norm=0.5, bonus=-0.10 (clamped to 0 inside the weighted sum),
    # then +(-0.10) from the additive penalty branch.
    final, bonus = blender._blend_score(0.5, None)
    assert bonus == pytest.approx(-0.10, abs=1e-9)
    assert final == pytest.approx(0.40, abs=1e-9), (
        "expected 0.5*1.0 + 0 + (-0.10) = 0.40; the additive "
        "unreachable-penalty branch may have been dropped"
    )


# ──────────────────────────────────────────────────────────────────────
# Pin 4 — ``fit_to_budget`` is greedy.
# A large memory in the middle of the ranked list must NOT terminate
# the packing loop; later, smaller memories that fit must still be
# selected.
# ──────────────────────────────────────────────────────────────────────


def _result(content: str, score: float) -> RetrievalResult:
    """Build a RetrievalResult with deterministic content for token
    estimation. ``estimate_tokens`` returns ``int(len(text.split()) * 1.3)``
    so a single-token "tiny" string is 1 token and 200 space-separated
    tokens is 260 estimated tokens.
    """
    mem = Memory(content=content, category="test")
    return RetrievalResult(memory=mem, score=score, match_source="vector")


@pytest.mark.unit
def test_pin4_fit_to_budget_skips_oversize_and_keeps_packing() -> None:
    """Packing pattern:
        scores  100,   90,   80,   70
        sizes   1tok,  1tok, 200wd, 1tok
        budget  10 tokens

    Highest score 100 (1 tok) → fits → tokens=1
    Next 90 (1 tok)            → fits → tokens=2
    Next 80 (≈260 tok)         → doesn't fit; selected non-empty → SKIP
    Next 70 (1 tok)            → fits → tokens=3

    Expected: 3 of 4 selected; the HUGE memory is skipped. If the loop
    aborts (break) on the oversize miss, the test fails because the
    final tiny memory is dropped.
    """
    huge_text = " ".join(["word"] * 200)  # ≈260 estimated tokens
    results = [
        _result("first", score=100.0),
        _result("second", score=90.0),
        _result(huge_text, score=80.0),
        _result("fourth", score=70.0),
    ]
    selected = fit_to_budget(results, token_budget=10)

    contents = [r.memory.content for r in selected]
    assert len(selected) == 3, (
        f"fit_to_budget regressed to abort-on-first-miss: expected 3 "
        f"selected (the 3 tinies, skipping the HUGE middle entry); "
        f"got {len(selected)} → {contents!r}"
    )
    assert contents == ["first", "second", "fourth"], (
        f"fit_to_budget must skip the oversize middle entry and pack "
        f"the remaining tiny entry; got {contents!r}"
    )


@pytest.mark.unit
def test_pin4_fit_to_budget_returns_at_least_one_when_all_oversize() -> None:
    """Documented edge case: if every candidate exceeds the budget, the
    function still returns the top-scored entry (the ``elif not selected``
    branch). Pin this so a future refactor that "tightens" the budget
    check doesn't silently start returning empty lists for tiny budgets.
    """
    huge_a = " ".join(["a"] * 200)
    huge_b = " ".join(["b"] * 200)
    results = [
        _result(huge_a, score=100.0),
        _result(huge_b, score=50.0),
    ]
    selected = fit_to_budget(results, token_budget=5)
    assert len(selected) == 1, (
        f"with all candidates oversize, fit_to_budget must still return "
        f"the top-scored result; got {len(selected)} entries"
    )
    assert selected[0].memory.content == huge_a, (
        "expected the highest-scored oversize entry to be returned"
    )
