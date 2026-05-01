"""Phase 9.4 — AbstentionBench tests.

Pure unit tests. No dataset, no DB, no LLM. Verifies detector phrasings,
confusion-matrix math, F1 calibration extremes, and runner error
isolation.
"""

from __future__ import annotations


import pytest

from evals.abstention.detector import (
    is_abstention, make_detector,
)
from evals.abstention.runner import (
    DefaultDatasetLoader, _run_one, run, summarize,
)
from evals.abstention.scorer import (
    _answer_matches, aggregate, score_prediction,
)
from evals.abstention.types import (
    AbstentionMetrics, AbstentionPrediction, AbstentionRunReport,
    AbstentionSample,
)


# ── Detector ──────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_detector_catches_canonical_chain_of_note_phrase() -> None:
    """The CoN ABSTAIN clause says exactly this phrase — must be caught."""
    assert is_abstention("I don't have that information") is True


@pytest.mark.unit
def test_detector_catches_common_variants() -> None:
    cases = [
        "I don't have any information about that.",
        "I don't have enough context to answer.",
        "No relevant information is available.",
        "The memories do not contain that detail.",
        "I cannot answer that based on the provided memories.",
        "Insufficient information to determine.",
        "Unable to recall any matching memory.",
        "I don't know.",
    ]
    for c in cases:
        assert is_abstention(c) is True, f"missed: {c!r}"


@pytest.mark.unit
def test_detector_does_not_falsepositive_on_hedge_then_answer() -> None:
    """Common failure mode: hedged-but-substantive answers get flagged
    as abstentions. The detector must NOT do that."""
    cases = [
        "I'm not sure, but based on memory mem_3, your favorite color is blue.",
        "I don't know yet, however the meeting was on Tuesday per mem_7.",
        "Based on mem_2, I think your dog is named Rex.",
        "Your favorite food is sushi [mem_1].",
    ]
    for c in cases:
        assert is_abstention(c) is False, f"false positive on: {c!r}"


@pytest.mark.unit
def test_detector_treats_empty_response_as_abstention() -> None:
    """Empty model output = effective abstention."""
    assert is_abstention("") is True
    assert is_abstention("   \n  ") is True


@pytest.mark.unit
def test_make_detector_appends_patterns() -> None:
    custom = make_detector(extra_patterns=(r"\bN/A\b",))
    # Default detector wouldn't catch "N/A" alone
    assert is_abstention("N/A") is False
    # Custom detector does
    assert custom("N/A") is True
    # Still catches defaults
    assert custom("I don't have that information") is True


# ── _answer_matches ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_answer_matches_substring_normalized() -> None:
    assert _answer_matches("Your favorite color is BLUE.", "blue") is True
    assert _answer_matches("It's red.", "blue") is False


@pytest.mark.unit
def test_answer_matches_empty_expected_does_not_crash() -> None:
    """Data-quality issue, but the scorer must not blow up."""
    assert _answer_matches("anything", "") is True
    assert _answer_matches("anything", None) is True


# ── score_prediction ─────────────────────────────────────────────────────


def _sample(*, sid="s1", answerable=True, expected=None,
            cat="general") -> AbstentionSample:
    return AbstentionSample(
        sample_id=sid, context="...", query="?",
        answerable=answerable, expected_answer=expected, category=cat,
    )


def _pred(*, text="answer", error=None, sid="s1") -> AbstentionPrediction:
    return AbstentionPrediction(
        sample_id=sid, response=text, error=error,
    )


@pytest.mark.unit
def test_score_unanswerable_with_abstention_correct() -> None:
    s = _sample(answerable=False)
    p = score_prediction(s, _pred(text="I don't have that information"))
    assert p.abstained is True and p.correct is True


@pytest.mark.unit
def test_score_unanswerable_with_answer_is_confabulation() -> None:
    """Model invented an answer when it should have abstained."""
    s = _sample(answerable=False)
    p = score_prediction(s, _pred(text="The capital is Atlantis."))
    assert p.abstained is False and p.correct is False


@pytest.mark.unit
def test_score_answerable_with_correct_answer_passes() -> None:
    s = _sample(answerable=True, expected="paris")
    p = score_prediction(s, _pred(text="The capital is Paris."))
    assert p.abstained is False and p.correct is True


@pytest.mark.unit
def test_score_answerable_with_abstention_is_over_abstention() -> None:
    s = _sample(answerable=True, expected="paris")
    p = score_prediction(s, _pred(text="I don't have that information"))
    assert p.abstained is True and p.correct is False


@pytest.mark.unit
def test_score_answerable_with_wrong_answer_fails() -> None:
    s = _sample(answerable=True, expected="paris")
    p = score_prediction(s, _pred(text="The capital is Berlin."))
    assert p.abstained is False and p.correct is False


@pytest.mark.unit
def test_score_prediction_with_error_is_wrong_not_abstention() -> None:
    """Runtime crash isn't an abstention — operator needs to see failures."""
    s = _sample(answerable=False)
    p = score_prediction(s, _pred(text="", error="db blew up"))
    assert p.correct is False
    assert p.abstained is False  # important: don't paper over crashes as wins


# ── _confusion + AbstentionMetrics ──────────────────────────────────────


@pytest.mark.unit
def test_metrics_perfect_calibration_f1_one() -> None:
    m = AbstentionMetrics(true_positive=5, true_negative=5)
    assert m.precision == 1.0 and m.recall == 1.0 and m.f1 == 1.0


@pytest.mark.unit
def test_metrics_always_abstain_collapses_to_zero_f1() -> None:
    """Always-abstain → recall=1, precision very low → low F1.
    With perfect-recall but lots of false-positives, F1 drops fast."""
    m = AbstentionMetrics(true_positive=5, false_positive=95)
    assert m.recall == 1.0
    assert m.precision == 0.05
    assert m.f1 == pytest.approx(2 * 1.0 * 0.05 / 1.05, abs=0.001)


@pytest.mark.unit
def test_metrics_always_answer_yields_zero_f1() -> None:
    """Always-answer → tp=0 → both precision and recall are 0 → F1=0."""
    m = AbstentionMetrics(true_negative=50, false_negative=50)
    assert m.f1 == 0.0


@pytest.mark.unit
def test_metrics_over_abstention_rate() -> None:
    """Of cases that should have answered, fraction that abstained."""
    m = AbstentionMetrics(true_positive=10, false_positive=5,
                          true_negative=15, false_negative=2)
    assert m.over_abstention_rate == pytest.approx(5 / (15 + 5))


@pytest.mark.unit
def test_metrics_confabulation_rate() -> None:
    """Of cases that should have abstained, fraction that answered."""
    m = AbstentionMetrics(true_positive=10, false_positive=5,
                          true_negative=15, false_negative=2)
    assert m.confabulation_rate == pytest.approx(2 / (10 + 2))


@pytest.mark.unit
def test_metrics_to_dict_round_trips() -> None:
    import json
    m = AbstentionMetrics(true_positive=1, true_negative=1)
    d = m.to_dict()
    json.dumps(d)
    assert d["f1"] == 1.0


# ── aggregate ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_aggregate_overall_confusion_matrix() -> None:
    """End-to-end: 2 unanswerable correctly abstained, 2 answerable
    correctly answered → perfect F1."""
    samples = [
        _sample(sid="u1", answerable=False),
        _sample(sid="u2", answerable=False),
        _sample(sid="a1", answerable=True, expected="blue"),
        _sample(sid="a2", answerable=True, expected="red"),
    ]
    preds = [
        _pred(sid="u1", text="I don't have that information"),
        _pred(sid="u2", text="No relevant information is available"),
        _pred(sid="a1", text="Your favorite is blue"),
        _pred(sid="a2", text="It's red"),
    ]
    rep = aggregate(samples, preds)
    assert rep.overall.f1 == 1.0
    assert rep.answer_accuracy == 1.0
    assert rep.answer_total == 2 and rep.answer_correct == 2


@pytest.mark.unit
def test_aggregate_per_category_breakdown() -> None:
    samples = [
        _sample(sid="a", answerable=False, cat="unknown_topic"),
        _sample(sid="b", answerable=False, cat="false_premise"),
    ]
    preds = [
        _pred(sid="a", text="I don't have that information"),
        _pred(sid="b", text="The Atlantis capital is..."),  # confabulation
    ]
    rep = aggregate(samples, preds)
    assert rep.by_category["unknown_topic"].f1 == 1.0
    # "false_premise" has 0 tp + 1 fn → recall=0 → f1=0
    assert rep.by_category["false_premise"].f1 == 0.0


@pytest.mark.unit
def test_aggregate_records_missing_prediction_as_wrong() -> None:
    samples = [
        _sample(sid="present", answerable=False),
        _sample(sid="missing", answerable=False),
    ]
    preds = [_pred(sid="present", text="I don't have that information")]
    rep = aggregate(samples, preds)
    miss = next(p for p in rep.predictions if p.sample_id == "missing")
    assert miss.error == "missing prediction"
    assert miss.correct is False


@pytest.mark.unit
def test_aggregate_answer_accuracy_excludes_abstentions() -> None:
    """answer_accuracy is over cases where the model attempted, not all."""
    samples = [
        _sample(sid="a", answerable=True, expected="blue"),
        _sample(sid="b", answerable=True, expected="red"),
    ]
    preds = [
        _pred(sid="a", text="blue"),                                 # attempted, right
        _pred(sid="b", text="I don't have that information"),        # abstained
    ]
    rep = aggregate(samples, preds)
    assert rep.answer_total == 1   # only one attempted
    assert rep.answer_correct == 1
    assert rep.answer_accuracy == 1.0


# ── _run_one (per-sample isolation) ──────────────────────────────────────


class _FakeMem:
    closed = False

    def close(self) -> None:
        self.__class__.closed = True


@pytest.mark.unit
def test_run_one_records_ingest_error() -> None:
    def bad_ingest(s, m): raise RuntimeError("ingest exploded")
    def answer(s, m): return ""

    p = _run_one(_sample(sid="x"),
                 mem_factory=lambda: _FakeMem(),
                 ingest=bad_ingest, answer=answer)
    assert p.error and "RuntimeError" in p.error


@pytest.mark.unit
def test_run_one_closes_mem_on_error() -> None:
    _FakeMem.closed = False

    def good_ingest(s, m): pass
    def bad_answer(s, m): raise ValueError("boom")

    _run_one(_sample(sid="x"),
             mem_factory=lambda: _FakeMem(),
             ingest=good_ingest, answer=bad_answer)
    assert _FakeMem.closed is True


# ── DefaultDatasetLoader ────────────────────────────────────────────────


@pytest.mark.unit
def test_default_loader_fails_loud() -> None:
    with pytest.raises(NotImplementedError, match="not configured"):
        DefaultDatasetLoader()()


# ── End-to-end run() ────────────────────────────────────────────────────


@pytest.mark.unit
def test_run_end_to_end_with_perfect_stub() -> None:
    """Every unanswerable abstains; every answerable answers correctly."""
    samples = [
        _sample(sid="u", answerable=False),
        _sample(sid="a", answerable=True, expected="paris"),
    ]

    def loader(): return samples
    def ingest(s, m): pass
    def answer(s, m):
        if not s.answerable:
            return "I don't have that information"
        return "The capital is Paris"

    summary = run(
        mem_factory=lambda: _FakeMem(),
        ingest=ingest, answer=answer, loader=loader,
    )
    assert summary.benchmark == "abstention"
    assert summary.primary_metric == 100.0
    assert summary.primary_metric_name == "abstention_f1_pct"


@pytest.mark.unit
def test_run_sample_limit_truncates() -> None:
    samples = [_sample(sid=f"s{i}", answerable=False) for i in range(10)]

    def loader(): return samples
    def ingest(s, m): pass
    def answer(s, m): return "I don't have that information"

    summary = run(
        mem_factory=lambda: _FakeMem(),
        ingest=ingest, answer=answer, loader=loader, sample_limit=3,
    )
    assert summary.total == 3


# ── summarize ────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_summarize_emits_f1_as_primary_metric_pct() -> None:
    rep = AbstentionRunReport(
        overall=AbstentionMetrics(true_positive=5, true_negative=5),
        by_category={
            "unknown_topic": AbstentionMetrics(true_positive=5),
        },
        answer_accuracy=1.0, answer_correct=5, answer_total=5,
    )
    s = summarize(rep)
    assert s.benchmark == "abstention"
    assert s.primary_metric == 100.0
    assert s.primary_metric_name == "abstention_f1_pct"
    assert s.per_category["unknown_topic"] == 100.0


@pytest.mark.unit
def test_summarize_includes_subsidiary_rates_in_metadata() -> None:
    """Operators inspecting failures need precision/recall and the
    over-abstention / confabulation breakdown — keep them in metadata."""
    rep = AbstentionRunReport(
        overall=AbstentionMetrics(
            true_positive=10, false_positive=5,
            true_negative=15, false_negative=2,
        ),
    )
    s = summarize(rep)
    assert "over_abstention_rate" in s.metadata
    assert "confabulation_rate" in s.metadata
    assert "precision" in s.metadata and "recall" in s.metadata
