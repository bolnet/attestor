"""Unit tests for the knowledge-updates supersession runner.

The runner depends on AgentMemory + DB, so end-to-end tests are
@pytest.mark.live. These cover the pure verdict-classification logic
and fixture loading.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.knowledge_updates.runner import (
    SuiteReport,
    _normalize,
    classify_top1,
    load_fixtures,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = REPO_ROOT / "evals" / "knowledge_updates" / "fixtures.json"


# ── _normalize ────────────────────────────────────────────────────────


@pytest.mark.unit
def test_normalize_lowercases_and_strips_punctuation() -> None:
    assert _normalize("Hello, World!") == "hello world"
    assert _normalize("$2,650") == "2650"
    assert _normalize("Dr. Khoury") == "dr khoury"


@pytest.mark.unit
def test_normalize_handles_extra_whitespace() -> None:
    assert _normalize("  multi   space\tinput\n") == "multi space input"


# ── classify_top1 ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_classify_top1_new_wins_on_match() -> None:
    """top1 mentions only the new fact → new_wins."""
    v = classify_top1(
        top1_content="Switched to matcha — coffee was making me anxious.",
        gold_answer="Matcha",
        stale_answer="Coffee",
    )
    # "matcha" is in body, but so is "coffee" — actually mentions both.
    # Let's make a cleaner test instead.
    assert v == "ambiguous"


@pytest.mark.unit
def test_classify_top1_clean_new_wins() -> None:
    v = classify_top1(
        top1_content="Daughter's birthday was on Sunday after we postponed.",
        gold_answer="Sunday",
        stale_answer="Saturday",
    )
    assert v == "new_wins"


@pytest.mark.unit
def test_classify_top1_clean_stale_wins() -> None:
    """top1 mentions only the old fact → stale_wins (the failure mode)."""
    v = classify_top1(
        top1_content="My daughter's birthday party is on Saturday.",
        gold_answer="Sunday",
        stale_answer="Saturday",
    )
    assert v == "stale_wins"


@pytest.mark.unit
def test_classify_top1_miss_when_neither_present() -> None:
    """top1 found something else → miss (retrieval didn't surface either fact)."""
    v = classify_top1(
        top1_content="Random unrelated content about astronomy.",
        gold_answer="Sunday",
        stale_answer="Saturday",
    )
    assert v == "miss"


@pytest.mark.unit
def test_classify_top1_miss_on_empty_top1() -> None:
    """No retrieval hit → miss."""
    v = classify_top1(
        top1_content=None,
        gold_answer="Sunday",
        stale_answer="Saturday",
    )
    assert v == "miss"


@pytest.mark.unit
def test_classify_top1_handles_dollar_amounts() -> None:
    v = classify_top1(
        top1_content="Landlord raised my rent to $2,650 starting May 1st.",
        gold_answer="$2,650",
        stale_answer="$2,400",
    )
    assert v == "new_wins"


@pytest.mark.unit
def test_classify_top1_handles_multi_word_entities() -> None:
    v = classify_top1(
        top1_content="Bob Patel took over as CTO this Monday.",
        gold_answer="Bob Patel",
        stale_answer="Alice Chen",
    )
    assert v == "new_wins"


# ── load_fixtures ─────────────────────────────────────────────────────


@pytest.mark.unit
def test_load_fixtures_returns_50_cases() -> None:
    cases = load_fixtures(FIXTURES)
    assert len(cases) == 50


@pytest.mark.unit
def test_load_fixtures_all_unique_ids() -> None:
    cases = load_fixtures(FIXTURES)
    ids = [c["case_id"] for c in cases]
    assert len(set(ids)) == len(ids)


@pytest.mark.unit
def test_load_fixtures_10_categories_5_each() -> None:
    cases = load_fixtures(FIXTURES)
    from collections import Counter
    cnt = Counter(c["category"] for c in cases)
    assert len(cnt) == 10
    assert all(n == 5 for n in cnt.values()), cnt


@pytest.mark.unit
def test_load_fixtures_each_case_has_required_keys() -> None:
    cases = load_fixtures(FIXTURES)
    required = {
        "case_id", "category", "sessions",
        "question_date", "question", "gold_answer", "stale_answer",
    }
    for c in cases:
        missing = required - set(c.keys())
        assert not missing, f"{c.get('case_id')} missing {missing}"


@pytest.mark.unit
def test_load_fixtures_sessions_have_user_turns() -> None:
    """Each fixture must have at least one user turn per session."""
    cases = load_fixtures(FIXTURES)
    for c in cases:
        for s in c["sessions"]:
            user_turns = [t for t in s["turns"] if t["role"] == "user"]
            assert user_turns, (
                f"{c['case_id']} session {s['session_id']} has no user turn"
            )


@pytest.mark.unit
def test_load_fixtures_aborts_on_empty_file(tmp_path) -> None:
    p = tmp_path / "empty.json"
    p.write_text(json.dumps({"version": 1, "cases": []}))
    with pytest.raises(SystemExit):
        load_fixtures(p)


# ── SuiteReport accumulation ──────────────────────────────────────────


@pytest.mark.unit
def test_suite_report_score_pct_zero_on_empty() -> None:
    r = SuiteReport()
    assert r.score_pct == 0.0


@pytest.mark.unit
def test_suite_report_aggregates_per_category() -> None:
    from evals.knowledge_updates.runner import CaseResult

    r = SuiteReport()
    r.add(CaseResult(
        case_id="ku_a_1", category="numeric",
        question="?", gold_answer="x", stale_answer="y",
        top1_content="x", top1_score=1.0, verdict="new_wins",
    ))
    r.add(CaseResult(
        case_id="ku_a_2", category="numeric",
        question="?", gold_answer="x", stale_answer="y",
        top1_content="y", top1_score=1.0, verdict="stale_wins",
    ))
    r.add(CaseResult(
        case_id="ku_b_1", category="categorical",
        question="?", gold_answer="x", stale_answer="y",
        top1_content="x", top1_score=1.0, verdict="new_wins",
    ))

    assert r.total == 3
    assert r.new_wins == 2
    assert r.stale_wins == 1
    assert r.score_pct == pytest.approx(2 / 3 * 100.0)
    assert r.by_category["numeric"]["new_wins"] == 1
    assert r.by_category["numeric"]["stale_wins"] == 1
    assert r.by_category["categorical"]["new_wins"] == 1
