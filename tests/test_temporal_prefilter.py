"""Unit tests for ``attestor.retrieval.temporal_prefilter.detect_window``.

Pure regex behavior — no LLM, no I/O. Every test pins ``question_date``
explicitly so we don't depend on freezegun being installed and so the
test is reproducible across machines/timezones.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


# ──────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────

# A Wednesday — 2026-04-29 14:00 UTC. Convenient anchor for weekday
# math (chosen so "last Monday" = -2 days, "last Friday" = -5 days).
WEDNESDAY = datetime(2026, 4, 29, 14, 0, 0, tzinfo=timezone.utc)
# A Monday — 2026-04-27 09:00 UTC, used for the "last Monday on a
# Monday" edge case.
MONDAY = datetime(2026, 4, 27, 9, 0, 0, tzinfo=timezone.utc)


def _expect_target(
    *, question: str, question_date: datetime, target: datetime,
    tolerance_days: int = 3,
) -> None:
    """Assert detect_window returns the expected target + symmetric window."""
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        question, question_date=question_date, tolerance_days=tolerance_days,
    )
    assert detected is not None, f"no phrase matched in {question!r}"
    # Allow up to 1-second drift (datetime constructions in implementation
    # may use replace(microsecond=0) etc.).
    delta = abs((detected.target - target).total_seconds())
    assert delta < 1.0, (
        f"target mismatch for {question!r}: "
        f"expected {target.isoformat()}, got {detected.target.isoformat()}"
    )
    # Window symmetry: end - start == 2 * tolerance_days
    assert detected.window.start is not None
    assert detected.window.end is not None
    span = (detected.window.end - detected.window.start).total_seconds()
    expected_span = 2 * tolerance_days * 86400
    assert abs(span - expected_span) < 1.0, (
        f"window span mismatch: expected {expected_span}s, got {span}s"
    )
    # Target sits in the middle of the window (± 1s).
    mid = detected.window.start + (detected.window.end - detected.window.start) / 2
    assert abs((mid - detected.target).total_seconds()) < 1.0


# ──────────────────────────────────────────────────────────────────────
# Numeric N units ago
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_two_weeks_ago() -> None:
    _expect_target(
        question="What did I think two weeks ago?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(weeks=2),
    )


@pytest.mark.unit
def test_five_days_ago() -> None:
    _expect_target(
        question="Anything notable 5 days ago?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=5),
    )


@pytest.mark.unit
def test_ten_years_ago() -> None:
    _expect_target(
        question="What was I doing ten years ago?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=365 * 10),
    )


@pytest.mark.unit
def test_one_month_ago_uses_30_day_approximation() -> None:
    _expect_target(
        question="Status one month ago?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=30),
    )


# ──────────────────────────────────────────────────────────────────────
# Number-word forms (a/an/one/.../ten)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_a_week_ago_means_one_week() -> None:
    _expect_target(
        question="A week ago I changed jobs.",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(weeks=1),
    )


@pytest.mark.unit
def test_an_hour_does_not_match_units_only_dwmy() -> None:
    """The detector only handles day/week/month/year — 'an hour ago'
    must NOT match (no support for sub-day granularity)."""
    from attestor.retrieval.temporal_prefilter import detect_window

    result = detect_window("an hour ago", question_date=WEDNESDAY)
    assert result is None


@pytest.mark.unit
def test_three_days_ago_word_form() -> None:
    _expect_target(
        question="Three days ago we shipped",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=3),
    )


# ──────────────────────────────────────────────────────────────────────
# yesterday / day before yesterday
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_yesterday() -> None:
    _expect_target(
        question="What happened yesterday?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=1),
    )


@pytest.mark.unit
def test_day_before_yesterday() -> None:
    _expect_target(
        question="Did I work the day before yesterday?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=2),
    )


@pytest.mark.unit
def test_day_before_yesterday_picked_over_yesterday() -> None:
    """When 'the day before yesterday' appears, the longer phrase
    must win — never collapse to a 1-day match on the trailing
    'yesterday'."""
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "the day before yesterday I noted X",
        question_date=WEDNESDAY,
    )
    assert detected is not None
    delta_days = round((WEDNESDAY - detected.target).total_seconds() / 86400)
    assert delta_days == 2, f"expected 2 days back, got {delta_days}"


# ──────────────────────────────────────────────────────────────────────
# last <unit>
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_last_week() -> None:
    _expect_target(
        question="What did we ship last week?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(weeks=1),
    )


@pytest.mark.unit
def test_last_month_30_day_approximation() -> None:
    _expect_target(
        question="Last month's revenue?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=30),
    )


@pytest.mark.unit
def test_last_year() -> None:
    _expect_target(
        question="Was I sick last year?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=365),
    )


# ──────────────────────────────────────────────────────────────────────
# last <weekday>
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_last_monday_on_wednesday_points_two_days_back() -> None:
    """Wednesday → previous Monday is 2 days back."""
    _expect_target(
        question="Notes from last Monday?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=2),
    )


@pytest.mark.unit
def test_last_friday_on_wednesday_points_five_days_back() -> None:
    _expect_target(
        question="What did I commit last Friday?",
        question_date=WEDNESDAY,
        target=WEDNESDAY - timedelta(days=5),
    )


@pytest.mark.unit
def test_last_monday_on_a_monday_means_seven_days_back() -> None:
    """Edge case: when question_date IS a Monday, 'last Monday' must
    mean the PREVIOUS Monday (7 days back), not 0 days."""
    _expect_target(
        question="Recall last Monday's review.",
        question_date=MONDAY,
        target=MONDAY - timedelta(days=7),
    )


# ──────────────────────────────────────────────────────────────────────
# this morning|afternoon|evening
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_this_morning_targets_today() -> None:
    """'this morning' should target the same calendar day as
    question_date — exact target time is implementation-defined but
    should be within the same UTC day."""
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "What did I think this morning?", question_date=WEDNESDAY,
    )
    assert detected is not None
    assert detected.target.date() == WEDNESDAY.date()


@pytest.mark.unit
def test_this_evening_targets_today() -> None:
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "Plan for this evening?", question_date=WEDNESDAY,
    )
    assert detected is not None
    assert detected.target.date() == WEDNESDAY.date()


# ──────────────────────────────────────────────────────────────────────
# Forward-looking
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_two_weeks_later_is_forward() -> None:
    _expect_target(
        question="Two weeks later we shipped.",
        question_date=WEDNESDAY,
        target=WEDNESDAY + timedelta(weeks=2),
    )


@pytest.mark.unit
def test_three_days_after_is_forward() -> None:
    _expect_target(
        question="Three days after that meeting?",
        question_date=WEDNESDAY,
        target=WEDNESDAY + timedelta(days=3),
    )


@pytest.mark.unit
def test_one_year_hence_is_forward() -> None:
    _expect_target(
        question="One year hence I'll review.",
        question_date=WEDNESDAY,
        target=WEDNESDAY + timedelta(days=365),
    )


# ──────────────────────────────────────────────────────────────────────
# No match / negative cases
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_no_phrase_returns_none() -> None:
    from attestor.retrieval.temporal_prefilter import detect_window

    assert detect_window("What is my favorite color?", question_date=WEDNESDAY) is None
    assert detect_window("", question_date=WEDNESDAY) is None
    assert detect_window("    ", question_date=WEDNESDAY) is None


@pytest.mark.unit
def test_unrelated_word_last_does_not_match() -> None:
    """'last' alone (without week/month/year/<weekday>) should not
    match — e.g. 'the last time' is not a temporal anchor."""
    from attestor.retrieval.temporal_prefilter import detect_window

    assert detect_window("the last time we spoke", question_date=WEDNESDAY) is None


# ──────────────────────────────────────────────────────────────────────
# Multiple matches → pick the FIRST (left-to-right)
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_multiple_phrases_picks_first() -> None:
    """When 'yesterday' appears before 'last week', the detector picks
    yesterday (1 day back), not last week (7 days back)."""
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "Yesterday I thought about last week's review",
        question_date=WEDNESDAY,
    )
    assert detected is not None
    delta_days = round((WEDNESDAY - detected.target).total_seconds() / 86400)
    assert delta_days == 1, f"expected yesterday (1d back), got {delta_days}d"


@pytest.mark.unit
def test_multiple_phrases_first_two_weeks_then_yesterday() -> None:
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "Two weeks ago I started the project; yesterday I shipped it.",
        question_date=WEDNESDAY,
    )
    assert detected is not None
    delta_days = round((WEDNESDAY - detected.target).total_seconds() / 86400)
    assert delta_days == 14, f"expected two weeks (14d back), got {delta_days}d"


# ──────────────────────────────────────────────────────────────────────
# Tolerance window
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_custom_tolerance_widens_window() -> None:
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "yesterday", question_date=WEDNESDAY, tolerance_days=10,
    )
    assert detected is not None
    span = (detected.window.end - detected.window.start).total_seconds()
    assert abs(span - 20 * 86400) < 1.0, (
        f"expected 20-day span at tolerance=10, got {span/86400:.2f}d"
    )


@pytest.mark.unit
def test_zero_tolerance_collapses_window_to_target() -> None:
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "yesterday", question_date=WEDNESDAY, tolerance_days=0,
    )
    assert detected is not None
    assert detected.window.start == detected.target
    assert detected.window.end == detected.target


# ──────────────────────────────────────────────────────────────────────
# matched_text — the literal substring
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_matched_text_is_literal_substring() -> None:
    from attestor.retrieval.temporal_prefilter import detect_window

    q = "What happened YESTERDAY at the meeting?"
    detected = detect_window(q, question_date=WEDNESDAY)
    assert detected is not None
    assert detected.matched_text.lower() in q.lower()
    # Case-insensitive match preserves original casing in the question.
    assert "YESTERDAY" in q


@pytest.mark.unit
def test_matched_text_for_two_weeks_ago() -> None:
    from attestor.retrieval.temporal_prefilter import detect_window

    detected = detect_window(
        "Two weeks ago we shipped",
        question_date=WEDNESDAY,
    )
    assert detected is not None
    assert "two weeks ago" in detected.matched_text.lower()


# ──────────────────────────────────────────────────────────────────────
# question_date defaults to "now"
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_default_question_date_is_now() -> None:
    """When question_date is omitted, detect_window should use
    datetime.now(UTC). We verify the window is centered close to now."""
    from attestor.retrieval.temporal_prefilter import detect_window

    now = datetime.now(timezone.utc)
    detected = detect_window("yesterday")
    assert detected is not None
    # Target should be ~1 day before now (allow 5s slack for clock drift
    # between the call and the assertion).
    expected = now - timedelta(days=1)
    assert abs((detected.target - expected).total_seconds()) < 5.0
