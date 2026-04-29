"""Regex-only temporal pre-filter for the recall pipeline (Phase 3 RC4).

Detects relative time phrases in a user question ("two weeks ago",
"yesterday", "last Monday") and computes the implied event-date
window. The orchestrator passes that window through the existing
``time_window`` kwarg to the vector + BM25 lanes, so retrieval narrows
to memories whose ``event_date`` is plausibly contemporaneous with
what the question is asking about.

Why this exists
---------------
Per-RCA evidence on LongMemEval-S, the embedder collapses the
temporal context of "what did I think TWO WEEKS AGO" into noise — the
content keywords dominate the cosine. Hard-pre-filtering the candidate
set by event-time recovers the +1.5% the dense lane was leaving on
the floor.

Pure regex; no LLM. A separate LLM-based ``TemporalQueryExpander``
exists at ``attestor.retrieval.temporal_query`` for harder cases
("before I had kids", "back when I lived in Seattle"). The two are
orthogonal — this module returns ``None`` and the LLM lane can still
fire downstream.

Public API
----------

    detect_window(
        question: str,
        *,
        question_date: Optional[datetime] = None,
        tolerance_days: int = 3,
    ) -> Optional[DetectedPhrase]

Returns ``None`` when no phrase matches. When multiple phrases match,
the FIRST (left-to-right) wins — so "Yesterday I thought about last
week" anchors to yesterday, not last week.

Configuration lives in ``configs/attestor.yaml`` under
``retrieval.temporal_prefilter``:

    retrieval:
      temporal_prefilter:
        enabled: false        # default off — flip per bench run
        tolerance_days: 3     # half-width of the window in days

Trace event emitted (when ATTESTOR_TRACE=1):

  - ``recall.stage.temporal_prefilter`` — matched phrase + computed
    target + window bounds
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Tuple

from attestor.retrieval.temporal_query import TimeWindow

logger = logging.getLogger("attestor.retrieval.temporal_prefilter")


# ──────────────────────────────────────────────────────────────────────
# Public dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DetectedPhrase:
    """Diagnostic record of which phrase was matched + the computed window.

    ``matched_text`` is the literal substring from the question that
    triggered the detection — useful for trace logs and audit. ``target``
    is the implied event date (single point), and ``window`` is the
    ``[target - tolerance, target + tolerance]`` interval that's actually
    handed to the retrieval lanes.
    """

    matched_text: str
    target: datetime
    window: TimeWindow


# ──────────────────────────────────────────────────────────────────────
# Number-word + unit lookups
# ──────────────────────────────────────────────────────────────────────

_NUMBER_WORDS: dict[str, int] = {
    "a": 1, "an": 1, "one": 1,
    "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}

# 1 month ≈ 30 days, 1 year ≈ 365 days. Calendar-accurate math would
# need ``dateutil.relativedelta`` (an extra dep we don't pull in for
# a soft pre-filter — the tolerance window absorbs the drift).
_UNIT_DAYS: dict[str, int] = {
    "day": 1, "days": 1,
    "week": 7, "weeks": 7,
    "month": 30, "months": 30,
    "year": 365, "years": 365,
}

_WEEKDAYS: dict[str, int] = {
    # Python: Monday=0, Sunday=6
    "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
    "friday": 4, "saturday": 5, "sunday": 6,
}


# ──────────────────────────────────────────────────────────────────────
# Regex catalogue
# ──────────────────────────────────────────────────────────────────────
#
# Order matters. We compile each pattern, scan the question, and pick
# the match with the SMALLEST start offset — i.e. the leftmost phrase
# wins when two patterns both match. Inside a single pattern, longer
# phrases must come before shorter ones (e.g. "the day before
# yesterday" before "yesterday") so the regex engine doesn't collapse
# the longer phrase into the shorter one.

_NUMBER_RE = r"(?:\d+|a|an|one|two|three|four|five|six|seven|eight|nine|ten)"
_UNIT_RE = r"(?:day|week|month|year)s?"
_WEEKDAY_RE = r"(?:monday|tuesday|wednesday|thursday|friday|saturday|sunday)"

# 1. N units ago
_RE_N_UNITS_AGO = re.compile(
    rf"\b({_NUMBER_RE})\s+({_UNIT_RE})\s+ago\b",
    re.IGNORECASE,
)

# 2. the day before yesterday — must be tried BEFORE plain yesterday
_RE_DAY_BEFORE_YESTERDAY = re.compile(
    r"\bthe\s+day\s+before\s+yesterday\b",
    re.IGNORECASE,
)

# 3. yesterday
_RE_YESTERDAY = re.compile(r"\byesterday\b", re.IGNORECASE)

# 4. last week|month|year
_RE_LAST_UNIT = re.compile(
    r"\blast\s+(week|month|year)\b",
    re.IGNORECASE,
)

# 5. last <weekday>
_RE_LAST_WEEKDAY = re.compile(
    rf"\blast\s+({_WEEKDAY_RE})\b",
    re.IGNORECASE,
)

# 6. this morning|afternoon|evening
_RE_THIS_PART_OF_DAY = re.compile(
    r"\bthis\s+(morning|afternoon|evening)\b",
    re.IGNORECASE,
)

# 7. Forward-looking: N units later|after|hence
_RE_N_UNITS_FORWARD = re.compile(
    rf"\b({_NUMBER_RE})\s+({_UNIT_RE})\s+(later|after|hence)\b",
    re.IGNORECASE,
)


# ──────────────────────────────────────────────────────────────────────
# Public entry point
# ──────────────────────────────────────────────────────────────────────


def detect_window(
    question: str,
    *,
    question_date: Optional[datetime] = None,
    tolerance_days: int = 3,
) -> Optional[DetectedPhrase]:
    """Return a ``DetectedPhrase`` when ``question`` contains a
    relative time phrase, else ``None``.

    ``question_date`` defaults to ``datetime.now(timezone.utc)``.
    ``tolerance_days`` widens the window symmetrically — people say
    "last week" when they mean 8 days ago, and the tolerance absorbs
    that drift so the lane filter doesn't false-negative.

    Defensive: returns None on empty/whitespace input. Never raises.
    """
    if not question or not question.strip():
        return None
    if question_date is None:
        question_date = datetime.now(timezone.utc)

    try:
        match = _earliest_match(question, question_date)
    except Exception as e:  # noqa: BLE001
        # Belt-and-braces: regex matching is the only thing here that
        # could plausibly raise (catastrophic backtracking on
        # adversarial input). Degrade silently.
        logger.debug("temporal_prefilter.detect_window: matcher raised: %s", e)
        return None

    if match is None:
        return None

    matched_text, target = match
    window = _build_window(target, tolerance_days=tolerance_days)
    return DetectedPhrase(
        matched_text=matched_text,
        target=target,
        window=window,
    )


# ──────────────────────────────────────────────────────────────────────
# Internals
# ──────────────────────────────────────────────────────────────────────


def _earliest_match(
    question: str, question_date: datetime,
) -> Optional[Tuple[str, datetime]]:
    """Run every pattern, pick the leftmost hit, return (matched_text, target).

    Returns ``None`` when no pattern matches.
    """
    candidates: List[Tuple[int, str, datetime]] = []

    # Pattern 1: N units ago
    for m in _RE_N_UNITS_AGO.finditer(question):
        n = _word_to_int(m.group(1))
        days = _UNIT_DAYS[m.group(2).lower()]
        target = question_date - timedelta(days=n * days)
        candidates.append((m.start(), m.group(0), target))

    # Pattern 2: the day before yesterday
    for m in _RE_DAY_BEFORE_YESTERDAY.finditer(question):
        candidates.append(
            (m.start(), m.group(0), question_date - timedelta(days=2)),
        )

    # Pattern 3: yesterday
    for m in _RE_YESTERDAY.finditer(question):
        candidates.append(
            (m.start(), m.group(0), question_date - timedelta(days=1)),
        )

    # Pattern 4: last <unit>
    for m in _RE_LAST_UNIT.finditer(question):
        unit = m.group(1).lower()
        days = _UNIT_DAYS[unit]
        candidates.append(
            (m.start(), m.group(0), question_date - timedelta(days=days)),
        )

    # Pattern 5: last <weekday>
    for m in _RE_LAST_WEEKDAY.finditer(question):
        wd = _WEEKDAYS[m.group(1).lower()]
        delta = _days_back_to_previous_weekday(question_date.weekday(), wd)
        candidates.append(
            (m.start(), m.group(0), question_date - timedelta(days=delta)),
        )

    # Pattern 6: this morning|afternoon|evening — same calendar day
    for m in _RE_THIS_PART_OF_DAY.finditer(question):
        candidates.append((m.start(), m.group(0), question_date))

    # Pattern 7: forward-looking N units later|after|hence
    for m in _RE_N_UNITS_FORWARD.finditer(question):
        n = _word_to_int(m.group(1))
        days = _UNIT_DAYS[m.group(2).lower()]
        target = question_date + timedelta(days=n * days)
        candidates.append((m.start(), m.group(0), target))

    if not candidates:
        return None

    # Leftmost wins; on tie (same start, e.g. "the day before
    # yesterday" overlaps "yesterday" if both patterns matched), prefer
    # the LONGER substring so we honor the more specific phrase.
    candidates.sort(key=lambda c: (c[0], -len(c[1])))
    _, matched_text, target = candidates[0]
    return matched_text, target


def _word_to_int(token: str) -> int:
    """Convert a number token (digit string OR word) to int."""
    t = token.strip().lower()
    if t.isdigit():
        return int(t)
    return _NUMBER_WORDS[t]


def _days_back_to_previous_weekday(today_wd: int, target_wd: int) -> int:
    """Days from today back to the most recent occurrence of target_wd
    that is STRICTLY BEFORE today.

    When today_wd == target_wd, returns 7 (the previous occurrence,
    not today). Otherwise returns ``(today_wd - target_wd) mod 7``,
    falling back to 7 if that would be 0.
    """
    delta = (today_wd - target_wd) % 7
    return delta if delta != 0 else 7


def _build_window(target: datetime, *, tolerance_days: int) -> TimeWindow:
    """Symmetric window around ``target`` of half-width ``tolerance_days``.

    ``tolerance_days=0`` collapses to a point (start == end == target),
    which is valid per ``TimeWindow``'s invariants (start ≤ end).
    Negative tolerance is treated as 0 — we never invert the window.
    """
    half = max(0, int(tolerance_days))
    delta = timedelta(days=half)
    return TimeWindow(
        start=target - delta,
        end=target + delta,
        interpretation=f"temporal_prefilter ±{half}d",
    )
