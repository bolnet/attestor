"""Abstention detector (Phase 9.4.1).

Decides whether a model's free-text response counts as abstention.

The Chain-of-Note ABSTAIN clause instructs the model to say
"I don't have that information" — that's the canonical phrase. Real
models drift in phrasing, so we ship a conservative phrase list that
catches the common variants without false-positiving on confident
"I don't know yet, but I think..." answers (which ARE answers).

The detector is configurable: operators with their own house rules
can swap in a different one via the Detector protocol.
"""

from __future__ import annotations

import re
from typing import Iterable, Protocol, Sequence


class Detector(Protocol):
    """Returns True iff the response is an abstention."""

    def __call__(self, response: str) -> bool: ...


# ── Default phrase list ──────────────────────────────────────────────────
#
# Patterns are matched against the lowercased response. Order doesn't
# matter — any hit returns True. Each pattern intentionally requires
# wording that signals "no available info", not weak hedges like
# "I'm not sure" (which often precede a real attempt).
DEFAULT_ABSTENTION_PATTERNS: tuple = (
    # Canonical CoN ABSTAIN phrasing
    r"\bi don'?t have (that|this|the|any|enough) (information|context|info|data|knowledge|details?)\b",
    # Common knowledge-gap phrasings
    r"\bno (relevant |available |such |sufficient )?(information|context|memory|data) (is |was )?(available|provided|found|present)\b",
    r"\b(i (cannot|can'?t|am unable to|don'?t (have the )?ability to)) (answer|determine|tell|provide|recall)\b",
    # "I don't know" — only when essentially the whole response, to avoid
    # catching hedged-but-substantive answers like "I don't know yet, but..."
    r"^\s*(sorry,?\s+)?i don'?t know[\.\!\?]?\s*$",
    # "Insufficient information"-style
    r"\b(insufficient|inadequate|not enough) (information|context|data|memory)\b",
    r"\bunable to (answer|determine|find|recall)\b",
    # "The memories don't contain..." — model citing the absence
    r"\b(memories?|context|information) (do(es)? not|don'?t) (contain|include|mention|reference)\b",
)


def _compile(patterns: Iterable[str]) -> tuple:
    return tuple(re.compile(p, re.IGNORECASE) for p in patterns)


_DEFAULT_COMPILED = _compile(DEFAULT_ABSTENTION_PATTERNS)


def is_abstention(
    response: str,
    *,
    patterns: Sequence = _DEFAULT_COMPILED,
) -> bool:
    """True iff ``response`` matches any abstention pattern.

    Empty / whitespace-only responses are treated as abstentions — a
    model that returns nothing has effectively declined to answer.
    """
    if not response or not response.strip():
        return True
    text = response.strip()
    return any(p.search(text) for p in patterns)


def make_detector(extra_patterns: Sequence[str] = ()) -> Detector:
    """Build a detector that adds operator-specific patterns.

    Useful when the deployed answerer model has its own characteristic
    abstention phrasing not covered by the defaults.
    """
    compiled = _DEFAULT_COMPILED + _compile(extra_patterns)

    def _detect(response: str) -> bool:
        return is_abstention(response, patterns=compiled)

    return _detect
