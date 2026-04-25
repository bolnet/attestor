"""Prompts and parsers for LongMemEval question classification.

Pure functions — no LLM calls, no I/O. Designed to be unit-testable in
isolation. The runner script (``classify_lme_v2.py``) imports from here.

Three prompt builders:
  - ``build_binary_prompt(target, pos_shots, neg_shots)``
        binary YES/NO classifier for one category
  - ``build_multiclass_prompt(shots)``
        single-call multi-class classifier
  - ``build_conflict_resolution_prompt(question, candidates)``
        arbitrate when ≥2 binary classifiers fire YES

Two parsers:
  - ``parse_binary_response(raw)`` → ``"YES" | "NO" | "UNKNOWN"``
  - ``parse_category_response(raw, valid)`` → category name or ``"UNKNOWN"``
"""

from __future__ import annotations

from dataclasses import dataclass

CATEGORIES: list[str] = [
    "temporal-reasoning",
    "multi-session",
    "knowledge-update",
    "single-session-user",
    "single-session-assistant",
    "single-session-preference",
]
CATEGORY_SET: frozenset[str] = frozenset(CATEGORIES)


# Crisp definitions. Each one names the *invariant* that distinguishes the
# category from every other one.
CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "temporal-reasoning":
        "Answer is the RESULT of arithmetic across TWO OR MORE event dates: "
        "a duration, a gap, an age, a date computed from a relative phrase, "
        "or the chronological order of multiple events. The answer itself is "
        "a number-of-days/weeks/months/years, a date, or a sequence.",
    "multi-session":
        "Answer requires combining, counting, summing, or comparing facts "
        "that appear across MULTIPLE separate conversation sessions. The "
        "answer is typically a count, total, or comparison — not a date or "
        "duration.",
    "knowledge-update":
        "Answer is the CURRENT state of an attribute that the user updated "
        "over the course of the conversation history. The user previously "
        "stated one value (a possession, a status, a count, a habit) and "
        "later stated a new one; the question asks for the latest value.",
    "single-session-user":
        "Answer is a concrete fact the USER stated in ONE single session "
        "(what they did, where they went, who was there, how much something "
        "cost in that one event). All evidence is in one session.",
    "single-session-assistant":
        "Answer is something the ASSISTANT said in ONE prior session. The "
        "question explicitly references the prior assistant turn — phrasings "
        "like 'remind me what you said', 'in our previous conversation you "
        "mentioned', 'you recommended X — what was the detail'.",
    "single-session-preference":
        "Question asks for a RECOMMENDATION, suggestion, advice, or tip "
        "tailored to the user's preferences. Phrasings like 'Can you "
        "suggest', 'Any tips for', 'I'm planning X — any ideas', 'what "
        "should I do'. Answer is forward-looking advice, not factual recall.",
}


# Disambiguations — guidance for distinguishing this category from its
# closest neighbors. Reframed (was CATEGORY_ANTI_RULES) so they act as
# tie-breakers rather than hard rejection criteria. Earlier "NOT X if Y"
# wording made gpt-5.1 over-reject.
CATEGORY_ANTI_RULES: dict[str, list[str]] = {
    "temporal-reasoning": [
        "NOT temporal-reasoning when 'how long did I wait / how long did "
        "X take' has the duration directly stated in one session — that's "
        "single-session-user.",
        "NOT temporal-reasoning if a direct who/what/where recall is the "
        "answer and a temporal phrase is just a filter — that's "
        "single-session-user or multi-session.",
        "NOT temporal-reasoning if the answer is a count, total, or "
        "comparison of magnitudes — that's multi-session.",
    ],
    "multi-session": [
        "NOT multi-session if a single-session-user recall is the answer "
        "and the temporal phrase is just a filter.",
        "NOT multi-session if the answer is a computed duration, age, or "
        "chronological order — that's temporal-reasoning.",
        "NOT multi-session if the answer is the LATEST state of an "
        "attribute the user explicitly updated mid-history — that's "
        "knowledge-update.",
    ],
    "knowledge-update": [
        "NOT knowledge-update unless the user explicitly UPDATED an "
        "earlier value somewhere in the history (a real change, not just "
        "a stated current count).",
        "NOT knowledge-update if the answer is a count or total spanning "
        "sessions — that's multi-session.",
    ],
    "single-session-user": [
        "NOT single-session-user if the question explicitly cites a prior "
        "ASSISTANT turn ('remind me what you said', 'you mentioned X') "
        "— that's single-session-assistant.",
        "NOT single-session-user if the question is a request for "
        "advice / suggestions / tips — that's single-session-preference.",
    ],
    "single-session-assistant": [
        "NOT single-session-assistant if the question is about something "
        "the USER did, even if framed as a reminder.",
        "NOT single-session-assistant without an explicit prior-assistant "
        "reference ('you said', 'you recommended', 'in our previous "
        "conversation you mentioned').",
    ],
    "single-session-preference": [
        "NOT single-session-preference if it's a factual recall of a "
        "stated preference ('what was my favorite X' is recall) — that's "
        "single-session-user.",
        "NOT single-session-preference if a concrete answer is sought from "
        "stored facts rather than tailored forward-looking advice.",
    ],
}


@dataclass(frozen=True)
class Shot:
    """Minimal example record for few-shot prompts."""
    question: str
    category: str  # may be the target (positive) or another (negative)


def _bullet_lines(items: list[str]) -> str:
    return "\n".join(f"  - {item}" for item in items)


def build_binary_prompt(
    target: str,
    pos_shots: list[Shot],
    neg_shots: list[Shot],
) -> str:
    """Build a YES/NO binary classifier prompt for one target category.

    Includes the target's crisp definition, its anti-rules, and interleaved
    pos/neg few-shot examples. Negatives carry their gold category in the
    example so the model can see WHY the answer is NO."""
    if target not in CATEGORY_SET:
        raise ValueError(f"unknown category: {target}")

    lines: list[str] = [
        f'You decide whether a question belongs to the LongMemEval category '
        f'"{target}".',
        "",
        f"Definition of {target}:",
        f"  {CATEGORY_DESCRIPTIONS[target]}",
        "",
        f"Disambiguation guide (use when the question is borderline):",
        _bullet_lines(CATEGORY_ANTI_RULES[target]),
        "",
        f"Decision rule: reply YES if the question matches the definition "
        f"of {target}. Reply NO only when the question more clearly fits a "
        f"different category according to the guide above. When in doubt "
        f"and the definition plausibly fits, prefer YES.",
        "",
        "Output: a SINGLE word — YES or NO. No explanation, no quotes.",
        "",
        "Examples:",
        "",
    ]
    for pos, neg in zip(pos_shots, neg_shots):
        lines.append(f"Question: {pos.question}")
        lines.append("Answer: YES")
        lines.append("")
        lines.append(f"Question: {neg.question}")
        lines.append(f"Answer: NO  (this is {neg.category}, not {target})")
        lines.append("")
    lines.append("Now classify the next question. Reply with only YES or NO.")
    return "\n".join(lines)


def build_multiclass_prompt(shots: dict[str, list[Shot]]) -> str:
    """Build a single-call multi-class prompt with category guide + shots."""
    lines: list[str] = [
        "You classify questions into ONE of these LongMemEval categories. "
        "Pick the SINGLE best fit. When two categories could plausibly "
        "fit, prefer the one whose definition matches MORE EXACTLY (use "
        "the disambiguations below as tie-breakers).",
        "",
        "Reply with EXACTLY the category name (lowercase, with hyphens). "
        "No prose, no explanation, no quotes.",
        "",
        "Categories:",
    ]
    for c in CATEGORIES:
        lines.append(f"  {c}:")
        lines.append(f"    {CATEGORY_DESCRIPTIONS[c]}")
    lines.append("")
    lines.append("Disambiguations (only consult when two categories tie):")
    for c in CATEGORIES:
        for ar in CATEGORY_ANTI_RULES[c]:
            lines.append(f"  · {ar}")
    lines.append("")
    lines.append("Examples:")
    lines.append("")

    # Round-robin so all categories appear interleaved (positional balance).
    n = max(len(shots.get(c, [])) for c in CATEGORIES)
    for i in range(n):
        for c in CATEGORIES:
            row = shots.get(c, [])
            if i < len(row):
                lines.append(f"Question: {row[i].question}")
                lines.append(f"Category: {c}")
                lines.append("")
    lines.append("Now classify the next question. Reply with only the category name.")
    return "\n".join(lines)


def build_conflict_resolution_prompt(
    question: str, candidates: list[str]
) -> str:
    """Build a prompt that asks the model to pick the BEST category from a
    short candidate list (typically 2-3 categories that all said YES)."""
    if not candidates:
        raise ValueError("candidates must be non-empty")
    bad = [c for c in candidates if c not in CATEGORY_SET]
    if bad:
        raise ValueError(f"unknown candidates: {bad}")

    lines: list[str] = [
        "Multiple binary classifiers think this question belongs to their "
        "category. Pick the SINGLE best fit using the definitions and "
        "anti-rules below. Tie-break in favor of the category whose "
        "definition matches MOST exactly (not most loosely).",
        "",
        "Candidates:",
    ]
    for c in candidates:
        lines.append(f"  {c}:")
        lines.append(f"    Definition: {CATEGORY_DESCRIPTIONS[c]}")
        for ar in CATEGORY_ANTI_RULES[c]:
            lines.append(f"    · {ar}")
    lines.append("")
    lines.append(f"Question: {question}")
    lines.append("")
    lines.append(
        "Reply with EXACTLY one of the candidate names (lowercase, hyphens). "
        "No explanation."
    )
    return "\n".join(lines)


# ── Parsers ────────────────────────────────────────────────────────────────

def parse_binary_response(raw: str) -> str:
    """Extract YES/NO/UNKNOWN from a model's free-text reply.

    Tolerant: strips whitespace, quotes, markdown; checks first 'word'.
    Returns 'UNKNOWN' if neither YES nor NO is found at the start."""
    if raw is None:
        return "UNKNOWN"
    cleaned = raw.strip().strip("`*\"'").upper()
    if cleaned.startswith("YES"):
        return "YES"
    if cleaned.startswith("NO"):
        return "NO"
    return "UNKNOWN"


def parse_category_response(raw: str, valid: list[str] | None = None) -> str:
    """Extract a category name from a model's free-text reply.

    Tolerant: strips quotes/punct, lowercases, then finds the first valid
    category name as a substring. Returns 'UNKNOWN' if none match.
    ``valid`` defaults to the full ``CATEGORIES`` list."""
    valid_list = valid if valid is not None else CATEGORIES
    if raw is None:
        return "UNKNOWN"
    cleaned = (
        raw.strip()
        .strip("`*\"'")
        .replace("_", "-")  # tolerate underscore variants
        .lower()
    )
    if not cleaned:
        return "UNKNOWN"
    # Exact match first.
    for c in valid_list:
        if cleaned == c:
            return c
    # Then prefix match.
    for c in valid_list:
        if cleaned.startswith(c):
            return c
    # Then substring match.
    for c in valid_list:
        if c in cleaned:
            return c
    return "UNKNOWN"
