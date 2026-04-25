"""Unit tests for scripts/lme_prompts.py — prompt builders + parsers.

Pure deterministic tests; no LLM calls. Run via:
    poetry run python -m pytest tests/test_lme_prompts.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Add scripts/ to import path so this test can import the script-side module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lme_prompts import (  # noqa: E402
    CATEGORIES,
    CATEGORY_ANTI_RULES,
    CATEGORY_DESCRIPTIONS,
    CATEGORY_SET,
    Shot,
    build_binary_prompt,
    build_conflict_resolution_prompt,
    build_multiclass_prompt,
    parse_binary_response,
    parse_category_response,
)

# ───────────────────────────────────────────────────────────────────────────
# Constants & metadata coverage
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_six_categories_defined() -> None:
    assert len(CATEGORIES) == 6
    assert len(CATEGORY_SET) == 6
    expected = {
        "temporal-reasoning", "multi-session", "knowledge-update",
        "single-session-user", "single-session-assistant",
        "single-session-preference",
    }
    assert CATEGORY_SET == expected


@pytest.mark.unit
def test_every_category_has_description() -> None:
    for c in CATEGORIES:
        assert c in CATEGORY_DESCRIPTIONS
        assert len(CATEGORY_DESCRIPTIONS[c]) >= 60, (
            f"{c} description too short — needs to teach the invariant"
        )


@pytest.mark.unit
def test_every_category_has_at_least_two_anti_rules() -> None:
    for c in CATEGORIES:
        rules = CATEGORY_ANTI_RULES.get(c, [])
        assert len(rules) >= 2, (
            f"{c} needs ≥2 anti-rules to discriminate from neighbors"
        )
        for r in rules:
            assert r.lower().startswith("not "), (
                f"{c} anti-rule must start with 'NOT ' for clarity: {r!r}"
            )


# ───────────────────────────────────────────────────────────────────────────
# Anti-rule content — encodes the specific confusions we observed
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_temporal_reasoning_anti_rules_block_known_pitfalls() -> None:
    """The 'how long did I wait' / 'last week' filter pitfalls must be
    explicitly excluded — these were the FNs/FPs in the live cascade smoke."""
    body = " ".join(CATEGORY_ANTI_RULES["temporal-reasoning"]).lower()
    assert "how long did i wait" in body or "directly stated" in body, (
        "temporal-reasoning must exclude single-session 'how long did I wait'"
    )
    assert "filter" in body or "last week" in body, (
        "temporal-reasoning must exclude time-as-filter cases"
    )
    assert "count" in body, (
        "temporal-reasoning must redirect counts to multi-session"
    )


@pytest.mark.unit
def test_multi_session_anti_rules_redirect_to_neighbors() -> None:
    body = " ".join(CATEGORY_ANTI_RULES["multi-session"]).lower()
    assert "single-session" in body, (
        "multi-session must redirect single-session evidence cases"
    )
    assert "temporal-reasoning" in body, (
        "multi-session must redirect duration/age computations"
    )
    assert "knowledge-update" in body, (
        "multi-session must redirect updated-attribute cases"
    )


@pytest.mark.unit
def test_single_session_assistant_requires_explicit_assistant_reference() -> None:
    body = (
        CATEGORY_DESCRIPTIONS["single-session-assistant"]
        + " "
        + " ".join(CATEGORY_ANTI_RULES["single-session-assistant"])
    ).lower()
    assert (
        "previous conversation" in body
        or "you said" in body
        or "you mentioned" in body
        or "you recommended" in body
        or "remind me what you" in body
    ), (
        "single-session-assistant must require explicit prior-assistant-turn "
        "reference — distinct from single-session-user"
    )


@pytest.mark.unit
def test_single_session_preference_excludes_recall_questions() -> None:
    body = " ".join(CATEGORY_ANTI_RULES["single-session-preference"]).lower()
    assert "recall" in body or "factual" in body, (
        "single-session-preference must exclude factual-recall masquerading "
        "as preference questions ('what was my favorite X')"
    )


# ───────────────────────────────────────────────────────────────────────────
# build_binary_prompt
# ───────────────────────────────────────────────────────────────────────────


def _shots(target: str) -> tuple[list[Shot], list[Shot]]:
    pos = [
        Shot(question=f"pos-q-{i} for {target}", category=target)
        for i in range(1, 4)
    ]
    other = next(c for c in CATEGORIES if c != target)
    neg = [
        Shot(question=f"neg-q-{i}", category=other)
        for i in range(1, 4)
    ]
    return pos, neg


@pytest.mark.unit
def test_binary_prompt_rejects_unknown_target() -> None:
    pos, neg = _shots("temporal-reasoning")
    with pytest.raises(ValueError, match="unknown category"):
        build_binary_prompt("not-a-real-category", pos, neg)


@pytest.mark.unit
@pytest.mark.parametrize("target", CATEGORIES)
def test_binary_prompt_contains_definition_and_anti_rules(target: str) -> None:
    pos, neg = _shots(target)
    prompt = build_binary_prompt(target, pos, neg)
    # Definition is in the prompt verbatim.
    assert CATEGORY_DESCRIPTIONS[target] in prompt
    # Every anti-rule appears in the prompt.
    for rule in CATEGORY_ANTI_RULES[target]:
        assert rule in prompt, f"missing anti-rule for {target}: {rule!r}"
    # YES/NO output instruction is present.
    assert "YES or NO" in prompt
    # Decision rule must encourage YES when the definition plausibly fits
    # — anti-fix for the over-rejection regression that gpt-5.1 exhibited.
    assert "prefer YES" in prompt or "prefer-yes" in prompt.lower(), (
        f"binary prompt for {target} missing 'prefer YES' tie-breaker — "
        f"without it, models over-reject when an anti-rule could plausibly apply"
    )
    # Anti-rules must be framed as DISAMBIGUATION (tie-breaker), not a
    # hard rejection gate that ANDs with the definition.
    assert "only if ALL of the following hold" not in prompt, (
        "binary prompt has the over-restrictive AND-gate that caused the "
        "v2 baseline to collapse to 88% defaults — soften it"
    )


@pytest.mark.unit
def test_binary_prompt_interleaves_pos_neg_examples() -> None:
    pos, neg = _shots("temporal-reasoning")
    prompt = build_binary_prompt("temporal-reasoning", pos, neg)
    # Find positions of "Answer: YES" and "Answer: NO" — they should alternate.
    yes_idx = [i for i, ln in enumerate(prompt.splitlines()) if ln.strip() == "Answer: YES"]
    no_idx = [i for i, ln in enumerate(prompt.splitlines()) if ln.strip().startswith("Answer: NO")]
    # Same count — paired.
    assert len(yes_idx) == len(pos) == len(no_idx)
    # Each YES is followed by a NO before the next YES (interleaved).
    for y, n in zip(yes_idx, no_idx):
        assert n > y, "negative must come AFTER its paired positive"
    for i in range(1, len(yes_idx)):
        assert yes_idx[i] > no_idx[i - 1], "examples must alternate pos→neg→pos→neg"


@pytest.mark.unit
def test_binary_prompt_negatives_carry_their_gold_category() -> None:
    """The NO label includes a parenthetical with the *real* gold category.
    This teaches the model WHY the negative is a negative."""
    pos, neg = _shots("temporal-reasoning")
    prompt = build_binary_prompt("temporal-reasoning", pos, neg)
    for n in neg:
        assert f"this is {n.category}" in prompt, (
            "negative shots must reveal their actual gold category"
        )


# ───────────────────────────────────────────────────────────────────────────
# build_multiclass_prompt
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_multiclass_prompt_lists_all_categories_with_definitions() -> None:
    shots = {c: [Shot(question=f"q-{c}", category=c)] for c in CATEGORIES}
    prompt = build_multiclass_prompt(shots)
    for c in CATEGORIES:
        assert c in prompt
        assert CATEGORY_DESCRIPTIONS[c] in prompt
        for ar in CATEGORY_ANTI_RULES[c]:
            assert ar in prompt, f"multiclass missing {c} anti-rule: {ar!r}"
    assert "Reply with EXACTLY the category name" in prompt
    # Anti-rules must be framed as tie-breakers, not hard rejection gates.
    assert "tie-break" in prompt.lower() or "more exactly" in prompt.lower(), (
        "multi-class prompt missing tie-breaker framing — without it, "
        "anti-rules dominate and the model returns UNKNOWN too often"
    )


# ───────────────────────────────────────────────────────────────────────────
# build_conflict_resolution_prompt
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_conflict_resolution_includes_only_candidates() -> None:
    candidates = ["temporal-reasoning", "multi-session"]
    prompt = build_conflict_resolution_prompt(
        "How many days between X and Y?", candidates
    )
    # Both candidates fully described.
    for c in candidates:
        assert c in prompt
        assert CATEGORY_DESCRIPTIONS[c] in prompt
        for ar in CATEGORY_ANTI_RULES[c]:
            assert ar in prompt
    # Non-candidate categories should NOT appear (no point burning tokens).
    for c in CATEGORIES:
        if c not in candidates:
            assert CATEGORY_DESCRIPTIONS[c] not in prompt, (
                f"conflict prompt leaked non-candidate {c}"
            )
    # Question must be present verbatim.
    assert "How many days between X and Y?" in prompt


@pytest.mark.unit
def test_conflict_resolution_rejects_invalid_candidate() -> None:
    with pytest.raises(ValueError, match="unknown candidates"):
        build_conflict_resolution_prompt("q?", ["temporal-reasoning", "bogus"])


@pytest.mark.unit
def test_conflict_resolution_rejects_empty_candidates() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        build_conflict_resolution_prompt("q?", [])


@pytest.mark.unit
def test_conflict_resolution_includes_tie_break_rule() -> None:
    prompt = build_conflict_resolution_prompt(
        "q?", ["temporal-reasoning", "multi-session"]
    )
    assert "exact" in prompt.lower() or "tie-break" in prompt.lower()


# ───────────────────────────────────────────────────────────────────────────
# parse_binary_response
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("raw,expected", [
    ("YES", "YES"),
    ("yes", "YES"),
    ("Yes.", "YES"),
    ("  YES  ", "YES"),
    ("**YES**", "YES"),
    ("'YES'", "YES"),
    ('"yes"', "YES"),
    ("YES, this is temporal-reasoning", "YES"),
    ("NO", "NO"),
    ("no.", "NO"),
    ("  no", "NO"),
    ("NO — anti-rule applies", "NO"),
])
def test_parse_binary_response_accepts_common_forms(raw: str, expected: str) -> None:
    assert parse_binary_response(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", [
    "", "   ", "Maybe", "Unsure", "I think yes", "Probably no",
    None,
])
def test_parse_binary_response_returns_unknown_for_garbage(raw) -> None:
    assert parse_binary_response(raw) == "UNKNOWN"


# ───────────────────────────────────────────────────────────────────────────
# parse_category_response
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("raw,expected", [
    ("temporal-reasoning", "temporal-reasoning"),
    ("Temporal-Reasoning", "temporal-reasoning"),
    ("'temporal-reasoning'", "temporal-reasoning"),
    ("temporal-reasoning.", "temporal-reasoning"),
    ("multi-session", "multi-session"),
    ("Category: knowledge-update", "knowledge-update"),
    ("temporal_reasoning", "temporal-reasoning"),  # underscore variant
    ("single-session-preference", "single-session-preference"),
])
def test_parse_category_accepts_canonical_and_minor_variants(raw: str, expected: str) -> None:
    assert parse_category_response(raw) == expected


@pytest.mark.unit
@pytest.mark.parametrize("raw", ["", None, "I'm not sure", "category"])
def test_parse_category_returns_unknown_for_garbage(raw) -> None:
    assert parse_category_response(raw) == "UNKNOWN"


@pytest.mark.unit
def test_parse_category_respects_valid_subset() -> None:
    """When ``valid`` is a subset (e.g. conflict-resolution candidates),
    only those candidates can be returned."""
    raw = "single-session-user"
    valid = ["temporal-reasoning", "multi-session"]
    assert parse_category_response(raw, valid=valid) == "UNKNOWN"
    raw2 = "temporal-reasoning"
    assert parse_category_response(raw2, valid=valid) == "temporal-reasoning"


# ───────────────────────────────────────────────────────────────────────────
# Cross-cutting: every category's binary prompt must be self-consistent
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("target", CATEGORIES)
def test_binary_prompt_does_not_leak_other_category_definitions(target: str) -> None:
    """Each binary prompt should only contain its OWN definition — keep
    prompts focused (and cheap)."""
    pos, neg = _shots(target)
    prompt = build_binary_prompt(target, pos, neg)
    for other in CATEGORIES:
        if other == target:
            continue
        assert CATEGORY_DESCRIPTIONS[other] not in prompt, (
            f"binary prompt for {target} leaked {other}'s definition"
        )
