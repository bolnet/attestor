"""Phase 3.2 — extraction prompt content guards.

These tests are anti-regression: they assert the load-bearing pieces of
the prompts (speaker-lock IMPORTANT lines, JSON schemas, source_span,
operation set) stay intact. Drift here costs LongMemEval points
silently — Mem0's published +53.6 single-session-assistant fix was
exactly this speaker-lock line.
"""

from __future__ import annotations

import string

import pytest

from attestor.extraction.prompts import (
    AGENT_FACT_EXTRACTION_PROMPT,
    MEMORY_UPDATE_PROMPT,
    PROMPT_VARS,
    USER_FACT_EXTRACTION_PROMPT,
    format_agent_fact_prompt,
    format_memory_update_prompt,
    format_user_fact_prompt,
)


def _placeholders(template: str) -> set[str]:
    """Extract format-string field names from a template."""
    formatter = string.Formatter()
    return {
        name for _, name, _, _ in formatter.parse(template)
        if name
    }


# ───────────────────────────────────────────────────────────────────────────
# Speaker-lock IMPORTANT lines (the +53.6 fix — DON'T REMOVE)
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_user_prompt_has_speaker_lock_line() -> None:
    """User extractor MUST refuse to extract from anything but the user msg."""
    assert "IMPORTANT" in USER_FACT_EXTRACTION_PROMPT
    assert "USER'S MESSAGE BELOW" in USER_FACT_EXTRACTION_PROMPT


@pytest.mark.unit
def test_agent_prompt_has_speaker_lock_line() -> None:
    """Agent extractor MUST refuse to extract from anything but the assistant msg."""
    assert "IMPORTANT" in AGENT_FACT_EXTRACTION_PROMPT
    assert "ASSISTANT'S MESSAGE BELOW" in AGENT_FACT_EXTRACTION_PROMPT


@pytest.mark.unit
def test_user_prompt_warns_off_recent_context() -> None:
    """The 'recent context' block must be flagged as disambiguation-only."""
    assert "DO NOT extract" in USER_FACT_EXTRACTION_PROMPT


@pytest.mark.unit
def test_agent_prompt_warns_off_recent_context() -> None:
    assert "disambiguation only" in AGENT_FACT_EXTRACTION_PROMPT.lower()


# ───────────────────────────────────────────────────────────────────────────
# JSON schemas — required keys
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_user_fact_schema_keys_present() -> None:
    for key in ("text", "category", "entity", "confidence", "source_span"):
        assert f'"{key}"' in USER_FACT_EXTRACTION_PROMPT, f"missing key: {key}"


@pytest.mark.unit
def test_agent_fact_schema_keys_present() -> None:
    for key in ("text", "category", "entity", "confidence", "source_span"):
        assert f'"{key}"' in AGENT_FACT_EXTRACTION_PROMPT, f"missing key: {key}"


@pytest.mark.unit
def test_user_fact_categories_are_complete() -> None:
    """Categories list is the contract: extractor classifier outputs must
    fall in this set or downstream supersession breaks."""
    expected = {
        "preference", "career", "project", "technical",
        "personal", "location", "relationship", "event", "financial",
    }
    for cat in expected:
        assert cat in USER_FACT_EXTRACTION_PROMPT, f"missing category: {cat}"


@pytest.mark.unit
def test_agent_fact_categories_are_complete() -> None:
    expected = {
        "recommendation", "decision", "commitment",
        "constraint", "calculation", "refusal",
    }
    for cat in expected:
        assert cat in AGENT_FACT_EXTRACTION_PROMPT, f"missing category: {cat}"


# ───────────────────────────────────────────────────────────────────────────
# Source-span citation requirement (audit trail)
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_user_prompt_requires_source_span() -> None:
    assert "source_span" in USER_FACT_EXTRACTION_PROMPT
    # Schema uses [<start_char>, <end_char>] tokens — this is the audit
    # contract, every fact must cite back to char offsets in the source.
    assert "start_char" in USER_FACT_EXTRACTION_PROMPT
    assert "end_char" in USER_FACT_EXTRACTION_PROMPT


@pytest.mark.unit
def test_agent_prompt_requires_source_span() -> None:
    assert "source_span" in AGENT_FACT_EXTRACTION_PROMPT


# ───────────────────────────────────────────────────────────────────────────
# MEMORY_UPDATE — operation set is the contract
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_memory_update_operations_complete() -> None:
    """All four operations + INVALIDATE (the v4 supersession primitive)."""
    for op in ("ADD", "UPDATE", "INVALIDATE", "NOOP"):
        assert op in MEMORY_UPDATE_PROMPT, f"missing operation: {op}"


@pytest.mark.unit
def test_memory_update_requires_evidence() -> None:
    """Every decision must cite the source episode."""
    assert "evidence_episode_id" in MEMORY_UPDATE_PROMPT


@pytest.mark.unit
def test_memory_update_invalidate_keeps_old_row() -> None:
    """INVALIDATE marks superseded; never deletes (timeline must replay)."""
    assert "DO NOT delete" in MEMORY_UPDATE_PROMPT
    assert "supersedes" in MEMORY_UPDATE_PROMPT


# ───────────────────────────────────────────────────────────────────────────
# Format-string contract
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_user_prompt_placeholders_match_declared() -> None:
    actual = _placeholders(USER_FACT_EXTRACTION_PROMPT)
    assert actual == PROMPT_VARS["USER_FACT"], (
        f"USER_FACT placeholders drifted: declared={PROMPT_VARS['USER_FACT']} "
        f"actual={actual}"
    )


@pytest.mark.unit
def test_agent_prompt_placeholders_match_declared() -> None:
    actual = _placeholders(AGENT_FACT_EXTRACTION_PROMPT)
    assert actual == PROMPT_VARS["AGENT_FACT"]


@pytest.mark.unit
def test_memory_update_placeholders_match_declared() -> None:
    actual = _placeholders(MEMORY_UPDATE_PROMPT)
    assert actual == PROMPT_VARS["MEMORY_UPDATE"]


# ───────────────────────────────────────────────────────────────────────────
# Format helpers
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_format_user_fact_prompt_substitutes_message() -> None:
    out = format_user_fact_prompt(
        ts="2026-04-26T10:00:00Z",
        user_message="I prefer dark mode",
        recent_context_summary="thread is about UI prefs",
    )
    assert "I prefer dark mode" in out
    assert "2026-04-26T10:00:00Z" in out
    assert "thread is about UI prefs" in out


@pytest.mark.unit
def test_format_agent_fact_prompt_substitutes_message() -> None:
    out = format_agent_fact_prompt(
        ts="2026-04-26T10:00:01Z",
        assistant_message="Switching the UI to dark mode now.",
    )
    assert "Switching the UI to dark mode now." in out
    assert "(none)" in out  # default for recent_context_summary


@pytest.mark.unit
def test_format_memory_update_prompt_substitutes_inputs() -> None:
    out = format_memory_update_prompt(
        existing_memories_json='[{"id": "m1", "content": "old"}]',
        new_facts_json='[{"text": "new"}]',
    )
    assert '"id": "m1"' in out
    assert '"text": "new"' in out


@pytest.mark.unit
def test_format_helpers_dont_break_on_braces_in_input() -> None:
    """User content may contain JSON / code blocks; format must not blow up."""
    out = format_user_fact_prompt(
        ts="t", user_message='use {x: 1} for the config',
    )
    assert "{x: 1}" in out
