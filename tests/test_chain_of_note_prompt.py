"""Phase 6.1 — Chain-of-Note prompt content guards.

Anti-regression on the load-bearing pieces (esp. ABSTAIN clause).
"""

from __future__ import annotations

import pytest

from attestor.prompts.chain_of_note import DEFAULT_CHAIN_OF_NOTE_PROMPT


@pytest.mark.unit
def test_prompt_has_five_numbered_steps() -> None:
    for n in ("1.", "2.", "3.", "4.", "5."):
        assert n in DEFAULT_CHAIN_OF_NOTE_PROMPT, f"missing step {n}"


@pytest.mark.unit
def test_prompt_has_notes_step() -> None:
    assert "NOTES" in DEFAULT_CHAIN_OF_NOTE_PROMPT
    assert "relevant" in DEFAULT_CHAIN_OF_NOTE_PROMPT.lower()
    assert "irrelevant" in DEFAULT_CHAIN_OF_NOTE_PROMPT.lower()


@pytest.mark.unit
def test_prompt_has_synthesis_step() -> None:
    assert "SYNTHESIS" in DEFAULT_CHAIN_OF_NOTE_PROMPT


@pytest.mark.unit
def test_prompt_has_cite_step() -> None:
    """Cite by id in square brackets, e.g. [mem_42]."""
    assert "CITE" in DEFAULT_CHAIN_OF_NOTE_PROMPT
    assert "[mem_42]" in DEFAULT_CHAIN_OF_NOTE_PROMPT


@pytest.mark.unit
def test_prompt_has_abstain_clause() -> None:
    """The +abstention fix from AbstentionBench — DON'T REMOVE."""
    assert "ABSTAIN" in DEFAULT_CHAIN_OF_NOTE_PROMPT
    assert "I don't have that information" in DEFAULT_CHAIN_OF_NOTE_PROMPT
    assert "do not invent" in DEFAULT_CHAIN_OF_NOTE_PROMPT


@pytest.mark.unit
def test_prompt_has_conflict_resolution_step() -> None:
    """Prefer later valid_from / higher confidence."""
    assert "CONFLICT" in DEFAULT_CHAIN_OF_NOTE_PROMPT
    assert "valid_from" in DEFAULT_CHAIN_OF_NOTE_PROMPT
    assert "confidence" in DEFAULT_CHAIN_OF_NOTE_PROMPT


@pytest.mark.unit
def test_prompt_has_memories_placeholder() -> None:
    """ContextPack.render_prompt fills this — must exist exactly once."""
    assert DEFAULT_CHAIN_OF_NOTE_PROMPT.count("{memories_json}") == 1


@pytest.mark.unit
def test_prompt_renders_without_other_format_fields() -> None:
    """No stray {placeholders} that would break str.format."""
    rendered = DEFAULT_CHAIN_OF_NOTE_PROMPT.format(memories_json="[]")
    assert rendered.count("{") == 0
    assert "[]" in rendered


@pytest.mark.unit
def test_prompt_module_re_exports() -> None:
    """Top-level attestor.prompts package exposes the constant."""
    from attestor.prompts import DEFAULT_CHAIN_OF_NOTE_PROMPT as via_pkg
    assert via_pkg is DEFAULT_CHAIN_OF_NOTE_PROMPT
