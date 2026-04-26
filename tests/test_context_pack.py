"""Phase 6.1 — ContextPack + ContextPackEntry dataclass tests."""

from __future__ import annotations

import json

import pytest

from attestor.models import ContextPack, ContextPackEntry
from attestor.prompts.chain_of_note import DEFAULT_CHAIN_OF_NOTE_PROMPT


def _entry(id_: str = "m-1", score: float = 0.9) -> ContextPackEntry:
    return ContextPackEntry(
        id=id_, content=f"fact for {id_}", category="preference",
        entity="user", valid_from="2026-04-01T00:00:00+00:00",
        valid_until=None, confidence=0.9,
        source_episode_id="ep-1", score=score,
    )


# ──────────────────────────────────────────────────────────────────────────
# Entry
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_entry_to_dict_round_trips() -> None:
    e = _entry()
    d = e.to_dict()
    assert d["id"] == "m-1"
    assert d["score"] == 0.9
    assert d["source_episode_id"] == "ep-1"


@pytest.mark.unit
def test_entry_is_frozen() -> None:
    e = _entry()
    with pytest.raises(Exception):  # FrozenInstanceError
        e.score = 0.1  # type: ignore[misc]


# ──────────────────────────────────────────────────────────────────────────
# Pack
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_pack_empty_is_legal() -> None:
    """Abstention case — no relevant memories — must produce a valid pack."""
    p = ContextPack(
        query="anything", memories=[], as_of=None, token_count=0,
        chain_of_note_prompt=DEFAULT_CHAIN_OF_NOTE_PROMPT,
    )
    assert p.memory_count == 0
    rendered = p.render_prompt()
    assert "[]" in rendered
    assert "ABSTAIN" in rendered  # the agent still gets the abstain instruction


@pytest.mark.unit
def test_pack_memories_json_is_valid_json() -> None:
    p = ContextPack(
        query="prefs", memories=[_entry("m-1"), _entry("m-2")],
        as_of=None, token_count=42,
        chain_of_note_prompt=DEFAULT_CHAIN_OF_NOTE_PROMPT,
    )
    parsed = json.loads(p.memories_json())
    assert isinstance(parsed, list)
    assert len(parsed) == 2
    assert {m["id"] for m in parsed} == {"m-1", "m-2"}


@pytest.mark.unit
def test_pack_render_prompt_substitutes_memories() -> None:
    p = ContextPack(
        query="prefs", memories=[_entry("m-special")],
        as_of=None, token_count=1,
        chain_of_note_prompt=DEFAULT_CHAIN_OF_NOTE_PROMPT,
    )
    out = p.render_prompt()
    assert '"id": "m-special"' in out
    # Placeholder should be gone
    assert "{memories_json}" not in out


@pytest.mark.unit
def test_pack_to_dict_includes_all_fields() -> None:
    p = ContextPack(
        query="q", memories=[_entry()], as_of="2026-04-26T00:00:00+00:00",
        token_count=10, chain_of_note_prompt=DEFAULT_CHAIN_OF_NOTE_PROMPT,
    )
    d = p.to_dict()
    for key in ("query", "memories", "as_of", "token_count", "chain_of_note_prompt"):
        assert key in d
    assert d["as_of"] == "2026-04-26T00:00:00+00:00"
    assert d["token_count"] == 10
    assert isinstance(d["memories"], list)


@pytest.mark.unit
def test_pack_memories_preserve_input_order() -> None:
    """The pack must NOT re-sort — score-descending ordering is the
    orchestrator's job; the pack is a transport, not a re-ranker."""
    high = _entry("m-high", score=0.9)
    low = _entry("m-low", score=0.1)
    # Caller hands us in low→high order; pack must keep that order.
    p = ContextPack(
        query="q", memories=[low, high], as_of=None, token_count=2,
        chain_of_note_prompt=DEFAULT_CHAIN_OF_NOTE_PROMPT,
    )
    parsed = json.loads(p.memories_json())
    assert [m["id"] for m in parsed] == ["m-low", "m-high"]


@pytest.mark.unit
def test_pack_is_frozen() -> None:
    p = ContextPack(
        query="q", memories=[], as_of=None, token_count=0,
        chain_of_note_prompt="x",
    )
    with pytest.raises(Exception):
        p.query = "modified"  # type: ignore[misc]
