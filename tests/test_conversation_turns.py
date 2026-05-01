"""Phase 3.1 — ConversationTurn dataclass invariants."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from attestor.conversation.turns import ConversationTurn


@pytest.mark.unit
def test_turn_minimal_construction() -> None:
    t = ConversationTurn(
        thread_id="t-1", speaker="user", role="user", content="hi",
    )
    assert t.thread_id == "t-1"
    assert t.is_user
    assert not t.is_assistant
    assert t.ts.tzinfo is timezone.utc


@pytest.mark.unit
def test_turn_assistant_role() -> None:
    t = ConversationTurn(
        thread_id="t-1", speaker="planner-01", role="assistant", content="ok",
    )
    assert t.is_assistant
    assert not t.is_user


@pytest.mark.unit
def test_turn_rejects_empty_content() -> None:
    with pytest.raises(ValueError, match="content"):
        ConversationTurn(
            thread_id="t-1", speaker="user", role="user", content="",
        )


@pytest.mark.unit
def test_turn_rejects_empty_thread_id() -> None:
    with pytest.raises(ValueError, match="thread_id"):
        ConversationTurn(
            thread_id="", speaker="user", role="user", content="hi",
        )


@pytest.mark.unit
def test_turn_rejects_empty_speaker() -> None:
    with pytest.raises(ValueError, match="speaker"):
        ConversationTurn(
            thread_id="t-1", speaker="", role="user", content="hi",
        )


@pytest.mark.unit
def test_turn_rejects_invalid_role() -> None:
    with pytest.raises(ValueError, match="role"):
        ConversationTurn(
            thread_id="t-1", speaker="user", role="orchestrator", content="hi",
        )


@pytest.mark.unit
def test_turn_is_immutable() -> None:
    t = ConversationTurn(
        thread_id="t-1", speaker="user", role="user", content="hi",
    )
    with pytest.raises(Exception):  # FrozenInstanceError
        t.content = "modified"  # type: ignore[misc]


@pytest.mark.unit
def test_turn_explicit_timestamp() -> None:
    ts = datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc)
    t = ConversationTurn(
        thread_id="t-1", speaker="user", role="user", content="hi", ts=ts,
    )
    assert t.ts == ts


@pytest.mark.unit
def test_turn_metadata_default_is_isolated_dict() -> None:
    """Two turns must not share the same metadata dict instance."""
    t1 = ConversationTurn(
        thread_id="t-1", speaker="user", role="user", content="a",
    )
    t2 = ConversationTurn(
        thread_id="t-2", speaker="user", role="user", content="b",
    )
    assert t1.metadata is not t2.metadata
