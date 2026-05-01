"""Phase 3.3 — round-level extractor (extract_user_facts / extract_agent_facts).

Uses a stubbed LLM client so tests are deterministic and don't need
network / API keys. Each test crafts the LLM response shape and verifies
parsing, validation, clamping, and speaker-lock guards.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest

from attestor.conversation.turns import ConversationTurn
from attestor.extraction.round_extractor import (
    AGENT_FACT_CATEGORIES,
    USER_FACT_CATEGORIES,
    _parse_facts_payload,
    _strip_markdown_fences,
    _validate_fact,
    extract_agent_facts,
    extract_user_facts,
)


# ──────────────────────────────────────────────────────────────────────────
# Stub LLM
# ──────────────────────────────────────────────────────────────────────────


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]


class _StubChat:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_call: dict = {}

    def create(self, **kwargs: Any) -> _StubResponse:
        self.last_call = kwargs
        return _StubResponse(self._content)


class StubClient:
    def __init__(self, content: str) -> None:
        self.chat = type("Chat", (), {"completions": _StubChat(content)})()

    @property
    def last_prompt(self) -> str:
        msgs = self.chat.completions.last_call.get("messages", [])
        return msgs[0]["content"] if msgs else ""


def _user_turn(content: str = "I prefer dark mode") -> ConversationTurn:
    return ConversationTurn(
        thread_id="t-1", speaker="user", role="user", content=content,
        ts=datetime(2026, 4, 26, 10, 0, 0, tzinfo=timezone.utc),
    )


def _asst_turn(content: str = "Switching to dark mode now.") -> ConversationTurn:
    return ConversationTurn(
        thread_id="t-1", speaker="planner", role="assistant", content=content,
        ts=datetime(2026, 4, 26, 10, 0, 1, tzinfo=timezone.utc),
    )


# ──────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_strip_markdown_fences_handles_json_block() -> None:
    raw = "```json\n{\"facts\": []}\n```"
    assert _strip_markdown_fences(raw) == '{"facts": []}'


@pytest.mark.unit
def test_strip_markdown_fences_no_change_when_absent() -> None:
    raw = '{"facts": []}'
    assert _strip_markdown_fences(raw) == raw


@pytest.mark.unit
def test_parse_facts_payload_envelope_shape() -> None:
    raw = '{"facts": [{"text": "hi"}]}'
    facts = _parse_facts_payload(raw)
    assert facts == [{"text": "hi"}]


@pytest.mark.unit
def test_parse_facts_payload_array_shape() -> None:
    raw = '[{"text": "a"}, {"text": "b"}]'
    facts = _parse_facts_payload(raw)
    assert len(facts) == 2


@pytest.mark.unit
def test_parse_facts_payload_bad_json_returns_empty() -> None:
    """Extractor failures must never raise — return [] on parse error."""
    assert _parse_facts_payload("not json") == []
    assert _parse_facts_payload("") == []


@pytest.mark.unit
def test_parse_facts_payload_drops_non_dict_entries() -> None:
    raw = '[{"text": "ok"}, "garbage", 42, null]'
    facts = _parse_facts_payload(raw)
    assert facts == [{"text": "ok"}]


# ──────────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_validate_fact_minimal_user() -> None:
    f = _validate_fact(
        {"text": "user prefers Python", "category": "preference",
         "entity": "Python", "confidence": 0.9, "source_span": [0, 19]},
        speaker="user", content_len=100,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    assert f is not None
    assert f.text == "user prefers Python"
    assert f.category == "preference"
    assert f.confidence == 0.9
    assert f.source_span == [0, 19]


@pytest.mark.unit
def test_validate_fact_drops_empty_text() -> None:
    f = _validate_fact(
        {"text": "", "category": "preference"},
        speaker="user", content_len=100,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    assert f is None


@pytest.mark.unit
def test_validate_fact_unknown_category_falls_back_to_general() -> None:
    f = _validate_fact(
        {"text": "fact", "category": "made_up"},
        speaker="user", content_len=100,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    assert f.category == "general"


@pytest.mark.unit
def test_validate_fact_clamps_confidence() -> None:
    high = _validate_fact(
        {"text": "x", "confidence": 99.0},
        speaker="user", content_len=10,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    low = _validate_fact(
        {"text": "x", "confidence": -0.5},
        speaker="user", content_len=10,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    assert high.confidence == 1.0
    assert low.confidence == 0.0


@pytest.mark.unit
def test_validate_fact_clamps_source_span_to_content() -> None:
    """Spans outside the turn's content range get clamped."""
    f = _validate_fact(
        {"text": "x", "source_span": [-5, 9999]},
        speaker="user", content_len=20,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    assert f.source_span == [0, 20]


@pytest.mark.unit
def test_validate_fact_invalid_span_falls_back_to_full_range() -> None:
    f = _validate_fact(
        {"text": "x", "source_span": "not a list"},
        speaker="user", content_len=10,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    assert f.source_span == [0, 10]


@pytest.mark.unit
def test_validate_fact_swaps_invalid_start_end() -> None:
    """If start > end, end becomes start (avoids negative-length spans)."""
    f = _validate_fact(
        {"text": "x", "source_span": [50, 10]},
        speaker="user", content_len=100,
        allowed_categories=USER_FACT_CATEGORIES,
    )
    assert f.source_start <= f.source_end


# ──────────────────────────────────────────────────────────────────────────
# extract_user_facts
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_extract_user_facts_happy_path() -> None:
    stub = StubClient(
        '{"facts": [{"text": "user prefers dark mode", '
        '"category": "preference", "entity": "UI", '
        '"confidence": 0.95, "source_span": [0, 19]}]}'
    )
    facts = extract_user_facts(_user_turn(), client=stub)
    assert len(facts) == 1
    f = facts[0]
    assert f.text == "user prefers dark mode"
    assert f.category == "preference"
    assert f.speaker == "user"


@pytest.mark.unit
def test_extract_user_facts_empty_response() -> None:
    stub = StubClient('{"facts": []}')
    assert extract_user_facts(_user_turn(), client=stub) == []


@pytest.mark.unit
def test_extract_user_facts_speaker_lock_refuses_assistant_turn() -> None:
    """Calling extract_user_facts on an assistant turn returns []."""
    stub = StubClient('{"facts": [{"text": "x"}]}')
    facts = extract_user_facts(_asst_turn(), client=stub)
    assert facts == []
    # Stub should not have been called
    assert stub.chat.completions.last_call == {}


@pytest.mark.unit
def test_extract_user_facts_includes_speaker_lock_in_prompt() -> None:
    stub = StubClient('{"facts": []}')
    extract_user_facts(_user_turn(), client=stub)
    prompt = stub.last_prompt
    assert "USER'S MESSAGE BELOW" in prompt
    assert "I prefer dark mode" in prompt


@pytest.mark.unit
def test_extract_user_facts_drops_malformed_facts() -> None:
    """Mixed valid/invalid response — only valid entries returned."""
    stub = StubClient(
        '[{"text": "good fact"}, {"category": "preference"}, '  # 2nd has no text
        '{"text": "another good"}]'
    )
    facts = extract_user_facts(_user_turn(), client=stub)
    assert len(facts) == 2


@pytest.mark.unit
def test_extract_user_facts_handles_markdown_fences() -> None:
    stub = StubClient(
        '```json\n{"facts": [{"text": "fact in fence"}]}\n```'
    )
    facts = extract_user_facts(_user_turn(), client=stub)
    assert len(facts) == 1
    assert facts[0].text == "fact in fence"


# ──────────────────────────────────────────────────────────────────────────
# extract_agent_facts
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_extract_agent_facts_happy_path() -> None:
    stub = StubClient(
        '{"facts": [{"text": "agent recommended dark mode", '
        '"category": "recommendation", "entity": "UI", '
        '"confidence": 0.9, "source_span": [0, 25]}]}'
    )
    facts = extract_agent_facts(_asst_turn(), client=stub)
    assert len(facts) == 1
    assert facts[0].speaker == "planner"
    assert facts[0].category == "recommendation"


@pytest.mark.unit
def test_extract_agent_facts_speaker_lock_refuses_user_turn() -> None:
    stub = StubClient('{"facts": [{"text": "x"}]}')
    facts = extract_agent_facts(_user_turn(), client=stub)
    assert facts == []
    assert stub.chat.completions.last_call == {}


@pytest.mark.unit
def test_extract_agent_facts_uses_assistant_lock_line() -> None:
    stub = StubClient('{"facts": []}')
    extract_agent_facts(_asst_turn(), client=stub)
    assert "ASSISTANT'S MESSAGE BELOW" in stub.last_prompt


@pytest.mark.unit
def test_agent_categories_are_distinct_from_user_categories() -> None:
    """Sanity check — agent categories shouldn't overlap with user ones."""
    assert USER_FACT_CATEGORIES.isdisjoint(AGENT_FACT_CATEGORIES)


@pytest.mark.unit
def test_extract_agent_facts_unknown_category_falls_back() -> None:
    """An assistant-side 'preference' is invalid for agent facts."""
    stub = StubClient(
        '[{"text": "x", "category": "preference"}]'
    )
    facts = extract_agent_facts(_asst_turn(), client=stub)
    assert facts[0].category == "general"


# ──────────────────────────────────────────────────────────────────────────
# Resilience
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_extract_user_facts_empty_llm_response() -> None:
    stub = StubClient("")
    assert extract_user_facts(_user_turn(), client=stub) == []


@pytest.mark.unit
def test_extract_user_facts_garbage_response() -> None:
    """LLM hallucinates non-JSON prose → return [] rather than crash."""
    stub = StubClient("Sorry, I can't help with that.")
    assert extract_user_facts(_user_turn(), client=stub) == []
