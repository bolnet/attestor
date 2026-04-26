"""Phase 7.3 — REFLECTION_PROMPT + ReflectionEngine.reflect() tests."""

from __future__ import annotations

import json
from typing import Any, List

import pytest

from attestor.consolidation.reflection import (
    REFLECTION_PROMPT,
    ChangedBelief,
    Contradiction,
    ReflectionEngine,
    ReflectionResult,
    StablePattern,
    _parse_response,
)
from attestor.models import Memory


# ──────────────────────────────────────────────────────────────────────────
# Stub LLM
# ──────────────────────────────────────────────────────────────────────────


class _Choice:
    def __init__(self, c: str) -> None:
        self.message = type("M", (), {"content": c})()


class _Resp:
    def __init__(self, c: str) -> None:
        self.choices = [_Choice(c)]


class _Chat:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self._p = payload
        self._exc = exc
        self.calls: list = []

    def create(self, **kw: Any) -> _Resp:
        self.calls.append(kw)
        if self._exc is not None:
            raise self._exc
        return _Resp(self._p)


class StubClient:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self.chat = type("C", (), {"completions": _Chat(payload, exc=exc)})()

    @property
    def call_count(self) -> int:
        return len(self.chat.completions.calls)

    @property
    def last_prompt(self) -> str:
        return self.chat.completions.calls[-1]["messages"][0]["content"]


def _mem(mid: str, content: str = "fact") -> Memory:
    return Memory(id=mid, content=content, category="preference",
                  entity="user", confidence=0.9)


# ──────────────────────────────────────────────────────────────────────────
# Prompt content guards
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_prompt_has_four_categories() -> None:
    for kind in ("STABLE PREFERENCES", "STABLE CONSTRAINTS",
                 "CHANGED BELIEFS", "CONTRADICTIONS"):
        assert kind in REFLECTION_PROMPT


@pytest.mark.unit
def test_prompt_forbids_auto_resolve() -> None:
    """Contradictions must be flagged for human review — load-bearing."""
    assert "do NOT auto-resolve" in REFLECTION_PROMPT.lower() or \
        "Do NOT auto-resolve" in REFLECTION_PROMPT
    assert "HUMAN REVIEW" in REFLECTION_PROMPT


@pytest.mark.unit
def test_prompt_requires_3_evidence() -> None:
    assert "3+" in REFLECTION_PROMPT or "at least 3" in REFLECTION_PROMPT


@pytest.mark.unit
def test_prompt_format_placeholders() -> None:
    rendered = REFLECTION_PROMPT.format(user_id="u-1", facts_json="[]")
    assert "u-1" in rendered
    assert "{" not in rendered or "{" in REFLECTION_PROMPT  # only schema braces


# ──────────────────────────────────────────────────────────────────────────
# _parse_response — JSON shape → ReflectionResult
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_full_payload() -> None:
    raw = json.dumps({
        "stable_preferences": [
            {"text": "user prefers dark mode",
             "evidence": ["m1", "m2", "m3"], "confidence": 0.92},
        ],
        "stable_constraints": [
            {"text": "never schedule meetings before 10am",
             "evidence": ["m4", "m5", "m6"], "confidence": 0.85},
        ],
        "changed_beliefs": [
            {"old": "m7", "new": "m8", "reason": "moved to SF"},
        ],
        "contradictions_for_review": [
            {"facts": ["m9", "m10"], "rationale": "favorite color contradicts"},
        ],
    })
    r = _parse_response(raw)
    assert len(r.stable_preferences) == 1
    assert r.stable_preferences[0].evidence == ["m1", "m2", "m3"]
    assert r.stable_preferences[0].confidence == 0.92
    assert len(r.stable_constraints) == 1
    assert len(r.changed_beliefs) == 1
    assert r.changed_beliefs[0].old == "m7"
    assert len(r.contradictions_for_review) == 1
    assert r.contradictions_for_review[0].facts == ["m9", "m10"]
    assert r.error is None


@pytest.mark.unit
def test_parse_handles_markdown_fence() -> None:
    raw = "```json\n" + json.dumps({"stable_preferences": []}) + "\n```"
    r = _parse_response(raw)
    assert r.stable_preferences == []
    assert r.error is None


@pytest.mark.unit
def test_parse_bad_json_returns_error() -> None:
    r = _parse_response("not json")
    assert r.error == "bad json"
    assert r.stable_preferences == []


@pytest.mark.unit
def test_parse_drops_pattern_without_text() -> None:
    raw = json.dumps({
        "stable_preferences": [
            {"evidence": ["m1"]},   # no text — drop
            {"text": "good", "evidence": ["m1"]},
        ]
    })
    r = _parse_response(raw)
    assert len(r.stable_preferences) == 1


@pytest.mark.unit
def test_parse_drops_pattern_without_evidence_list() -> None:
    raw = json.dumps({
        "stable_preferences": [
            {"text": "x", "evidence": "not a list"},
            {"text": "y", "evidence": ["m1"]},
        ]
    })
    r = _parse_response(raw)
    assert len(r.stable_preferences) == 1
    assert r.stable_preferences[0].text == "y"


@pytest.mark.unit
def test_parse_clamps_confidence() -> None:
    raw = json.dumps({
        "stable_preferences": [
            {"text": "x", "evidence": ["m"], "confidence": 99.0},
            {"text": "y", "evidence": ["m"], "confidence": -1.0},
        ]
    })
    r = _parse_response(raw)
    assert r.stable_preferences[0].confidence == 1.0
    assert r.stable_preferences[1].confidence == 0.0


@pytest.mark.unit
def test_parse_drops_changed_belief_without_old_or_new() -> None:
    raw = json.dumps({
        "changed_beliefs": [
            {"old": "m1"},                # no new
            {"new": "m2"},                # no old
            {"old": "m3", "new": "m4", "reason": "ok"},
        ]
    })
    r = _parse_response(raw)
    assert len(r.changed_beliefs) == 1
    assert r.changed_beliefs[0].old == "m3"


@pytest.mark.unit
def test_parse_drops_contradiction_with_only_one_fact() -> None:
    """A 'contradiction' needs at least 2 facts."""
    raw = json.dumps({
        "contradictions_for_review": [
            {"facts": ["m1"], "rationale": "lonely"},
            {"facts": ["m2", "m3"], "rationale": "real"},
        ]
    })
    r = _parse_response(raw)
    assert len(r.contradictions_for_review) == 1


# ──────────────────────────────────────────────────────────────────────────
# Engine.reflect — end-to-end with stub
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_reflect_empty_facts_returns_empty_result_no_llm() -> None:
    stub = StubClient(payload='{"stable_preferences": []}')
    eng = ReflectionEngine(client=stub)
    r = eng.reflect([])
    assert r.total_patterns == 0
    assert stub.call_count == 0


@pytest.mark.unit
def test_reflect_no_client_returns_error() -> None:
    eng = ReflectionEngine(client=None)
    r = eng.reflect([_mem("m1")])
    assert r.error == "no llm client configured"


@pytest.mark.unit
def test_reflect_passes_user_id_and_facts_into_prompt() -> None:
    stub = StubClient(payload='{"stable_preferences": []}')
    eng = ReflectionEngine(client=stub)
    eng.reflect([_mem("mem-x", "user prefers dark mode")], user_id="u-42")
    assert "u-42" in stub.last_prompt
    assert "mem-x" in stub.last_prompt
    assert "user prefers dark mode" in stub.last_prompt


@pytest.mark.unit
def test_reflect_llm_error_returns_error_result() -> None:
    stub = StubClient(exc=RuntimeError("api down"))
    eng = ReflectionEngine(client=stub)
    r = eng.reflect([_mem("m1")])
    assert r.error is not None
    assert "api down" in r.error


@pytest.mark.unit
def test_reflect_full_round_trip() -> None:
    payload = json.dumps({
        "stable_preferences": [
            {"text": "user prefers dark mode",
             "evidence": ["m1", "m2", "m3"], "confidence": 0.9},
        ],
        "contradictions_for_review": [
            {"facts": ["m4", "m5"], "rationale": "favorite color disagrees"},
        ],
    })
    stub = StubClient(payload=payload)
    eng = ReflectionEngine(client=stub)
    r = eng.reflect([_mem(f"m{i}") for i in range(5)], user_id="u-1")
    assert r.total_patterns == 1
    assert len(r.contradictions_for_review) == 1
    assert r.error is None
