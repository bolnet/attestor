"""Phase 3.4 — conflict_resolver: ADD/UPDATE/INVALIDATE/NOOP decisions.

Stub-LLM driven. Verifies:
  - empty new_facts → []
  - empty existing → all ADD without LLM call
  - happy path: 4 decisions, one per operation, bound by index
  - text-match binding when LLM reorders
  - LLM failure → all-ADD fallback
  - bad JSON → all-ADD fallback
  - per-decision validation (UPDATE without existing_id → ADD)
  - Decision dataclass invariants
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from attestor.extraction.conflict_resolver import (
    VALID_OPERATIONS,
    Decision,
    _coerce,
    _parse_decisions_payload,
    resolve_conflicts,
)
from attestor.extraction.round_extractor import ExtractedFact
from attestor.models import Memory


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
    def __init__(self, content: str = "", *, raise_exc: Exception | None = None) -> None:
        self._content = content
        self._exc = raise_exc
        self.calls: list = []

    def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        if self._exc is not None:
            raise self._exc
        return _StubResponse(self._content)


class StubClient:
    def __init__(self, content: str = "", *, raise_exc: Exception | None = None) -> None:
        self.chat = type(
            "Chat", (),
            {"completions": _StubChat(content, raise_exc=raise_exc)},
        )()

    @property
    def call_count(self) -> int:
        return len(self.chat.completions.calls)


# ──────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────


def _fact(text: str, category: str = "preference") -> ExtractedFact:
    return ExtractedFact(
        text=text, category=category, entity=None,
        confidence=0.9, source_span=[0, len(text)], speaker="user",
    )


def _mem(memory_id: str, content: str) -> Memory:
    return Memory(id=memory_id, content=content, category="preference")


# ──────────────────────────────────────────────────────────────────────────
# Decision dataclass invariants
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_decision_rejects_invalid_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        Decision(
            operation="DELETE",  # not in VALID_OPERATIONS
            new_fact=_fact("x"), existing_id=None,
            rationale="r", evidence_episode_id="ep",
        )


@pytest.mark.unit
def test_decision_update_requires_existing_id() -> None:
    with pytest.raises(ValueError, match="existing_id"):
        Decision(
            operation="UPDATE",
            new_fact=_fact("x"), existing_id=None,
            rationale="r", evidence_episode_id="ep",
        )


@pytest.mark.unit
def test_decision_invalidate_requires_existing_id() -> None:
    with pytest.raises(ValueError, match="existing_id"):
        Decision(
            operation="INVALIDATE",
            new_fact=_fact("x"), existing_id=None,
            rationale="r", evidence_episode_id="ep",
        )


@pytest.mark.unit
def test_decision_add_allowed_without_existing_id() -> None:
    d = Decision(
        operation="ADD", new_fact=_fact("x"), existing_id=None,
        rationale="r", evidence_episode_id="ep",
    )
    assert d.operation == "ADD"


@pytest.mark.unit
def test_valid_operations_set_complete() -> None:
    assert VALID_OPERATIONS == {"ADD", "UPDATE", "INVALIDATE", "NOOP"}


# ──────────────────────────────────────────────────────────────────────────
# Short-circuit paths (no LLM)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_empty_new_facts_returns_empty() -> None:
    stub = StubClient()
    assert resolve_conflicts([], existing=[_mem("m1", "x")],
                             evidence_episode_id="ep", client=stub) == []
    assert stub.call_count == 0


@pytest.mark.unit
def test_resolve_empty_existing_returns_all_add_without_llm() -> None:
    """No existing memories → no contradictions possible → skip the LLM."""
    stub = StubClient()
    facts = [_fact("a"), _fact("b")]
    decisions = resolve_conflicts(facts, existing=[],
                                  evidence_episode_id="ep", client=stub)
    assert len(decisions) == 2
    assert all(d.operation == "ADD" for d in decisions)
    assert stub.call_count == 0


# ──────────────────────────────────────────────────────────────────────────
# Happy path — index-aligned binding
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_index_aligned_one_per_operation() -> None:
    facts = [_fact("a"), _fact("b"), _fact("c"), _fact("d")]
    existing = [_mem("m1", "old-a"), _mem("m2", "old-b")]
    payload = json.dumps({
        "decisions": [
            {"operation": "ADD",        "existing_id": None,  "rationale": "new"},
            {"operation": "UPDATE",     "existing_id": "m1",  "rationale": "refined"},
            {"operation": "INVALIDATE", "existing_id": "m2",  "rationale": "contradicts"},
            {"operation": "NOOP",       "existing_id": None,  "rationale": "dup"},
        ]
    })
    stub = StubClient(payload)
    decisions = resolve_conflicts(facts, existing=existing,
                                  evidence_episode_id="ep-42", client=stub)
    assert [d.operation for d in decisions] == ["ADD", "UPDATE", "INVALIDATE", "NOOP"]
    assert decisions[1].existing_id == "m1"
    assert decisions[2].existing_id == "m2"
    assert all(d.evidence_episode_id == "ep-42" for d in decisions)


@pytest.mark.unit
def test_resolve_index_aligned_pure_array_shape() -> None:
    facts = [_fact("a")]
    existing = [_mem("m1", "x")]
    payload = '[{"operation": "NOOP", "existing_id": null, "rationale": "dup"}]'
    stub = StubClient(payload)
    decisions = resolve_conflicts(facts, existing=existing,
                                  evidence_episode_id="ep", client=stub)
    assert decisions[0].operation == "NOOP"


# ──────────────────────────────────────────────────────────────────────────
# Text-match binding (LLM reordered)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_text_match_when_count_differs() -> None:
    facts = [_fact("alpha"), _fact("beta"), _fact("gamma")]
    existing = [_mem("m1", "old-alpha")]
    # LLM returned only 2 decisions, in reverse order, with new_fact text bound
    payload = json.dumps({
        "decisions": [
            {"operation": "NOOP", "existing_id": None,
             "rationale": "duplicate gamma",
             "new_fact": {"text": "gamma"}},
            {"operation": "ADD", "existing_id": None,
             "rationale": "fresh alpha",
             "new_fact": {"text": "alpha"}},
        ]
    })
    stub = StubClient(payload)
    decisions = resolve_conflicts(facts, existing=existing,
                                  evidence_episode_id="ep", client=stub)
    # Order must follow the input facts list
    assert [d.new_fact.text for d in decisions] == ["alpha", "beta", "gamma"]
    # 'beta' had no match → defaults to ADD
    assert decisions[1].operation == "ADD"
    assert "omitted" in decisions[1].rationale.lower() or "default" in decisions[1].rationale.lower()


# ──────────────────────────────────────────────────────────────────────────
# Coercion / validation of individual raw decisions
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_coerce_unknown_operation_falls_back_to_add() -> None:
    f = _fact("x")
    d = _coerce({"operation": "MERGE"}, f, "ep")
    assert d.operation == "ADD"
    assert "invalid operation" in d.rationale.lower()


@pytest.mark.unit
def test_coerce_update_missing_existing_id_falls_back_to_add() -> None:
    f = _fact("x")
    d = _coerce({"operation": "UPDATE", "existing_id": None}, f, "ep")
    assert d.operation == "ADD"
    assert "missing existing_id" in d.rationale.lower()


@pytest.mark.unit
def test_coerce_existing_id_string_or_null_handled() -> None:
    f = _fact("x")
    d_null = _coerce({"operation": "ADD", "existing_id": "null"}, f, "ep")
    d_empty = _coerce({"operation": "ADD", "existing_id": ""}, f, "ep")
    assert d_null.existing_id is None
    assert d_empty.existing_id is None


@pytest.mark.unit
def test_coerce_existing_id_non_string_coerced() -> None:
    """Some LLMs return numeric ids; coerce to str."""
    f = _fact("x")
    d = _coerce({"operation": "UPDATE", "existing_id": 42, "rationale": "r"},
                f, "ep")
    assert d.existing_id == "42"


@pytest.mark.unit
def test_coerce_evidence_episode_id_always_caller_supplied() -> None:
    """LLM may try to set evidence_episode_id; we override it."""
    f = _fact("x")
    d = _coerce(
        {"operation": "ADD", "evidence_episode_id": "lying-id"},
        f, evidence_episode_id="real-ep",
    )
    assert d.evidence_episode_id == "real-ep"


# ──────────────────────────────────────────────────────────────────────────
# Resilience — LLM failures
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_resolve_llm_exception_returns_all_add() -> None:
    facts = [_fact("a"), _fact("b")]
    existing = [_mem("m1", "x")]
    stub = StubClient(raise_exc=RuntimeError("network down"))
    decisions = resolve_conflicts(facts, existing=existing,
                                  evidence_episode_id="ep", client=stub)
    assert len(decisions) == 2
    assert all(d.operation == "ADD" for d in decisions)
    assert all("LLM resolver unavailable" in d.rationale for d in decisions)


@pytest.mark.unit
def test_resolve_bad_json_returns_all_add() -> None:
    stub = StubClient("Sorry, I cannot decide that.")
    facts = [_fact("a")]
    existing = [_mem("m1", "x")]
    decisions = resolve_conflicts(facts, existing=existing,
                                  evidence_episode_id="ep", client=stub)
    assert decisions[0].operation == "ADD"


# ──────────────────────────────────────────────────────────────────────────
# Parser
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_parse_decisions_envelope() -> None:
    raw = '{"decisions": [{"operation": "ADD"}]}'
    assert _parse_decisions_payload(raw) == [{"operation": "ADD"}]


@pytest.mark.unit
def test_parse_decisions_array() -> None:
    raw = '[{"operation": "ADD"}, {"operation": "NOOP"}]'
    assert len(_parse_decisions_payload(raw)) == 2


@pytest.mark.unit
def test_parse_decisions_handles_markdown() -> None:
    raw = "```json\n[{\"operation\": \"ADD\"}]\n```"
    assert _parse_decisions_payload(raw) == [{"operation": "ADD"}]


@pytest.mark.unit
def test_parse_decisions_bad_json_returns_empty() -> None:
    assert _parse_decisions_payload("nope") == []
    assert _parse_decisions_payload("") == []
