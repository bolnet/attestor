"""Conflict resolver — turn extracted facts into Decisions (Phase 3.4).

Compares each newly-extracted ``ExtractedFact`` against existing
similar memories and decides one of:

  ADD         — new info, no existing match, write a fresh memory
  UPDATE      — same entity+predicate, refined value, keep existing id
  INVALIDATE  — old memory contradicted; mark superseded (timeline replays)
  NOOP        — already represented; skip

Backed by ``MEMORY_UPDATE_PROMPT``. Decisions carry the
``evidence_episode_id`` so every supersession is auditable.

Like the extractor, this module is resilient to LLM failures: a parse
failure for a single fact yields an ADD-by-default decision (safe path —
better to write a duplicate-ish row than to drop it silently).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, List, Optional

from attestor.extraction.prompts import format_memory_update_prompt
from attestor.extraction.round_extractor import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    ExtractedFact,
    _parse_facts_payload,
    _strip_markdown_fences,
    default_llm_client,
)
from attestor.models import Memory

logger = logging.getLogger("attestor.extraction.resolver")


VALID_OPERATIONS = {"ADD", "UPDATE", "INVALIDATE", "NOOP"}


@dataclass(frozen=True)
class Decision:
    """One ADD/UPDATE/INVALIDATE/NOOP decision for a single new fact."""

    operation: str  # one of VALID_OPERATIONS
    new_fact: ExtractedFact
    existing_id: Optional[str]  # None for ADD/NOOP; the existing memory id otherwise
    rationale: str
    evidence_episode_id: str

    def __post_init__(self) -> None:
        if self.operation not in VALID_OPERATIONS:
            raise ValueError(
                f"Decision.operation must be one of {VALID_OPERATIONS}; "
                f"got {self.operation!r}"
            )
        if self.operation in {"UPDATE", "INVALIDATE"} and not self.existing_id:
            raise ValueError(
                f"{self.operation} requires existing_id"
            )


# ──────────────────────────────────────────────────────────────────────────
# Serialization helpers — keep the prompt input deterministic
# ──────────────────────────────────────────────────────────────────────────


def _fact_to_dict(f: ExtractedFact) -> dict:
    return {
        "text": f.text,
        "category": f.category,
        "entity": f.entity,
        "confidence": f.confidence,
        "source_span": f.source_span,
        "speaker": f.speaker,
    }


def _memory_to_dict(m: Memory) -> dict:
    """Compact memory representation for the prompt context."""
    return {
        "id": m.id,
        "content": m.content,
        "category": m.category,
        "entity": m.entity,
        "valid_from": m.valid_from,
        "confidence": m.confidence,
    }


# ──────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────


def resolve_conflicts(
    new_facts: List[ExtractedFact],
    existing: List[Memory],
    evidence_episode_id: str,
    *,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    client: Optional[Any] = None,
) -> List[Decision]:
    """Decide ADD/UPDATE/INVALIDATE/NOOP for each new fact.

    Returns one Decision per new_fact (in order). On any LLM/parse
    failure the new fact gets an ADD-by-default decision so the caller
    still persists something.

    If ``new_facts`` is empty, returns []. If ``existing`` is empty,
    short-circuits with all-ADD without an LLM call (no contradictions
    are possible against an empty set).
    """
    if not new_facts:
        return []

    if not existing:
        return [
            Decision(
                operation="ADD",
                new_fact=f,
                existing_id=None,
                rationale="no existing memories to compare against",
                evidence_episode_id=evidence_episode_id,
            )
            for f in new_facts
        ]

    prompt = format_memory_update_prompt(
        existing_memories_json=json.dumps(
            [_memory_to_dict(m) for m in existing], ensure_ascii=False,
        ),
        new_facts_json=json.dumps(
            [_fact_to_dict(f) for f in new_facts], ensure_ascii=False,
        ),
    )
    cli = client or default_llm_client()
    try:
        from attestor.llm_trace import traced_create
        response = traced_create(
            cli,
            role="conflict_resolver",
            model=model,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw_text = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("conflict resolver LLM call failed: %s", e)
        return _fallback_all_add(new_facts, evidence_episode_id)

    raw_decisions = _parse_decisions_payload(raw_text)
    if not raw_decisions:
        return _fallback_all_add(new_facts, evidence_episode_id)

    return _bind_decisions(
        raw_decisions=raw_decisions,
        new_facts=new_facts,
        evidence_episode_id=evidence_episode_id,
    )


# ──────────────────────────────────────────────────────────────────────────
# Internal — parsing + binding
# ──────────────────────────────────────────────────────────────────────────


def _parse_decisions_payload(raw: str) -> List[dict]:
    """Parse the LLM response into a list of decision dicts."""
    text = _strip_markdown_fences(raw)
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("decisions JSON parse failed: %s; raw=%r", e, raw[:200])
        return []
    if isinstance(parsed, list):
        return [d for d in parsed if isinstance(d, dict)]
    if isinstance(parsed, dict):
        decs = parsed.get("decisions", [])
        if isinstance(decs, list):
            return [d for d in decs if isinstance(d, dict)]
    return []


def _bind_decisions(
    raw_decisions: List[dict],
    new_facts: List[ExtractedFact],
    evidence_episode_id: str,
) -> List[Decision]:
    """Match raw LLM decisions back to the input facts.

    Strategy: bind by index when lengths match (the LLM kept the order),
    otherwise by exact text match. Anything we can't bind falls back to
    ADD so no fact is silently dropped.
    """
    decisions: List[Decision] = []

    # Index-aligned fast path
    if len(raw_decisions) == len(new_facts):
        for fact, raw in zip(new_facts, raw_decisions):
            decisions.append(_coerce(raw, fact, evidence_episode_id))
        return decisions

    # Text-match path: build a lookup by raw fact text
    by_text: dict = {}
    for raw in raw_decisions:
        nf = raw.get("new_fact") or {}
        text = nf.get("text") if isinstance(nf, dict) else None
        if isinstance(text, str):
            by_text[text.strip()] = raw

    for fact in new_facts:
        raw = by_text.get(fact.text.strip())
        if raw is not None:
            decisions.append(_coerce(raw, fact, evidence_episode_id))
        else:
            decisions.append(_default_add(fact, evidence_episode_id,
                                          rationale="LLM omitted decision; default ADD"))
    return decisions


def _coerce(
    raw: dict,
    fact: ExtractedFact,
    evidence_episode_id: str,
) -> Decision:
    """Best-effort: build a valid Decision from a raw LLM dict."""
    op = raw.get("operation")
    if not isinstance(op, str) or op.upper() not in VALID_OPERATIONS:
        return _default_add(fact, evidence_episode_id,
                            rationale=f"invalid operation: {op!r}")
    op = op.upper()

    existing_id = raw.get("existing_id")
    if existing_id in (None, "", "null"):
        existing_id = None
    elif not isinstance(existing_id, str):
        existing_id = str(existing_id)

    # UPDATE / INVALIDATE without existing_id is broken — fall back to ADD
    if op in {"UPDATE", "INVALIDATE"} and not existing_id:
        return _default_add(
            fact, evidence_episode_id,
            rationale=f"{op} missing existing_id; defaulted to ADD",
        )

    rationale = raw.get("rationale")
    if not isinstance(rationale, str):
        rationale = "(no rationale provided)"

    # evidence_episode_id from the LLM is informational; we always trust
    # the caller's value, since the caller knows which episode actually
    # produced the fact.
    return Decision(
        operation=op,
        new_fact=fact,
        existing_id=existing_id,
        rationale=rationale,
        evidence_episode_id=evidence_episode_id,
    )


def _default_add(
    fact: ExtractedFact,
    evidence_episode_id: str,
    *,
    rationale: str,
) -> Decision:
    return Decision(
        operation="ADD",
        new_fact=fact,
        existing_id=None,
        rationale=rationale,
        evidence_episode_id=evidence_episode_id,
    )


def _fallback_all_add(
    new_facts: List[ExtractedFact], evidence_episode_id: str,
) -> List[Decision]:
    return [
        _default_add(
            f, evidence_episode_id,
            rationale="LLM resolver unavailable; defaulted to ADD",
        )
        for f in new_facts
    ]
