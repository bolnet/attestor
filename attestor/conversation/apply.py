"""Apply ADD/UPDATE/INVALIDATE/NOOP decisions through the supersession path.

Phase 3.5. Each Decision becomes one or two writes against the document
store + (optionally) the vector store. INVALIDATE goes through the
existing ``TemporalManager.supersede`` helper so the timeline still
replays cleanly.

Returns ``AppliedDecision`` records so the caller can audit what
landed (and what was a no-op).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from attestor.models import Memory

if TYPE_CHECKING:
    # Decision and ExtractedFact are only used in type annotations
    # (PEP 563 lazy via `from __future__ import annotations`). Importing
    # them at runtime would close a cycle:
    #   extraction.conflict_resolver → extraction.round_extractor →
    #   conversation.turns → conversation.__init__ → conversation.apply
    # Type-checking-only keeps the wheel importable in a fresh install.
    from attestor.extraction.conflict_resolver import Decision
    from attestor.extraction.round_extractor import ExtractedFact

logger = logging.getLogger("attestor.conversation.apply")


@dataclass(frozen=True)
class AppliedDecision:
    """The outcome of applying one Decision."""

    operation: str       # mirrors Decision.operation
    memory_id: str | None   # the affected memory (new for ADD/UPDATE,
                               # the superseded one for INVALIDATE, None for NOOP)
    new_memory_id: str | None = None  # only set on INVALIDATE — the
                                          # replacement memory's id


def _fact_to_memory(
    fact: ExtractedFact,
    *,
    user_id: str,
    project_id: str | None,
    session_id: str | None,
    scope: str,
    source_episode_id: str,
    extraction_model: str,
    parent_agent_id: str | None = None,
) -> Memory:
    """Build a Memory dataclass from an ExtractedFact + tenancy context."""
    return Memory(
        content=fact.text,
        category=fact.category,
        entity=fact.entity,
        confidence=fact.confidence,
        user_id=user_id,
        project_id=project_id,
        session_id=session_id,
        scope=scope,
        source_episode_id=source_episode_id,
        source_span=list(fact.source_span),
        extraction_model=extraction_model,
        agent_id=fact.speaker,
        parent_agent_id=parent_agent_id,
    )


def apply_decisions(
    decisions: list[Decision],
    *,
    mem: Any,                      # AgentMemory; typed Any to avoid cycle
    user_id: str,
    project_id: str | None,
    session_id: str | None,
    scope: str,
    extraction_model: str,
    parent_agent_id: str | None = None,
) -> list[AppliedDecision]:
    """Apply each Decision through ``mem``'s document + vector stores.

    ADD         → store.insert(new_memory)
    UPDATE      → load existing, refresh content/category/confidence,
                  store.update; re-embed if vector store available
    INVALIDATE  → store.insert(new_memory) + temporal.supersede(old, new.id)
    NOOP        → record outcome, do nothing

    All errors are logged and surfaced as the operation failing for that
    decision. The pipeline does not raise — partial application is the
    contract (write what we can; report the rest).
    """
    out: list[AppliedDecision] = []

    for d in decisions:
        try:
            if d.operation == "ADD":
                out.append(_apply_add(
                    d, mem=mem, user_id=user_id, project_id=project_id,
                    session_id=session_id, scope=scope,
                    extraction_model=extraction_model,
                    parent_agent_id=parent_agent_id,
                ))
            elif d.operation == "UPDATE":
                out.append(_apply_update(d, mem=mem))
            elif d.operation == "INVALIDATE":
                out.append(_apply_invalidate(
                    d, mem=mem, user_id=user_id, project_id=project_id,
                    session_id=session_id, scope=scope,
                    extraction_model=extraction_model,
                    parent_agent_id=parent_agent_id,
                ))
            elif d.operation == "NOOP":
                out.append(AppliedDecision(operation="NOOP", memory_id=d.existing_id))
            else:
                # Decision.__post_init__ blocks this, but be defensive.
                logger.warning("unknown operation %r; skipping", d.operation)
                out.append(AppliedDecision(operation="NOOP", memory_id=None))
        except Exception as e:
            logger.warning(
                "failed to apply decision %s for fact %r: %s",
                d.operation, d.new_fact.text[:60], e,
            )
            out.append(AppliedDecision(operation="ERROR", memory_id=None))
    return out


def _apply_add(
    d: Decision,
    *,
    mem: Any,
    user_id: str,
    project_id: str | None,
    session_id: str | None,
    scope: str,
    extraction_model: str,
    parent_agent_id: str | None,
) -> AppliedDecision:
    new_mem = _fact_to_memory(
        d.new_fact, user_id=user_id, project_id=project_id,
        session_id=session_id, scope=scope,
        source_episode_id=d.evidence_episode_id,
        extraction_model=extraction_model,
        parent_agent_id=parent_agent_id,
    )
    inserted = mem._store.insert(new_mem)
    if mem._vector_store is not None:
        try:
            mem._vector_store.add(inserted.id, inserted.content)
        except Exception as e:
            logger.warning("vector add failed for %s: %s", inserted.id, e)
    return AppliedDecision(operation="ADD", memory_id=inserted.id)


def _apply_update(d: Decision, *, mem: Any) -> AppliedDecision:
    existing = mem._store.get(d.existing_id)
    if existing is None:
        logger.warning(
            "UPDATE target %s vanished; falling through as ADD-on-failed-update",
            d.existing_id,
        )
        return AppliedDecision(operation="ERROR", memory_id=d.existing_id)

    updates: dict = {"content": d.new_fact.text}
    if d.new_fact.category and d.new_fact.category != "general":
        updates["category"] = d.new_fact.category
    if d.new_fact.entity:
        updates["entity"] = d.new_fact.entity
    # Confidence: take the higher of old vs new (don't downgrade audit trail)
    updates["confidence"] = max(existing.confidence, d.new_fact.confidence)
    existing = replace(existing, **updates)
    mem._store.update(existing)

    if mem._vector_store is not None:
        try:
            mem._vector_store.add(existing.id, existing.content)
        except Exception as e:
            logger.warning("vector re-embed failed for %s: %s", existing.id, e)
    return AppliedDecision(operation="UPDATE", memory_id=existing.id)


def _apply_invalidate(
    d: Decision,
    *,
    mem: Any,
    user_id: str,
    project_id: str | None,
    session_id: str | None,
    scope: str,
    extraction_model: str,
    parent_agent_id: str | None,
) -> AppliedDecision:
    """Insert the new memory, then mark the old one superseded by it.

    The old row is NOT deleted — temporal replay must still find it.
    """
    new_mem = _fact_to_memory(
        d.new_fact, user_id=user_id, project_id=project_id,
        session_id=session_id, scope=scope,
        source_episode_id=d.evidence_episode_id,
        extraction_model=extraction_model,
        parent_agent_id=parent_agent_id,
    )
    inserted = mem._store.insert(new_mem)
    if mem._vector_store is not None:
        try:
            mem._vector_store.add(inserted.id, inserted.content)
        except Exception as e:
            logger.warning("vector add failed for %s: %s", inserted.id, e)

    old = mem._store.get(d.existing_id)
    if old is None:
        logger.warning(
            "INVALIDATE target %s vanished; new row %s stands alone",
            d.existing_id, inserted.id,
        )
        return AppliedDecision(
            operation="INVALIDATE", memory_id=None, new_memory_id=inserted.id,
        )
    mem._temporal.supersede(old, inserted.id)
    return AppliedDecision(
        operation="INVALIDATE", memory_id=old.id, new_memory_id=inserted.id,
    )
