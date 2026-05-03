"""Temporal logic: timeline queries, supersession, contradiction detection."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone

from attestor.models import Memory
from attestor.store.base import DocumentStore


# Strip values that carry a unit hint (currency symbol, percent, common SI /
# imperial units) so two memories that differ only in such a value collapse
# to the same skeleton. Bare integers ("fact 1" vs "fact 2") are NOT
# stripped — they're often distinct enumerations rather than updates to the
# same fact, and false-firing on them surfaced in test_core regressions.
_NUM_PATTERN = re.compile(
    r"(?:[$€£¥])\s*[\d,]+(?:\.\d+)?\s*(?:[kmbt])?\b"           # currency
    r"|"
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:%|percent|kg|lbs?|mi(?:les)?|km|m|cm|mm|"
    r"hours?|hrs?|days?|years?|yrs?|months?|weeks?|wks?|"
    r"times?|x/(?:day|week|month)|/(?:day|week|month))\b",     # units
    flags=re.IGNORECASE,
)
_DATE_TAG_PATTERN = re.compile(r"\[\d{4}-\d{2}-\d{2}[^\]]*\]")
_WS_PATTERN = re.compile(r"\s+")
# Min content length before the entity-None fallback is allowed to fire.
# Trivially short memories (e.g. "fact 1") are too noisy to safely group.
_MIN_FALLBACK_LEN = 12


def _content_skeleton(content: str) -> str:
    """Lower-case, strip unit-bearing numeric values + inline date tags,
    normalize whitespace.

    Two memories with the same skeleton are talking about the same fact at
    different points in time (or with different values). Used as the
    grouping key for entity-None contradiction detection.
    """
    s = _DATE_TAG_PATTERN.sub("", content.lower())
    s = _NUM_PATTERN.sub("NUM", s)
    s = _WS_PATTERN.sub(" ", s).strip()
    return s


class TemporalManager:
    """Handles temporal queries, contradiction detection, and supersession."""

    def __init__(self, store: DocumentStore):
        self.store = store

    def timeline(
        self, entity: str, namespace: str | None = None
    ) -> list[Memory]:
        """Get all memories about an entity ordered by event_date/created_at."""
        memories = self.store.list_memories(
            entity=entity, namespace=namespace, limit=100_000
        )
        return sorted(
            memories,
            key=lambda m: m.event_date or m.created_at,
        )

    def current_facts(
        self,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
    ) -> list[Memory]:
        """Return only active, non-superseded memories."""
        memories = self.store.list_memories(
            status="active", category=category, entity=entity,
            namespace=namespace, limit=100_000,
        )
        return [m for m in memories if m.valid_until is None]

    def check_contradictions(self, new_memory: Memory) -> list[Memory]:
        """Find active memories that potentially contradict the new one.

        Two strategies:

        1. **Entity-tagged path** (existing): same entity + same category +
           same namespace + different content. This is the original v3
           rule and stays intact for entity-rich callers.

        2. **Entity-None fallback** (added 2026-05-02 for KU sample
           852ce960): same category + same namespace + same content
           skeleton (numeric-stripped) + different value. Catches LME-S
           knowledge-update pairs like "pre-approved for $350,000" vs
           "pre-approved for $400,000" where the LME extractor never
           tagged an entity. Does NOT fire on "Likes Python" vs "Likes
           JavaScript" because their skeletons differ — so this is safe
           for plain preference memories.
        """
        if new_memory.entity:
            candidates = self.store.list_memories(
                status="active",
                category=new_memory.category,
                entity=new_memory.entity,
                namespace=new_memory.namespace,
                limit=100_000,
            )
        else:
            # Entity-None fallback: narrow by category+namespace, then
            # filter by skeleton match on the Python side. The skeleton
            # match is intentionally strict — same template + same
            # unit-bearing value pattern — so semantically-different
            # memories with the same category don't collide. Skip the
            # fallback entirely for very short content (no signal).
            if len(new_memory.content.strip()) < _MIN_FALLBACK_LEN:
                return []
            new_skel = _content_skeleton(new_memory.content)
            # If skeleton == content (no values stripped), this isn't an
            # update-to-same-fact case — it's two distinct facts. Skip.
            if new_skel == new_memory.content.strip().lower():
                return []
            candidates = self.store.list_memories(
                status="active",
                category=new_memory.category,
                namespace=new_memory.namespace,
                limit=100_000,
            )
            candidates = [
                c for c in candidates
                if not c.entity
                and _content_skeleton(c.content) == new_skel
            ]

        contradictions = []
        for existing in candidates:
            if existing.valid_until is not None:
                continue
            if existing.id == new_memory.id:
                continue
            if existing.content.strip() != new_memory.content.strip():
                contradictions.append(existing)
        return contradictions

    def supersede(self, old_memory: Memory, new_memory_id: str) -> Memory:
        """Mark old memory as superseded by a new one."""
        old_memory = replace(
            old_memory,
            status="superseded",
            valid_until=datetime.now(timezone.utc).isoformat(),
            superseded_by=new_memory_id,
        )
        return self.store.update(old_memory)
