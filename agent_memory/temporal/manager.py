"""Temporal logic: timeline queries, supersession, contradiction detection."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from agent_memory.models import Memory
from agent_memory.store.base import DocumentStore


class TemporalManager:
    """Handles temporal queries, contradiction detection, and supersession."""

    def __init__(self, store: DocumentStore):
        self.store = store

    def timeline(self, entity: str) -> List[Memory]:
        """Get all memories about an entity ordered by event_date/created_at."""
        memories = self.store.list_memories(entity=entity, limit=100_000)
        return sorted(
            memories,
            key=lambda m: m.event_date or m.created_at,
        )

    def current_facts(
        self, category: Optional[str] = None, entity: Optional[str] = None
    ) -> List[Memory]:
        """Return only active, non-superseded memories."""
        memories = self.store.list_memories(
            status="active", category=category, entity=entity, limit=100_000,
        )
        return [m for m in memories if m.valid_until is None]

    def check_contradictions(self, new_memory: Memory) -> List[Memory]:
        """Find active memories that potentially contradict the new one.

        Rule-based: same entity + same category + different content.
        """
        if not new_memory.entity:
            return []

        candidates = self.store.list_memories(
            status="active",
            category=new_memory.category,
            entity=new_memory.entity,
            limit=100_000,
        )
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
        old_memory.status = "superseded"
        old_memory.valid_until = datetime.now(timezone.utc).isoformat()
        old_memory.superseded_by = new_memory_id
        return self.store.update(old_memory)
