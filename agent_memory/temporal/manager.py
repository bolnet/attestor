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
        rows = self.store.execute(
            """SELECT * FROM memories
               WHERE entity = ?
               ORDER BY COALESCE(event_date, created_at) ASC""",
            [entity],
        )
        return [Memory.from_row(r) for r in rows]

    def current_facts(
        self, category: Optional[str] = None, entity: Optional[str] = None
    ) -> List[Memory]:
        """Return only active, non-superseded memories."""
        conditions = ["status = 'active'", "valid_until IS NULL"]
        params: list = []

        if category:
            conditions.append("category = ?")
            params.append(category)
        if entity:
            conditions.append("entity = ?")
            params.append(entity)

        where = " AND ".join(conditions)
        rows = self.store.execute(
            f"SELECT * FROM memories WHERE {where} ORDER BY created_at DESC",
            params,
        )
        return [Memory.from_row(r) for r in rows]

    def check_contradictions(self, new_memory: Memory) -> List[Memory]:
        """Find active memories that potentially contradict the new one.

        Rule-based: same entity + same category + different content.
        """
        if not new_memory.entity:
            return []

        candidates = self.store.execute(
            """SELECT * FROM memories
               WHERE entity = ? AND category = ? AND status = 'active'
               AND valid_until IS NULL AND id != ?""",
            [new_memory.entity, new_memory.category, new_memory.id],
        )
        contradictions = []
        for row in candidates:
            existing = Memory.from_row(row)
            if existing.content.strip() != new_memory.content.strip():
                contradictions.append(existing)
        return contradictions

    def supersede(self, old_memory: Memory, new_memory_id: str) -> Memory:
        """Mark old memory as superseded by a new one."""
        old_memory.status = "superseded"
        old_memory.valid_until = datetime.now(timezone.utc).isoformat()
        old_memory.superseded_by = new_memory_id
        return self.store.update(old_memory)
