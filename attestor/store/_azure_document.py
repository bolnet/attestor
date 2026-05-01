"""Cosmos DB document-role mixin for AzureBackend (split from azure_backend.py).

This module is private — consumers should import ``AzureBackend`` from
``attestor.store.azure_backend``. The mixin is stateless: it operates on
``self._memories_container`` (and friends) configured by ``AzureBackend.__init__``.
"""

from __future__ import annotations

import logging
from typing import Any

from attestor.models import Memory

logger = logging.getLogger("attestor")


class _AzureDocumentMixin:
    """Cosmos DB document role for AzureBackend."""

    # ── DocumentStore ──

    def _memory_to_doc(self, memory: Memory) -> dict[str, Any]:
        """Convert Memory to Cosmos DB document. id field is Cosmos PK."""
        return {
            "id": memory.id,
            "content": memory.content,
            "tags": memory.tags,
            "category": memory.category,
            "entity": memory.entity,
            "namespace": memory.namespace,
            "created_at": memory.created_at,
            "event_date": memory.event_date,
            "valid_from": memory.valid_from,
            "valid_until": memory.valid_until,
            "superseded_by": memory.superseded_by,
            "confidence": memory.confidence,
            "status": memory.status,
            "metadata": memory.metadata,
        }

    def _doc_to_memory(self, doc: dict[str, Any]) -> Memory:
        """Convert Cosmos DB document to Memory."""
        return Memory(
            id=doc["id"],
            content=doc["content"],
            tags=doc.get("tags", []),
            category=doc.get("category", "general"),
            entity=doc.get("entity"),
            namespace=doc.get("namespace", "default"),
            created_at=doc["created_at"],
            event_date=doc.get("event_date"),
            valid_from=doc["valid_from"],
            valid_until=doc.get("valid_until"),
            superseded_by=doc.get("superseded_by"),
            confidence=doc.get("confidence", 1.0),
            status=doc.get("status", "active"),
            metadata=doc.get("metadata", {}),
        )

    def insert(self, memory: Memory) -> Memory:
        doc = self._memory_to_doc(memory)
        self._memories_container.create_item(body=doc)
        return memory

    def get(self, memory_id: str) -> Memory | None:
        try:
            # Cross-partition point read when category unknown
            items = list(self._memories_container.query_items(
                query="SELECT * FROM c WHERE c.id = @id",
                parameters=[{"name": "@id", "value": memory_id}],
                enable_cross_partition_query=True,
            ))
            if not items:
                return None
            return self._doc_to_memory(items[0])
        except Exception:
            return None

    def update(self, memory: Memory) -> Memory:
        doc = self._memory_to_doc(memory)
        # Preserve existing embedding if present
        existing = self.get(memory.id)
        if existing:
            try:
                existing_items = list(self._memories_container.query_items(
                    query="SELECT * FROM c WHERE c.id = @id",
                    parameters=[{"name": "@id", "value": memory.id}],
                    enable_cross_partition_query=True,
                ))
                if existing_items and "embedding" in existing_items[0]:
                    doc["embedding"] = existing_items[0]["embedding"]
            except Exception:
                pass
        self._memories_container.upsert_item(body=doc)
        return memory

    def delete(self, memory_id: str) -> bool:
        try:
            items = list(self._memories_container.query_items(
                query="SELECT c.id, c.category FROM c WHERE c.id = @id",
                parameters=[{"name": "@id", "value": memory_id}],
                enable_cross_partition_query=True,
            ))
            if not items:
                return False
            category = items[0]["category"]
            self._memories_container.delete_item(item=memory_id, partition_key=category)
            return True
        except Exception:
            return False

    def list_memories(
        self,
        status: str | None = None,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
        after: str | None = None,
        before: str | None = None,
        limit: int = 100,
    ) -> list[Memory]:
        filters: list[str] = []
        params: list[dict[str, Any]] = []

        if status:
            filters.append("c.status = @status")
            params.append({"name": "@status", "value": status})
        if category:
            filters.append("c.category = @category")
            params.append({"name": "@category", "value": category})
        if entity:
            filters.append("c.entity = @entity")
            params.append({"name": "@entity", "value": entity})
        if namespace:
            filters.append("c.namespace = @namespace")
            params.append({"name": "@namespace", "value": namespace})
        if after:
            filters.append("c.created_at >= @after")
            params.append({"name": "@after", "value": after})
        if before:
            filters.append("c.created_at <= @before")
            params.append({"name": "@before", "value": before})

        where = " AND ".join(filters) if filters else "1=1"
        query = f"SELECT * FROM c WHERE {where} ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
        params.append({"name": "@limit", "value": limit})

        items = list(self._memories_container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,
        ))
        return [self._doc_to_memory(item) for item in items]

    def tag_search(
        self,
        tags: list[str],
        category: str | None = None,
        namespace: str | None = None,
        limit: int = 20,
    ) -> list[Memory]:
        # Build tag filter: check if any tag in the list is in c.tags
        tag_conditions = []
        params: list[dict[str, Any]] = []
        for i, tag in enumerate(tags):
            param_name = f"@tag{i}"
            tag_conditions.append(f"ARRAY_CONTAINS(c.tags, {param_name})")
            params.append({"name": param_name, "value": tag})

        tag_filter = f"({' OR '.join(tag_conditions)})"
        filters = [
            "c.status = 'active'",
            "NOT IS_DEFINED(c.valid_until) OR IS_NULL(c.valid_until)",
            tag_filter,
        ]

        if category:
            filters.append("c.category = @category")
            params.append({"name": "@category", "value": category})
        if namespace:
            filters.append("c.namespace = @namespace")
            params.append({"name": "@namespace", "value": namespace})

        where = " AND ".join(filters)
        query = f"SELECT * FROM c WHERE {where} ORDER BY c.created_at DESC OFFSET 0 LIMIT @limit"
        params.append({"name": "@limit", "value": limit})

        items = list(self._memories_container.query_items(
            query=query,
            parameters=params,
            enable_cross_partition_query=True,
        ))
        return [self._doc_to_memory(item) for item in items]

    def execute(
        self, query: str, params: Any | None = None
    ) -> list[dict[str, Any]]:
        """Execute raw Cosmos SQL query."""
        parameters = params if isinstance(params, list) else []
        items = list(self._memories_container.query_items(
            query=query,
            parameters=parameters,
            enable_cross_partition_query=True,
        ))
        return items

    def archive_before(self, date: str) -> int:
        """Archive memories created before the given date."""
        items = list(self._memories_container.query_items(
            query="SELECT * FROM c WHERE c.created_at < @date AND c.status = 'active'",
            parameters=[{"name": "@date", "value": date}],
            enable_cross_partition_query=True,
        ))
        count = 0
        for item in items:
            item["status"] = "archived"
            self._memories_container.upsert_item(body=item)
            count += 1
        return count

    def compact(self) -> int:
        """Delete all archived memories."""
        items = list(self._memories_container.query_items(
            query="SELECT c.id, c.category FROM c WHERE c.status = 'archived'",
            parameters=[],
            enable_cross_partition_query=True,
        ))
        count = 0
        for item in items:
            try:
                self._memories_container.delete_item(
                    item=item["id"], partition_key=item["category"]
                )
                count += 1
            except Exception as e:
                logger.warning("Failed to delete archived item %s: %s", item["id"], e)
        return count

    def stats(self) -> dict[str, Any]:
        """Return memory statistics."""
        total_items = list(self._memories_container.query_items(
            query="SELECT VALUE COUNT(1) FROM c",
            enable_cross_partition_query=True,
        ))
        total = total_items[0] if total_items else 0

        by_status: dict[str, int] = {}
        try:
            all_items = list(self._memories_container.query_items(
                query="SELECT c.status FROM c",
                enable_cross_partition_query=True,
            ))
            for item in all_items:
                s = item.get("status", "active")
                by_status[s] = by_status.get(s, 0) + 1
        except Exception:
            pass

        by_category: dict[str, int] = {}
        try:
            all_cats = list(self._memories_container.query_items(
                query="SELECT c.category FROM c",
                enable_cross_partition_query=True,
            ))
            for item in all_cats:
                c = item.get("category", "general")
                by_category[c] = by_category.get(c, 0) + 1
        except Exception:
            pass

        return {
            "total_memories": total,
            "by_status": by_status,
            "by_category": by_category,
        }
