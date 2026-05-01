"""Cosmos DB DiskANN vector-role mixin for AzureBackend (split from azure_backend.py).

This module is private — consumers should import ``AzureBackend`` from
``attestor.store.azure_backend``. The mixin is stateless: it operates on
``self._memories_container`` and ``self._embed`` configured by ``AzureBackend``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("attestor")


class _AzureVectorMixin:
    """Cosmos DB DiskANN vector role for AzureBackend."""

    # ── VectorStore ──

    def add(self, memory_id: str, content: str) -> None:
        """Generate embedding and patch onto the memory document."""
        embedding = self._embed(content)
        # Read the existing document, add embedding, upsert
        items = list(self._memories_container.query_items(
            query="SELECT * FROM c WHERE c.id = @id",
            parameters=[{"name": "@id", "value": memory_id}],
            enable_cross_partition_query=True,
        ))
        if not items:
            return
        doc = items[0]
        doc["embedding"] = embedding
        self._memories_container.upsert_item(body=doc)

    def search(self, query_text: str, limit: int = 20) -> list[dict[str, Any]]:
        """Vector similarity search using Cosmos DB DiskANN VectorDistance."""
        query_vec = self._embed(query_text)
        query = (
            "SELECT TOP @limit c.id, c.content, "
            "VectorDistance(c.embedding, @queryVector, false, "
            "{'distanceFunction': 'cosine'}) AS distance "
            "FROM c WHERE IS_DEFINED(c.embedding) "
            "ORDER BY VectorDistance(c.embedding, @queryVector, false, "
            "{'distanceFunction': 'cosine'})"
        )
        items = list(self._memories_container.query_items(
            query=query,
            parameters=[
                {"name": "@limit", "value": limit},
                {"name": "@queryVector", "value": query_vec},
            ],
            enable_cross_partition_query=True,
        ))
        return [
            {
                "memory_id": item["id"],
                "content": item["content"],
                "distance": item.get("distance", 0.0),
            }
            for item in items
        ]

    def count(self) -> int:
        """Count documents that have vector embeddings."""
        items = list(self._memories_container.query_items(
            query="SELECT VALUE COUNT(1) FROM c WHERE IS_DEFINED(c.embedding)",
            enable_cross_partition_query=True,
        ))
        return items[0] if items else 0
