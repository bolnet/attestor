"""OpenSearch Serverless vector-role mixin for AWSBackend (split from aws_backend.py).

This module is private — consumers should import ``AWSBackend`` from
``attestor.store.aws_backend``. The mixin is stateless: it operates on
``self._opensearch`` / ``self._opensearch_index`` configured by
``AWSBackend.__init__`` and the embedding function provided by ``self._embed``.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("attestor")


class _AWSVectorMixin:
    """OpenSearch Serverless vector role for AWSBackend."""

    # ── VectorStore — OpenSearch Serverless ──

    def add(self, memory_id: str, content: str, namespace: str = "default") -> None:
        if self._opensearch is None:
            logger.debug("OpenSearch not configured — skipping vector add")
            return

        embedding = self._embed(content)
        doc = {
            "memory_id": memory_id,
            "content": content,
            "namespace": namespace,
            "embedding": embedding,
        }
        self._opensearch.index(
            index=self._opensearch_index,
            id=memory_id,
            body=doc,
        )

    def search(self, query_text: str, limit: int = 20, namespace: str | None = None) -> list[dict[str, Any]]:
        if self._opensearch is None:
            logger.debug("OpenSearch not configured — returning empty results")
            return []

        query_vec = self._embed(query_text)
        knn_clause: dict[str, Any] = {
            "embedding": {
                "vector": query_vec,
                "k": limit,
            }
        }
        if namespace is not None:
            knn_clause["embedding"]["filter"] = {"term": {"namespace": namespace}}
        body = {
            "size": limit,
            "query": {
                "knn": knn_clause,
            },
        }
        resp = self._opensearch.search(index=self._opensearch_index, body=body)
        results = []
        for hit in resp.get("hits", {}).get("hits", []):
            source = hit["_source"]
            # cosine distance = 1 - cosine similarity; OpenSearch returns score
            results.append({
                "memory_id": source.get("memory_id", hit["_id"]),
                "content": source.get("content", ""),
                "distance": 1.0 - hit.get("_score", 0.0),
            })
        return results

    def count(self) -> int:
        if self._opensearch is None:
            return 0

        try:
            resp = self._opensearch.count(index=self._opensearch_index)
            return resp.get("count", 0)
        except Exception:
            return 0
