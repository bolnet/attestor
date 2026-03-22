"""ChromaDB vector store — zero-config local embeddings, no API key required."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import chromadb

from agent_memory.store.base import VectorStore

logger = logging.getLogger("agent_memory")


class ChromaStore(VectorStore):
    """Persistent vector store using ChromaDB with local sentence-transformer embeddings.

    Drop-in replacement for the old pgvector VectorStore.
    ChromaDB handles embedding generation internally via sentence-transformers,
    so no API key or external service is needed.

    For benchmarking, pass a custom embedding_function to override the default.
    """

    ROLES = {"vector"}

    def __init__(self, store_path: Path, embedding_function=None) -> None:
        chroma_path = store_path / "chroma"
        self._client = chromadb.PersistentClient(path=str(chroma_path))

        if embedding_function is not None:
            self._embedding_fn = embedding_function
            self._provider = "custom"
        else:
            from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
            self._embedding_fn = SentenceTransformerEmbeddingFunction(
                model_name="all-MiniLM-L6-v2",
            )
            self._provider = "local"

        self._collection = self._client.get_or_create_collection(
            name="memories",
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

    @property
    def provider(self) -> str:
        """Return the current embedding provider name."""
        return self._provider

    def add(self, memory_id: str, content: str) -> None:
        """Store content with auto-generated embedding. Upserts if id exists."""
        self._collection.upsert(
            ids=[memory_id],
            documents=[content],
        )

    def search(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Search by text query. Returns list of {memory_id, content, distance}."""
        if self._collection.count() == 0:
            return []

        # Clamp limit to collection size to avoid ChromaDB error
        actual_limit = min(limit, self._collection.count())
        results = self._collection.query(
            query_texts=[query_text],
            n_results=actual_limit,
        )

        output: List[Dict[str, Any]] = []
        ids = results["ids"][0] if results["ids"] else []
        documents = results["documents"][0] if results["documents"] else []
        distances = results["distances"][0] if results["distances"] else []

        for i, memory_id in enumerate(ids):
            output.append({
                "memory_id": memory_id,
                "content": documents[i],
                "distance": distances[i],
            })
        return output

    def delete(self, memory_id: str) -> bool:
        """Remove embedding by memory id. Returns True if it existed."""
        existing = self._collection.get(ids=[memory_id])
        if not existing["ids"]:
            return False
        self._collection.delete(ids=[memory_id])
        return True

    def count(self) -> int:
        """Return number of stored vectors."""
        return self._collection.count()

    def close(self) -> None:
        """No-op — PersistentClient handles cleanup."""
        pass
