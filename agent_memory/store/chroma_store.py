"""ChromaDB vector store — OpenAI embeddings via OpenRouter when available, local fallback."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List

import chromadb

logger = logging.getLogger("agent_memory")


def _get_embedding_function():
    """Select best available embedding function.

    Priority:
    1. OpenAI text-embedding-3-small via OpenRouter (1536D, SOTA quality)
    2. OpenAI text-embedding-3-small via OpenAI direct (1536D)
    3. Local sentence-transformers all-MiniLM-L6-v2 (384D, zero-config fallback)
    """
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    openai_key = os.environ.get("OPENAI_API_KEY")

    if openrouter_key:
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("Using OpenAI embeddings via OpenRouter (text-embedding-3-small)")
            return OpenAIEmbeddingFunction(
                api_key=openrouter_key,
                api_base="https://openrouter.ai/api/v1",
                model_name="openai/text-embedding-3-small",
            ), "openai-openrouter"
        except Exception as e:
            logger.warning("OpenRouter embedding setup failed: %s, falling back to local", e)

    if openai_key:
        try:
            from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
            logger.info("Using OpenAI embeddings direct (text-embedding-3-small)")
            return OpenAIEmbeddingFunction(
                api_key=openai_key,
                model_name="text-embedding-3-small",
            ), "openai-direct"
        except Exception as e:
            logger.warning("OpenAI embedding setup failed: %s, falling back to local", e)

    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    logger.info("Using local sentence-transformers embeddings (all-MiniLM-L6-v2)")
    return SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
    ), "local"


class ChromaStore:
    """Persistent vector store using ChromaDB.

    Uses OpenAI text-embedding-3-small (1536D) via OpenRouter/OpenAI when
    an API key is available. Falls back to local sentence-transformers (384D)
    for zero-config operation.

    IMPORTANT: Switching embedding models invalidates existing vectors.
    Use batch_embed() to re-embed after changing models.
    """

    def __init__(self, store_path: Path) -> None:
        chroma_path = store_path / "chroma"
        self._client = chromadb.PersistentClient(path=str(chroma_path))
        self._embedding_fn, self._provider = _get_embedding_function()
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
