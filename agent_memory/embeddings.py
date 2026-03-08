"""
Embedding utility — text-embedding-3-small (1536 dims).

Provider priority:
  1. OPENROUTER_API_KEY  → OpenRouter (openai/text-embedding-3-small)
  2. OPENAI_API_KEY      → OpenAI direct (text-embedding-3-small)

Falls back gracefully when neither key is set.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict
from typing import List, Optional


_client = None
_model: str = "text-embedding-3-small"
_initialized: bool = False

# LRU cache for query embeddings — avoids repeated API calls for same text
_CACHE_MAX = 256
_embedding_cache: OrderedDict[str, List[float]] = OrderedDict()


def _get_client():
    """Lazy-init the embedding client (OpenRouter preferred, OpenAI fallback)."""
    global _client, _model, _initialized
    if _initialized:
        return _client

    _initialized = True

    try:
        from openai import OpenAI
    except ImportError:
        _client = None
        return None

    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")

    if openrouter_key:
        _client = OpenAI(
            api_key=openrouter_key,
            base_url="https://openrouter.ai/api/v1",
        )
        _model = "openai/text-embedding-3-small"
    elif openai_key:
        _client = OpenAI(api_key=openai_key)
        _model = "text-embedding-3-small"
    else:
        _client = None

    return _client


def reset():
    """Reset the cached client and embedding cache (useful for testing)."""
    global _client, _model, _initialized
    _client = None
    _model = "text-embedding-3-small"
    _initialized = False
    _embedding_cache.clear()


def available() -> bool:
    """Check if embedding computation is available."""
    return _get_client() is not None


def get_embedding(text: str) -> Optional[List[float]]:
    """Generate a single 1536-dim embedding.

    Uses an LRU cache to avoid redundant API calls for identical text.
    Returns None if no API key is set or the call fails.
    """
    # Check cache first
    cache_key = hashlib.md5(text.encode()).hexdigest()
    if cache_key in _embedding_cache:
        _embedding_cache.move_to_end(cache_key)
        return _embedding_cache[cache_key]

    client = _get_client()
    if not client:
        return None

    try:
        response = client.embeddings.create(
            model=_model,
            input=text,
        )
        embedding = response.data[0].embedding
        # Store in cache, evict oldest if full
        _embedding_cache[cache_key] = embedding
        if len(_embedding_cache) > _CACHE_MAX:
            _embedding_cache.popitem(last=False)
        return embedding
    except Exception:
        return None


def get_embeddings_batch(
    texts: List[str], batch_size: int = 50
) -> List[Optional[List[float]]]:
    """Batch-embed multiple texts.

    Returns a list the same length as texts. Entries are None on failure.
    Automatically chunks into sub-batches to avoid API limits.
    """
    client = _get_client()
    if not client:
        return [None] * len(texts)

    all_embeddings: List[Optional[List[float]]] = []
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        try:
            response = client.embeddings.create(
                model=_model,
                input=chunk,
            )
            all_embeddings.extend(d.embedding for d in response.data)
        except Exception:
            all_embeddings.extend([None] * len(chunk))

    return all_embeddings
