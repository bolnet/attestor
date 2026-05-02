"""Shared embedding providers — single source of truth for all backends.

YAML is authoritative. ``configs/attestor.yaml`` ``stack.embedder.provider``
selects exactly one provider. There is **no Python-level fallback chain**:
if the configured provider can't initialize, attestor raises loudly with a
config-pointing message rather than silently picking a different backend.
Silent fallthrough caused subtle index-dim drift in past runs and is
deliberately removed.

Supported providers (canonical PG+Pinecone+Neo4j stack):
    pinecone   — Pinecone Inference (default; ``llama-text-embed-v2`` 1024-D)
    voyage     — Voyage AI (``voyage-4`` 1024-D)
    openai     — OpenAI / OpenRouter (``text-embedding-3-large``, Matryoshka)
    ollama     — Local Ollama daemon

Usage:
    provider = get_embedding_provider()              # reads YAML's stack.embedder.provider
    provider = get_embedding_provider("voyage")      # explicit override (still strict)
    vec = provider.embed("hello world")              # -> List[float]
    vecs = provider.embed_batch(["a", "b"])          # -> List[List[float]]
"""

from __future__ import annotations

import logging
import os
from typing import Any, Protocol, runtime_checkable

logger = logging.getLogger("attestor")


# ═══════════════════════════════════════════════════════════════════════
# Protocol
# ═══════════════════════════════════════════════════════════════════════


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Minimal interface for embedding providers."""

    @property
    def dimension(self) -> int: ...

    @property
    def provider_name(self) -> str: ...

    def embed(self, text: str) -> list[float]: ...

    def embed_batch(self, texts: list[str]) -> list[list[float]]: ...


# ═══════════════════════════════════════════════════════════════════════
# Concrete Providers
# ═══════════════════════════════════════════════════════════════════════


class OpenAIEmbeddingProvider:
    """OpenAI or OpenRouter embeddings.

    Default model is ``text-embedding-3-large`` (3072D native). Passing
    ``dimensions`` requests a Matryoshka-reduced vector (e.g. 1536 for
    stock pgvector HNSW compatibility).
    """

    def __init__(
        self,
        api_key: str,
        base_url: str | None = None,
        model: str = "text-embedding-3-large",
        dimensions: int | None = None,
    ) -> None:
        from openai import OpenAI

        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._dimensions = dimensions
        # Send a single trivial token to probe the embedding dimension.
        # We can't use the embed() public API because the cache key
        # would pollute downstream stores; "x" is shorter than "dim"
        # and conveys nothing about the codebase.
        resp = self._client.embeddings.create(**self._create_kwargs(["x"]))
        self._dimension = len(resp.data[0].embedding)

    def _create_kwargs(self, texts: list[str]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"input": texts, "model": self._model}
        if self._dimensions is not None:
            kwargs["dimensions"] = self._dimensions
        return kwargs

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "openai"

    def embed(self, text: str) -> list[float]:
        resp = self._client.embeddings.create(**self._create_kwargs([text]))
        return resp.data[0].embedding

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._client.embeddings.create(**self._create_kwargs(texts))
        return [d.embedding for d in resp.data]


class OllamaEmbeddingProvider:
    """Local Ollama embeddings (default model: bge-m3, 1024-D, 8K context).

    Talks to ``http://localhost:11434/api/embeddings`` (override with
    ``OLLAMA_HOST``). Probe-detects dimension at construction so swapping
    the model just works — bge-m3 → 1024-D, nomic-embed-text → 768-D,
    mxbai-embed-large → 1024-D, etc.

    Local-first means: no network, no API credit cost, no rate limits.
    The price is RAM + an Ollama daemon; both are usually already there
    if you're running a local agent stack.
    """

    DEFAULT_MODEL = "bge-m3"
    DEFAULT_HOST = "http://localhost:11434"

    def __init__(
        self,
        model: str | None = None,
        host: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        import requests

        self._requests = requests
        self._host = (host or os.environ.get("OLLAMA_HOST") or self.DEFAULT_HOST).rstrip("/")
        self._model = model or os.environ.get("ATTESTOR_EMBEDDING_MODEL") or self.DEFAULT_MODEL
        self._timeout = timeout
        # Probe dimension once at boot. Raises if Ollama unreachable or
        # model not pulled — caller catches and falls through to next
        # provider in the chain.
        probe = self._raw_embed("dim")
        self._dimension = len(probe)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return f"ollama:{self._model}"

    def _raw_embed(self, text: str) -> list[float]:
        # Ollama's /api/embeddings returns {"embedding": [...]}; the newer
        # /api/embed returns {"embeddings": [[...]]} for batch. Use the
        # singular endpoint for consistency across versions.
        r = self._requests.post(
            f"{self._host}/api/embeddings",
            json={"model": self._model, "prompt": text},
            timeout=self._timeout,
        )
        r.raise_for_status()
        data = r.json()
        emb = data.get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(
                f"Ollama embedding response missing 'embedding': {data}"
            )
        return emb

    def embed(self, text: str) -> list[float]:
        return self._raw_embed(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama doesn't batch on the /api/embeddings endpoint. Use the
        # newer /api/embed if available; fall back to per-text loop.
        try:
            r = self._requests.post(
                f"{self._host}/api/embed",
                json={"model": self._model, "input": texts},
                timeout=self._timeout,
            )
            r.raise_for_status()
            data = r.json()
            embs = data.get("embeddings")
            if isinstance(embs, list) and len(embs) == len(texts):
                return embs
        except Exception as e:
            logger.debug("ollama batch endpoint failed (%s); per-text fallback", e)
        return [self._raw_embed(t) for t in texts]


class VoyageEmbeddingProvider:
    """Voyage AI embeddings (Anthropic's recommended embedder partner).

    Default model is ``voyage-4`` (1024D, Matryoshka — supports 256/512/1024/2048).
    The 1024D default fits stock pgvector HNSW and matches Attestor's
    canonical schema after the v4.0.0a5 dim-mismatch fix.

    Honours ``input_type`` semantics: callers using ``embed()`` here pass
    documents (write path); query-side embedding goes through the same
    method, but the orchestrator's vector_store.search() embeds queries
    via the same provider so per-call differentiation isn't currently
    plumbed. Voyage docs recommend using ``input_type="query"`` /
    ``"document"`` for retrieval — wired here as a constructor flag.
    """

    DEFAULT_MODEL = "voyage-4"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        input_type: str = "document",
    ) -> None:
        try:
            import voyageai
        except ImportError:
            raise RuntimeError(
                "voyageai not installed. Run `pip install voyageai` "
                "or `pip install attestor[voyage]`."
            )

        api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY not set")

        # Default timeout=None in voyageai SDK = unbounded — observed
        # 17-min hang during a 133q LME-S smoke 2026-04-30 when one
        # embed call stalled and never woke up. 30s is well above any
        # healthy embed (median ~170ms in practice).
        self._client = voyageai.Client(api_key=api_key, timeout=30.0)
        self._model = model or os.environ.get("VOYAGE_EMBEDDING_MODEL") or self.DEFAULT_MODEL
        self._input_type = input_type

        dim_env = os.environ.get("VOYAGE_EMBEDDING_DIMENSIONS")
        if dimensions is not None:
            self._dimensions = dimensions
        elif dim_env:
            self._dimensions = int(dim_env)
        else:
            self._dimensions = None  # let Voyage default (1024 for voyage-4 family)

        # Probe dimension
        probe_kwargs: dict[str, Any] = {"model": self._model, "input_type": self._input_type}
        if self._dimensions is not None:
            probe_kwargs["output_dimension"] = self._dimensions
        result = self._client.embed(["dim"], **probe_kwargs)
        self._dimension = len(result.embeddings[0])

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return f"voyage:{self._model}"

    def _embed_kwargs(self) -> dict[str, Any]:
        kw: dict[str, Any] = {"model": self._model, "input_type": self._input_type}
        if self._dimensions is not None:
            kw["output_dimension"] = self._dimensions
        return kw

    def embed(self, text: str) -> list[float]:
        result = self._client.embed([text], **self._embed_kwargs())
        return result.embeddings[0]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.embed(texts, **self._embed_kwargs())
        return result.embeddings


class PineconeEmbeddingProvider:
    """Pinecone Inference embeddings (cloud-only, NOT supported on Pinecone Local).

    Default model is ``llama-text-embed-v2`` — Meta × Pinecone partnership,
    English-strong, dim is configurable from {384, 512, 768, 1024, 2048}.
    1024-D matches Attestor's canonical pgvector schema and the existing
    Voyage configuration so the embedder swap is dim-compatible.

    Other models accessible via the same provider (just change ``model``):

      - ``multilingual-e5-large``        (1024-D, multilingual, free-tier eligible)
      - ``llama-text-embed-v2``          (default; English-strong, configurable dim)
      - ``pinecone-sparse-english-v0``   (sparse — not supported by this dense provider)

    For Voyage / OpenAI / Cohere via Pinecone's proxy, use the
    ``input_type``-aware vendor-specific provider names per Pinecone docs.

    Environment:

      ``PINECONE_API_KEY``  — required. Cloud key from app.pinecone.io.
      ``PINECONE_EMBEDDING_MODEL``       — model override (optional).
      ``PINECONE_EMBEDDING_DIMENSIONS``  — dim override (optional).
    """

    DEFAULT_MODEL = "llama-text-embed-v2"
    DEFAULT_DIM = 1024

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        input_type: str = "passage",
    ) -> None:
        try:
            from pinecone import Pinecone
        except ImportError:
            raise RuntimeError(
                "pinecone not installed. Run `pip install pinecone` "
                "or `pip install attestor[pinecone]`."
            )

        api_key = api_key or os.environ.get("PINECONE_API_KEY")
        if not api_key or api_key == "pclocal":
            raise RuntimeError(
                "PINECONE_API_KEY not set or set to the Pinecone-Local "
                "stub 'pclocal'. Pinecone Inference is cloud-only — get "
                "a real API key at app.pinecone.io.",
            )

        # Cloud only — no host override. Pinecone Local doesn't serve
        # the inference API per its own docs.
        self._client = Pinecone(api_key=api_key)
        self._model = (
            model
            or os.environ.get("PINECONE_EMBEDDING_MODEL")
            or self.DEFAULT_MODEL
        )
        self._input_type = input_type

        dim_env = os.environ.get("PINECONE_EMBEDDING_DIMENSIONS")
        if dimensions is not None:
            self._dimensions = int(dimensions)
        elif dim_env:
            self._dimensions = int(dim_env)
        else:
            self._dimensions = self.DEFAULT_DIM

        # Probe — confirms key + model + dim before any production call.
        # llama-text-embed-v2 supports {384, 512, 768, 1024, 2048}; an
        # unsupported dim raises here loudly rather than at runtime.
        result = self._client.inference.embed(
            model=self._model,
            inputs=["dim"],
            parameters={
                "input_type": self._input_type,
                "dimension": self._dimensions,
            },
        )
        self._dimension = len(result.data[0].values)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return f"pinecone:{self._model}@{self._dimension}d"

    def _params(self) -> dict[str, Any]:
        return {
            "input_type": self._input_type,
            "dimension": self._dimensions,
        }

    def embed(self, text: str) -> list[float]:
        result = self._client.inference.embed(
            model=self._model, inputs=[text], parameters=self._params(),
        )
        return list(result.data[0].values)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        result = self._client.inference.embed(
            model=self._model, inputs=texts, parameters=self._params(),
        )
        return [list(d.values) for d in result.data]


# ═══════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════


def _try_voyage() -> EmbeddingProvider | None:
    """Try Voyage AI (Anthropic's recommended embedder partner).

    Activated by VOYAGE_API_KEY. Configurable via:
        VOYAGE_API_KEY              required
        VOYAGE_EMBEDDING_MODEL      default ``voyage-4`` (also: voyage-4-large/lite/nano,
                                    voyage-3.5, voyage-code-3, voyage-finance-2, voyage-law-2)
        VOYAGE_EMBEDDING_DIMENSIONS default 1024 (model default; voyage-4 family supports
                                    256/512/1024/2048 — pick to match your pgvector schema)
        VOYAGE_INPUT_TYPE           default ``document``
    """
    if not os.environ.get("VOYAGE_API_KEY"):
        return None
    try:
        provider = VoyageEmbeddingProvider(
            input_type=os.environ.get("VOYAGE_INPUT_TYPE", "document"),
        )
        logger.info(
            "Using %s embeddings (%dD)",
            provider.provider_name, provider.dimension,
        )
        return provider
    except Exception as e:
        logger.debug("Voyage embeddings unavailable: %s", e)
        return None


def _try_pinecone_inference() -> EmbeddingProvider | None:
    """Try Pinecone Inference (cloud-only — Local Docker doesn't serve it).

    Activated by ATTESTOR_PREFER_EMBEDDER=pinecone OR by configure_embedder
    setting PINECONE_EMBEDDING_MODEL when stack.embedder.provider="pinecone".
    Configurable via:
        PINECONE_API_KEY                   required (cloud key from app.pinecone.io)
        PINECONE_EMBEDDING_MODEL           default ``llama-text-embed-v2``
        PINECONE_EMBEDDING_DIMENSIONS      default 1024 (matches v4 schema)
        PINECONE_EMBEDDING_INPUT_TYPE      default ``passage``
    """
    if not os.environ.get("PINECONE_API_KEY"):
        return None
    if os.environ.get("PINECONE_API_KEY") == "pclocal":
        return None  # Local Docker stub doesn't serve Inference
    try:
        provider = PineconeEmbeddingProvider(
            input_type=os.environ.get("PINECONE_EMBEDDING_INPUT_TYPE", "passage"),
        )
        logger.info(
            "Using %s embeddings (%dD)",
            provider.provider_name, provider.dimension,
        )
        return provider
    except Exception as e:
        logger.debug("Pinecone Inference unavailable: %s", e)
        return None


def _try_ollama() -> EmbeddingProvider | None:
    """Probe local Ollama. Returns None if daemon unreachable or model
    missing — caller falls through to the next provider."""
    if os.environ.get("ATTESTOR_DISABLE_LOCAL_EMBED") in {"1", "true", "True"}:
        return None
    try:
        provider = OllamaEmbeddingProvider()
        logger.info(
            "Using %s embeddings (%dD)",
            provider.provider_name, provider.dimension,
        )
        return provider
    except Exception as e:
        logger.debug("Ollama embeddings unavailable: %s", e)
        return None


def _try_openai() -> EmbeddingProvider | None:
    """Try OpenAI (direct, no OpenRouter fallback).

    Controlled by env:
        OPENAI_API_KEY              required
        OPENAI_EMBEDDING_MODEL      default ``text-embedding-3-large``
        OPENAI_EMBEDDING_DIMENSIONS default ``1536`` (Matryoshka reduction)

    Returns ``None`` when ``OPENAI_API_KEY`` is unset; the strict dispatch
    in ``get_embedding_provider`` then raises with a config-pointing message.
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key:
        return None

    model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-large")
    dim_env = os.environ.get("OPENAI_EMBEDDING_DIMENSIONS", "1536")
    try:
        dimensions: int | None = int(dim_env) if dim_env else None
    except ValueError:
        dimensions = None

    try:
        provider = OpenAIEmbeddingProvider(
            api_key=openai_key,
            model=model,
            dimensions=dimensions,
        )
        logger.info("Using %s (%dD)", model, provider.dimension)
        return provider
    except Exception as e:
        logger.warning("OpenAI embeddings failed: %s", e)
        return None


_CLOUD_PROVIDERS = {
    "voyage": _try_voyage,
    "pinecone": _try_pinecone_inference,
}

# Module-level cache — prevents re-initialising the embedding provider
# (e.g. API sessions) on every call.
_cached_provider: EmbeddingProvider | None = None


def clear_embedding_cache() -> None:
    """Clear the cached embedding provider (for testing)."""
    global _cached_provider
    _cached_provider = None


_PROVIDER_DISPATCH = {
    "openai": _try_openai,
    "voyage": _try_voyage,
    "pinecone": _try_pinecone_inference,
    "ollama": _try_ollama,
}


def get_embedding_provider(
    preferred: str | None = None,
) -> EmbeddingProvider:
    """Get the configured embedding provider.

    YAML's ``stack.embedder.provider`` is authoritative. There is no
    auto-detect chain. Calling with no ``preferred`` reads from YAML;
    calling with ``preferred=X`` is the explicit override path. Init
    failure raises ``RuntimeError`` with a config-pointing message —
    silent fallthrough to a different backend is deliberately gone
    because it caused subtle index-dim drift in past runs.

    Args:
        preferred: Provider name — one of openai / voyage / pinecone /
                   ollama / bedrock / azure_openai / vertex_ai. When
                   ``None``, the value is read from
                   ``configs/attestor.yaml`` ``stack.embedder.provider``.

    Returns:
        An ``EmbeddingProvider`` instance (cached after first call).

    Raises:
        RuntimeError: if the configured provider name is unknown or the
            matching ``_try_*`` helper fails to initialize (e.g. missing
            API key).
    """
    global _cached_provider

    if _cached_provider is not None and preferred is None:
        return _cached_provider

    # No preferred argument — read it from YAML.
    if preferred is None:
        from attestor.config import get_stack

        cfg = get_stack().embedder
        return get_embedding_provider(preferred=cfg.provider)

    # Strict dispatch: only the named provider is tried. No fallthrough.
    try_fn = _PROVIDER_DISPATCH.get(preferred)
    if try_fn is None:
        raise RuntimeError(
            f"unknown embedder provider {preferred!r}; expected one of "
            f"{sorted(_PROVIDER_DISPATCH)}. Check stack.embedder.provider "
            f"in configs/attestor.yaml."
        )
    provider = try_fn()
    if provider is None:
        raise RuntimeError(
            f"embedder provider {preferred!r} (set in configs/attestor.yaml "
            f"under stack.embedder.provider) failed to initialize. Check "
            f"that the matching API key / env vars are set for that provider."
        )
    logger.info(
        "Using %s embeddings (%dD) [pinned by config]",
        provider.provider_name, provider.dimension,
    )
    _cached_provider = provider
    return provider
