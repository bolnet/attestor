"""Pinecone vector backend (experiment) — implements ``VectorStore``.

Drop-in alternative to ``PostgresBackend``'s vector role. Used for the
bakeoff against pgvector on the LME-S workload. See
``attestor/store/postgres_backend.py`` for the reference implementation
of the same Protocol surface.

Two operating modes:

  - **Pinecone Local** (default for the bakeoff): connect to the
    Docker emulator at ``http://localhost:5080``. No API key, no cloud
    spend, no persistence. Boot via:

        docker run -d --name pinecone-local -e PORT=5080 \
            -e PINECONE_HOST=localhost -p 5080-5090:5080-5090 \
            --platform linux/amd64 ghcr.io/pinecone-io/pinecone-local:latest

  - **Pinecone Cloud**: set ``host=None`` (or omit) and provide
    ``api_key`` to reach api.pinecone.io. Same code path; different
    endpoint.

Same Voyage ``voyage-4`` (1024-D) embeddings as the pgvector path —
only the index/store layer differs. Embedding is delegated to
``attestor.store.embeddings.get_embedding_provider()`` so swapping the
embedder is one config change away.

This is an experimental adapter — NOT yet promoted to the canonical
backend list. Default Attestor stack stays PG+pgvector+Neo4j per
``feedback_only_pg_neo4j_stack`` until the bakeoff produces a clear win.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("attestor.store.pinecone")


_DEFAULT_LOCAL_HOST = "http://localhost:5080"
_DEFAULT_INDEX_NAME = "attestor-experiment"
_DEFAULT_CLOUD = "aws"
_DEFAULT_REGION = "us-east-1"


class PineconeBackend:
    """Vector storage + similarity search backed by Pinecone.

    Implements the ``VectorStore`` Protocol surface (``add``,
    ``search``, ``delete``, ``count``, ``close``) plus a few helpers
    for hermetic test runs.
    """

    ROLES: Set[str] = {"vector"}

    def __init__(self, config: Dict[str, Any]) -> None:
        """Construct a Pinecone-backed vector store.

        Config keys:
            host           HTTP(S) base URL. Default: ``http://localhost:5080``
                          (Pinecone Local). Set to None / "cloud" to use
                          api.pinecone.io.
            api_key        Pinecone API key. ``"pclocal"`` works for the
                          local emulator (it's ignored). Required for
                          cloud.
            index_name     Index to upsert/query against. Created if
                          missing. Default: ``attestor-experiment``.
            dimension      Vector dim. Default: 1024 (matches Voyage
                          ``voyage-4``). Index creation uses this.
            metric         ``cosine`` | ``euclidean`` | ``dotproduct``.
                          Default: ``cosine``.
            cloud          Serverless cloud (cloud mode only).
                          Default: ``aws``.
            region         Serverless region. Default: ``us-east-1``.
            embedder       Embedding provider (any callable returning
                          ``List[float]`` from ``embed(text)``).
                          When None, resolved via
                          ``get_embedding_provider()`` — same chain as
                          the pgvector path.

        Local vs cloud transport: Pinecone Local's index endpoints
        are plain HTTP/gRPC (no TLS); cloud is HTTPS-only. We auto-
        detect by inspecting ``host`` — when it contains ``localhost``
        or ``127.`` we use the gRPC client with ``secure=False``.
        """
        self._config = config
        self._host = config.get("host", _DEFAULT_LOCAL_HOST)
        self._api_key = config.get("api_key") or os.environ.get(
            "PINECONE_API_KEY", "pclocal",
        )
        self._index_name = config.get("index_name", _DEFAULT_INDEX_NAME)
        self._dimension = int(config.get("dimension", 1024))
        self._metric = config.get("metric", "cosine")
        self._cloud = config.get("cloud", _DEFAULT_CLOUD)
        self._region = config.get("region", _DEFAULT_REGION)

        # Local mode → gRPC + insecure transport; cloud → standard HTTP.
        self._is_local = bool(self._host) and (
            "localhost" in str(self._host) or "127." in str(self._host)
        )

        if self._is_local:
            from pinecone.grpc import PineconeGRPC, GRPCClientConfig
            from pinecone import ServerlessSpec
            self._grpc_config_cls = GRPCClientConfig
            self._serverless_spec_cls = ServerlessSpec
            self._pc = PineconeGRPC(api_key=self._api_key, host=self._host)
        else:
            from pinecone import Pinecone, ServerlessSpec
            self._serverless_spec_cls = ServerlessSpec
            if self._host in (None, "cloud"):
                self._pc = Pinecone(api_key=self._api_key)
            else:
                self._pc = Pinecone(api_key=self._api_key, host=self._host)

        # Create the index if missing. Pinecone Local accepts the same
        # ServerlessSpec shape as cloud — region/cloud are recorded
        # but irrelevant for the emulator.
        existing = {i.name for i in self._pc.list_indexes()}
        if self._index_name not in existing:
            logger.info(
                "Pinecone: creating index %r (dim=%d, metric=%s)",
                self._index_name, self._dimension, self._metric,
            )
            self._pc.create_index(
                name=self._index_name,
                dimension=self._dimension,
                metric=self._metric,
                spec=self._serverless_spec_cls(
                    cloud=self._cloud, region=self._region,
                ),
            )
            # Wait for index to be ready — cloud takes a few seconds;
            # local is instant. Bounded poll keeps tests deterministic.
            self._wait_until_ready(timeout=30.0)

        # Bind the data-plane client. For Local, route to the index's
        # localhost port via gRPC + insecure config.
        if self._is_local:
            desc = self._pc.describe_index(self._index_name)
            self._index = self._pc.Index(
                host=desc.host,
                grpc_config=self._grpc_config_cls(secure=False),
            )
        else:
            self._index = self._pc.Index(name=self._index_name)

        # Embedder — same provider chain as pgvector for an apples-to-
        # apples comparison.
        self._embedder = config.get("embedder")
        if self._embedder is None:
            from attestor.store.embeddings import get_embedding_provider
            self._embedder = get_embedding_provider()

    # ── helpers ────────────────────────────────────────────────────

    def _wait_until_ready(self, timeout: float = 30.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                desc = self._pc.describe_index(self._index_name)
                if getattr(desc.status, "ready", False):
                    return
            except Exception as e:  # noqa: BLE001
                logger.debug(
                    "Pinecone: describe_index probe failed: %s", e,
                )
            time.sleep(0.5)
        raise TimeoutError(
            f"Pinecone index {self._index_name!r} not ready within {timeout}s",
        )

    def _embed(self, text: str) -> List[float]:
        """Run the configured embedder. Defensive: a missing embedder
        is a hard error here — without an embedding we can't write."""
        if self._embedder is None:
            raise RuntimeError(
                "Pinecone backend has no embedder configured; provide "
                "config['embedder'] or set up the canonical embedding "
                "provider chain.",
            )
        return list(self._embedder.embed(text))

    # ── VectorStore Protocol ──────────────────────────────────────

    def add(
        self, memory_id: str, content: str, namespace: str = "default",
    ) -> None:
        """Upsert one (memory_id, embedding(content)) record.

        ``namespace`` flows through to the Pinecone namespace param —
        same scoping shape as pgvector. Default namespace ("default")
        matches the Postgres path's behavior.
        """
        embedding = self._embed(content)
        self._index.upsert(
            vectors=[{
                "id": memory_id,
                "values": embedding,
                "metadata": {"namespace": namespace},
            }],
            namespace=namespace,
        )

    def upsert_with_embedding(
        self,
        memory_id: str,
        embedding: List[float],
        namespace: str = "default",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Upsert with a precomputed embedding — bypasses the embedder.

        Used by harness code that already paid the embedding cost
        (e.g. the bakeoff's hermetic re-runs that re-use the same
        Voyage cache across pgvector AND Pinecone).
        """
        meta = {"namespace": namespace, **(metadata or {})}
        self._index.upsert(
            vectors=[{"id": memory_id, "values": embedding, "metadata": meta}],
            namespace=namespace,
        )

    def search(
        self,
        query_text: str,
        limit: int = 20,
        namespace: Optional[str] = None,
        **kwargs: Any,  # accept and ignore as_of/time_window — unsupported here
    ) -> List[Dict[str, Any]]:
        """Cosine similarity search.

        Returns the same shape as pgvector's search output —
        ``[{memory_id, distance}, ...]`` — so the orchestrator can
        consume either backend interchangeably. The orchestrator
        derives ``vector_sim`` from ``distance`` downstream; we
        translate Pinecone's ``score`` (similarity, 0-1) to
        ``distance = 1 - score`` to match the convention.

        Bi-temporal kwargs (``as_of``, ``time_window``) are accepted
        but IGNORED — Pinecone has no temporal filter primitives. The
        orchestrator silently degrades to non-temporal search when
        those kwargs hit a backend that doesn't support them.
        """
        if kwargs.get("as_of") is not None or kwargs.get("time_window") is not None:
            logger.debug(
                "Pinecone: as_of/time_window passed but unsupported; "
                "ignoring (returning non-temporal results).",
            )

        query_vec = self._embed(query_text)
        ns = namespace or "default"
        result = self._index.query(
            vector=query_vec,
            top_k=int(limit),
            namespace=ns,
            include_values=False,
            include_metadata=False,
        )

        out: List[Dict[str, Any]] = []
        for match in (result.matches or []):
            score = float(getattr(match, "score", 0.0))
            out.append({
                "memory_id": getattr(match, "id", ""),
                "distance": max(0.0, 1.0 - score),  # cosine: dist = 1 - sim
            })
        return out

    def delete(self, memory_id: str) -> bool:
        """Delete one vector by id from the default namespace.

        Returns True for the common case (Pinecone delete is
        idempotent — it doesn't tell you whether the id existed).
        """
        try:
            self._index.delete(ids=[memory_id], namespace="default")
            return True
        except Exception as e:  # noqa: BLE001
            logger.debug("Pinecone delete failed for %s: %s", memory_id, e)
            return False

    def count(self) -> int:
        """Total vectors across all namespaces."""
        try:
            stats = self._index.describe_index_stats()
            return int(getattr(stats, "total_vector_count", 0))
        except Exception as e:  # noqa: BLE001
            logger.debug("Pinecone count failed: %s", e)
            return 0

    def close(self) -> None:
        """Pinecone SDK has no explicit close — connection is
        implicit per-request. Provided for Protocol compliance."""
        return None

    # ── helpers for hermetic test runs ────────────────────────────

    def reset(self) -> None:
        """Delete + recreate the index. Used between bakeoff fixtures
        so each case sees an empty index. Cheap against Pinecone Local
        (in-memory); avoid on cloud — it's slow."""
        from pinecone import ServerlessSpec

        try:
            self._pc.delete_index(self._index_name)
        except Exception as e:  # noqa: BLE001
            logger.debug("Pinecone reset: delete failed: %s", e)
        self._pc.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric=self._metric,
            spec=ServerlessSpec(cloud=self._cloud, region=self._region),
        )
        self._wait_until_ready()
        self._index = self._pc.Index(name=self._index_name)
