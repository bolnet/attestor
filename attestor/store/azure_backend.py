"""Azure Cosmos DB backend — document + vector (DiskANN) + graph (NetworkX) in one account.

Uses Cosmos DB NoSQL API for document and vector storage, with an in-memory
NetworkX graph persisted to Cosmos containers for the graph role.

Requires: azure-cosmos, azure-identity (optional, for DefaultAzureCredential)
"""

from __future__ import annotations

import json  # noqa: F401  (preserved for backwards-compat with prior module surface)
import logging
from typing import Any, ClassVar

from attestor.store._azure_document import _AzureDocumentMixin
from attestor.store._azure_graph import _AzureGraphMixin
from attestor.store._azure_vector import _AzureVectorMixin
from attestor.store._graph_utils import sanitize_rel_type as _sanitize_rel_type  # noqa: F401

logger = logging.getLogger("attestor")


class AzureBackend(_AzureDocumentMixin, _AzureVectorMixin, _AzureGraphMixin):
    """Multi-role Azure Cosmos DB backend: document + vector + graph.

    Architecture:
        - DocumentStore: Cosmos DB NoSQL API, container "memories", PK /category
        - VectorStore: DiskANN vector index on /embedding field (same container)
        - GraphStore: Cosmos containers "graph_entities" (PK /entity_type) and
          "graph_edges" (PK /from_key) + in-memory NetworkX MultiDiGraph

    Config keys:
        cosmos_endpoint: Cosmos DB account endpoint (or env AZURE_COSMOS_ENDPOINT)
        cosmos_key: Cosmos DB account key (or env AZURE_COSMOS_KEY)
        cosmos_database: Database name (default "attestor")

    When no cosmos_key is provided, falls back to DefaultAzureCredential.

    Tenancy note:
        The graph role is an in-process NetworkX MultiDiGraph and this
        deploy mode is single-tenant by assumption (one Azure account ==
        one tenant). The orchestrator's ``namespace`` kwarg is ignored
        here intentionally — adding multi-tenancy would require splitting
        the in-memory graph per namespace and re-loading from Cosmos on
        every recall. If a multi-tenant Azure deploy is needed, switch
        the graph role to the Neo4j backend, which enforces namespace
        scoping at the Cypher layer.
    """

    ROLES: ClassVar[set[str]] = {"document", "vector", "graph"}

    def __init__(self, config: dict[str, Any]) -> None:
        import os

        self._config = config

        # Resolve endpoint and key
        self._endpoint = (
            config.get("cosmos_endpoint")
            or os.environ.get("AZURE_COSMOS_ENDPOINT", "")
        )
        if not self._endpoint:
            raise ValueError(
                "Azure Cosmos DB endpoint required. Set cosmos_endpoint in config "
                "or AZURE_COSMOS_ENDPOINT environment variable."
            )

        cosmos_key = (
            config.get("cosmos_key")
            or config.get("auth", {}).get("api_key")
            or os.environ.get("AZURE_COSMOS_KEY", "")
        )
        self._database_name = config.get("cosmos_database", "attestor")

        # Lazy import azure.cosmos
        from azure.cosmos import CosmosClient, PartitionKey

        self._PartitionKey = PartitionKey

        if cosmos_key:
            self._client = CosmosClient(self._endpoint, credential=cosmos_key)
        else:
            # Fall back to DefaultAzureCredential (managed identity, CLI, etc.)
            from azure.identity import DefaultAzureCredential

            credential = DefaultAzureCredential()
            self._client = CosmosClient(self._endpoint, credential=credential)

        self._init_database()
        self._init_containers()
        self._init_graph_memory()

        # Embedding provider — lazy init
        self._embedder = None

    # ── Initialization ──

    def _init_database(self) -> None:
        """Ensure database exists (create if not)."""
        self._database = self._client.create_database_if_not_exists(
            id=self._database_name
        )

    def _resolve_vector_dim(self) -> int:
        """Resolve the vector dim for the Cosmos DiskANN index.

        Order of precedence:
          1. ``config["vector_dim"]`` — explicit override.
          2. ``configs/attestor.yaml`` ``stack.embedder.dimensions`` —
             the canonical embedder dim used by the rest of the stack.
        """
        explicit = self._config.get("vector_dim")
        if explicit is not None:
            return int(explicit)
        # Fall back to the canonical embedder dim from YAML so the
        # Cosmos index matches whatever embedder is wired today.
        from attestor.config import get_stack

        return int(get_stack().embedder.dimensions)

    def _init_containers(self) -> None:
        """Create containers with appropriate partition keys and vector policies.

        The Cosmos DiskANN index `dimensions` MUST match the embedder's
        output dim. Resolution order:

          1. Explicit ``config["vector_dim"]`` — programmatic / test override.
          2. Canonical YAML — ``stack.embedder.dimensions`` from
             ``configs/attestor.yaml`` (loaded via ``attestor.config.get_stack``).

        Hardcoding the dim caused silent index/embedding drift on the
        canonical stack (1024-D Pinecone / Voyage / OpenAI-3-small@1024)
        — see project memory ``pgvector_dim_lesson``.
        """
        vector_dim = self._resolve_vector_dim()
        # Memories container — PK on /category
        # Vector embedding policy for DiskANN index on /embedding
        vector_embedding_policy = {
            "vectorEmbeddings": [
                {
                    "path": "/embedding",
                    "dataType": "float32",
                    "distanceFunction": "cosine",
                    "dimensions": vector_dim,
                }
            ]
        }
        indexing_policy = {
            "indexingMode": "consistent",
            "automatic": True,
            "includedPaths": [{"path": "/*"}],
            "excludedPaths": [{"path": "/embedding/*"}],
            "vectorIndexes": [
                {"path": "/embedding", "type": "diskANN"}
            ],
        }
        self._memories_container = self._database.create_container_if_not_exists(
            id="memories",
            partition_key=self._PartitionKey("/category"),
            indexing_policy=indexing_policy,
            vector_embedding_policy=vector_embedding_policy,
        )

        # Graph entities container — PK on /entity_type
        self._entities_container = self._database.create_container_if_not_exists(
            id="graph_entities",
            partition_key=self._PartitionKey("/entity_type"),
        )

        # Graph edges container — PK on /from_key
        self._edges_container = self._database.create_container_if_not_exists(
            id="graph_edges",
            partition_key=self._PartitionKey("/from_key"),
        )

    def _init_graph_memory(self) -> None:
        """Load all entities and edges from Cosmos into an in-memory NetworkX graph."""
        try:
            import networkx as nx
        except ImportError:
            raise ImportError(
                "networkx is required for the Azure graph role. "
                "Install with: pip install 'attestor[azure]' or pip install networkx"
            )

        self._graph = nx.MultiDiGraph()

        # Load entities
        try:
            entities = list(self._entities_container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True,
            ))
            for entity in entities:
                key = entity.get("key", "")
                attrs = {
                    k: v for k, v in entity.items()
                    if k not in ("id", "key", "display_name", "entity_type", "_rid", "_self", "_etag", "_attachments", "_ts")
                }
                self._graph.add_node(
                    key,
                    display_name=entity.get("display_name", key),
                    entity_type=entity.get("entity_type", "general"),
                    **attrs,
                )
        except Exception as e:
            logger.debug("Could not load graph entities: %s", e)

        # Load edges
        try:
            edges = list(self._edges_container.query_items(
                query="SELECT * FROM c",
                enable_cross_partition_query=True,
            ))
            for edge in edges:
                from_key = edge.get("from_key", "")
                to_key = edge.get("to_key", "")
                rel_type = edge.get("relation_type", "RELATED_TO")
                meta = {
                    k: v for k, v in edge.items()
                    if k not in ("id", "from_key", "to_key", "relation_type", "_rid", "_self", "_etag", "_attachments", "_ts")
                }
                meta["relation_type"] = rel_type
                self._graph.add_edge(from_key, to_key, key=rel_type, **meta)
        except Exception as e:
            logger.debug("Could not load graph edges: %s", e)

    # ── Embedding Helpers ──

    def _ensure_embedding_fn(self) -> None:
        """Lazy-init embedding provider via shared module."""
        if self._embedder is not None:
            return
        from attestor.store.embeddings import get_embedding_provider

        # Provider comes from configs/attestor.yaml (stack.embedder.provider).
        # AWS/Azure/Postgres all share the same call shape — one source of truth.
        self._embedder = get_embedding_provider()

    def _embed(self, text: str) -> list[float]:
        """Generate embedding using the shared provider."""
        self._ensure_embedding_fn()
        return self._embedder.embed(text)

    # ── Lifecycle ──

    def save(self) -> None:
        """No-op — writes are immediate (write-through to Cosmos)."""
        pass

    def close(self) -> None:
        """Close the Cosmos client."""
        try:
            self._client.close()
        except Exception:
            pass
