"""AWS backend — DynamoDB (document) + OpenSearch Serverless (vector) + Neptune (graph).

Each AWS service is optional; the backend degrades gracefully if a service
endpoint is not configured. Auth uses the boto3 credential chain throughout.

The role implementations live in three private mixin modules to keep this
file focused on lifecycle (init / close / health) — see
``_aws_document.py``, ``_aws_vector.py``, and ``_aws_graph.py``.
"""

from __future__ import annotations

import logging
from typing import Any, ClassVar

from attestor.store._aws_document import (
    _AWSDocumentMixin,
    _decimal_to_float,
    _float_to_decimal,
)
from attestor.store._aws_graph import _AWSGraphMixin
from attestor.store._aws_vector import _AWSVectorMixin
from attestor.store._graph_utils import sanitize_rel_type as _sanitize_rel_type

# Re-export module-level helpers for backwards compatibility with callers /
# tests that import them directly from ``attestor.store.aws_backend``.
__all__ = [
    "AWSBackend",
    "_decimal_to_float",
    "_float_to_decimal",
    "_sanitize_rel_type",
]

logger = logging.getLogger("attestor")


class AWSBackend(_AWSDocumentMixin, _AWSVectorMixin, _AWSGraphMixin):
    """Multi-role AWS backend: DynamoDB + OpenSearch Serverless + Neptune.

    Accepts raw config dict with keys:
        region, dynamodb.table_prefix, opensearch.endpoint, opensearch.index,
        neptune.endpoint

    Auth: boto3 credential chain (env vars, IAM role, profiles — no password).
    """

    ROLES: ClassVar[set[str]] = {"document", "vector", "graph"}

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._parse_config(config)

        # Lazy imports — fail only when actually used
        import boto3

        self._session = boto3.Session(region_name=self._region)

        # Document: DynamoDB
        self._dynamodb = None
        self._table = None
        self._init_dynamodb()

        # Vector: OpenSearch Serverless
        self._opensearch = None
        self._init_opensearch()

        # Graph: Neptune (HTTP openCypher)
        self._neptune_endpoint = self._config_neptune_endpoint
        self._neptune_auth = None
        self._init_neptune()

        # Embeddings: lazy
        self._embedder = None
        self._embedding_fn = None

    def _parse_config(self, config: dict[str, Any]) -> None:
        """Extract config values with defaults."""
        self._region = config.get("region", "us-east-1")

        ddb = config.get("dynamodb", {})
        prefix = ddb.get("table_prefix", "attestor")
        self._table_name = f"{prefix}_memories"

        os_cfg = config.get("opensearch", {})
        self._opensearch_endpoint = os_cfg.get("endpoint", "")
        self._opensearch_index = os_cfg.get("index", "memories")

        neptune = config.get("neptune", {})
        self._config_neptune_endpoint = neptune.get("endpoint", "")

        tls = config.get("tls", {})
        self._tls_verify = tls.get("verify", True)

    # ══════════════════════════════════════════════════════════════════
    # DynamoDB Init
    # ══════════════════════════════════════════════════════════════════

    def _init_dynamodb(self) -> None:
        """Connect to DynamoDB and auto-create table if needed."""
        try:
            self._dynamodb = self._session.resource("dynamodb")
            self._ensure_table()
        except Exception as e:
            logger.warning("DynamoDB init failed (document store unavailable): %s", e)
            self._dynamodb = None

    def _ensure_table(self) -> None:
        """Create memories table with GSIs if it doesn't exist."""
        import botocore.exceptions

        try:
            table = self._dynamodb.Table(self._table_name)
            table.load()
            self._table = table
            logger.debug("DynamoDB table %r exists", self._table_name)
            return
        except botocore.exceptions.ClientError as e:
            if e.response["Error"]["Code"] != "ResourceNotFoundException":
                raise

        logger.info("Creating DynamoDB table %r", self._table_name)
        table = self._dynamodb.create_table(
            TableName=self._table_name,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[
                {"AttributeName": "id", "AttributeType": "S"},
                {"AttributeName": "status", "AttributeType": "S"},
                {"AttributeName": "category", "AttributeType": "S"},
                {"AttributeName": "entity", "AttributeType": "S"},
                {"AttributeName": "created_at", "AttributeType": "S"},
            ],
            GlobalSecondaryIndexes=[
                {
                    "IndexName": "status-created_at-index",
                    "KeySchema": [
                        {"AttributeName": "status", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
                {
                    "IndexName": "category-created_at-index",
                    "KeySchema": [
                        {"AttributeName": "category", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
                {
                    "IndexName": "entity-created_at-index",
                    "KeySchema": [
                        {"AttributeName": "entity", "KeyType": "HASH"},
                        {"AttributeName": "created_at", "KeyType": "RANGE"},
                    ],
                    "Projection": {"ProjectionType": "ALL"},
                },
            ],
            BillingMode="PAY_PER_REQUEST",
        )
        table.wait_until_exists()
        self._table = table
        logger.info("DynamoDB table %r created", self._table_name)

    # ══════════════════════════════════════════════════════════════════
    # OpenSearch Serverless Init
    # ══════════════════════════════════════════════════════════════════

    def _init_opensearch(self) -> None:
        """Connect to OpenSearch Serverless with SigV4 auth."""
        if not self._opensearch_endpoint:
            logger.debug("No opensearch.endpoint configured — vector store unavailable")
            return

        try:
            from opensearchpy import OpenSearch, RequestsHttpConnection
            from requests_aws4auth import AWS4Auth

            credentials = self._session.get_credentials().get_frozen_credentials()
            auth = AWS4Auth(
                credentials.access_key,
                credentials.secret_key,
                self._region,
                "aoss",
                session_token=credentials.token,
            )

            self._opensearch = OpenSearch(
                hosts=[{"host": self._opensearch_endpoint, "port": 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=self._tls_verify,
                connection_class=RequestsHttpConnection,
            )
            self._ensure_opensearch_index()
        except Exception as e:
            logger.warning("OpenSearch init failed (vector store unavailable): %s", e)
            self._opensearch = None

    def _ensure_opensearch_index(self) -> None:
        """Create k-NN index with HNSW engine if it doesn't exist."""
        if self._opensearch.indices.exists(self._opensearch_index):
            return

        self._ensure_embedding_fn()
        dim = self._embedder.dimension

        body = {
            "settings": {
                "index": {
                    "knn": True,
                    "knn.algo_param.ef_search": 100,
                }
            },
            "mappings": {
                "properties": {
                    "memory_id": {"type": "keyword"},
                    "content": {"type": "text"},
                    "namespace": {"type": "keyword"},
                    "embedding": {
                        "type": "knn_vector",
                        "dimension": dim,
                        "method": {
                            "name": "hnsw",
                            "space_type": "cosinesimil",
                            "engine": "nmslib",
                            "parameters": {"ef_construction": 128, "m": 24},
                        },
                    },
                }
            },
        }
        self._opensearch.indices.create(index=self._opensearch_index, body=body)
        logger.info("OpenSearch index %r created (%dD)", self._opensearch_index, dim)

    # ══════════════════════════════════════════════════════════════════
    # Neptune Init
    # ══════════════════════════════════════════════════════════════════

    def _init_neptune(self) -> None:
        """Prepare Neptune HTTP openCypher auth."""
        if not self._neptune_endpoint:
            logger.debug("No neptune.endpoint configured — graph store unavailable")
            return

        try:

            self._neptune_auth = True  # marker that Neptune is configured
            logger.debug("Neptune endpoint configured: %s", self._neptune_endpoint)
        except Exception as e:
            logger.warning("Neptune init failed (graph store unavailable): %s", e)
            self._neptune_auth = None

    # ══════════════════════════════════════════════════════════════════
    # Embeddings (shared provider)
    # ══════════════════════════════════════════════════════════════════

    def _ensure_embedding_fn(self) -> None:
        """Lazy-init embedding provider via shared module."""
        if self._embedder is not None:
            return

        from attestor.store.embeddings import get_embedding_provider

        # No hardcoded provider — read stack.embedder.provider from
        # configs/attestor.yaml (matches PostgresBackend's call site).
        # AWS deployments that want Bedrock should set it in YAML.
        self._embedder = get_embedding_provider()
        if self._embedder.provider_name == "openai":
            self._openai_client = getattr(self._embedder, "_client", True)
        self._embedding_fn = self._embedder

    def _embed(self, text: str) -> list[float]:
        """Generate embedding using the shared provider."""
        self._ensure_embedding_fn()
        return self._embedder.embed(text)

    # ══════════════════════════════════════════════════════════════════
    # Close / Cleanup
    # ══════════════════════════════════════════════════════════════════

    def close(self) -> None:
        """Clean up connections."""
        self._table = None
        self._dynamodb = None
        if self._opensearch is not None:
            try:
                self._opensearch.close()
            except Exception:
                pass
            self._opensearch = None
        self._neptune_auth = None
