"""AWS backend — DynamoDB (document) + OpenSearch Serverless (vector) + Neptune (graph).

Each AWS service is optional; the backend degrades gracefully if a service
endpoint is not configured. Auth uses the boto3 credential chain throughout.
"""

from __future__ import annotations

import json
import logging
import re
import time
from decimal import Decimal
from typing import Any, Dict, List, Optional, Set

from attestor.models import Memory
from attestor.store.base import DocumentStore, GraphStore, VectorStore

logger = logging.getLogger("attestor")

# DynamoDB stores Decimal, not float — helpers to convert
_FLOAT_FIELDS = {"confidence"}


def _float_to_decimal(value: Any) -> Any:
    """Convert float values to Decimal for DynamoDB compatibility."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, dict):
        return {k: _float_to_decimal(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_float_to_decimal(v) for v in value]
    return value


def _decimal_to_float(value: Any) -> Any:
    """Convert Decimal values back to float from DynamoDB."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, dict):
        return {k: _decimal_to_float(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decimal_to_float(v) for v in value]
    return value


def _sanitize_rel_type(rel_type: str) -> str:
    """Normalize relation type: uppercase, safe characters only."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", rel_type)
    return sanitized.upper()


class AWSBackend(DocumentStore, VectorStore, GraphStore):
    """Multi-role AWS backend: DynamoDB + OpenSearch Serverless + Neptune.

    Accepts raw config dict with keys:
        region, dynamodb.table_prefix, opensearch.endpoint, opensearch.index,
        neptune.endpoint

    Auth: boto3 credential chain (env vars, IAM role, profiles — no password).
    """

    ROLES: Set[str] = {"document", "vector", "graph"}

    def __init__(self, config: Dict[str, Any]) -> None:
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

    def _parse_config(self, config: Dict[str, Any]) -> None:
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
            from botocore.auth import SigV4Auth
            from botocore.awsrequest import AWSRequest

            self._neptune_auth = True  # marker that Neptune is configured
            logger.debug("Neptune endpoint configured: %s", self._neptune_endpoint)
        except Exception as e:
            logger.warning("Neptune init failed (graph store unavailable): %s", e)
            self._neptune_auth = None

    def _neptune_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Execute an openCypher query against Neptune via HTTP POST."""
        import requests
        from botocore.auth import SigV4Auth
        from botocore.awsrequest import AWSRequest

        url = f"https://{self._neptune_endpoint}:8182/openCypher"
        data = {"query": query}
        if params:
            data["parameters"] = json.dumps(params)

        # Sign the request with SigV4
        credentials = self._session.get_credentials().get_frozen_credentials()
        aws_request = AWSRequest(method="POST", url=url, data=data)
        SigV4Auth(credentials, "neptune-db", self._region).add_auth(aws_request)

        headers = dict(aws_request.headers)
        response = requests.post(url, data=data, headers=headers, verify=self._tls_verify)
        response.raise_for_status()

        result = response.json()
        return result.get("results", [])

    # ══════════════════════════════════════════════════════════════════
    # Embeddings (shared provider)
    # ══════════════════════════════════════════════════════════════════

    def _ensure_embedding_fn(self) -> None:
        """Lazy-init embedding provider via shared module."""
        if self._embedder is not None:
            return

        from attestor.store.embeddings import get_embedding_provider

        self._embedder = get_embedding_provider("bedrock")
        if self._embedder.provider_name == "openai":
            self._openai_client = getattr(self._embedder, "_client", True)
        self._embedding_fn = self._embedder

    def _embed(self, text: str) -> List[float]:
        """Generate embedding using the shared provider."""
        self._ensure_embedding_fn()
        return self._embedder.embed(text)

    # ══════════════════════════════════════════════════════════════════
    # DocumentStore — DynamoDB
    # ══════════════════════════════════════════════════════════════════

    def _memory_to_item(self, memory: Memory) -> Dict[str, Any]:
        """Convert Memory to DynamoDB item dict."""
        item: Dict[str, Any] = {
            "id": memory.id,
            "content": memory.content,
            "tags": memory.tags,
            "category": memory.category,
            "namespace": memory.namespace,
            "created_at": memory.created_at,
            "valid_from": memory.valid_from,
            "confidence": _float_to_decimal(memory.confidence),
            "status": memory.status,
            "metadata": _float_to_decimal(memory.metadata) if memory.metadata else {},
        }
        # Optional fields — DynamoDB can't store None for GSI key attributes
        if memory.entity:
            item["entity"] = memory.entity
        if memory.event_date:
            item["event_date"] = memory.event_date
        if memory.valid_until:
            item["valid_until"] = memory.valid_until
        if memory.superseded_by:
            item["superseded_by"] = memory.superseded_by
        return item

    def _item_to_memory(self, item: Dict[str, Any]) -> Memory:
        """Convert DynamoDB item to Memory."""
        item = _decimal_to_float(item)
        metadata = item.get("metadata", {})
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return Memory(
            id=item["id"],
            content=item["content"],
            tags=item.get("tags", []),
            category=item.get("category", "general"),
            namespace=item.get("namespace", "default"),
            entity=item.get("entity"),
            created_at=item["created_at"],
            event_date=item.get("event_date"),
            valid_from=item.get("valid_from", item["created_at"]),
            valid_until=item.get("valid_until"),
            superseded_by=item.get("superseded_by"),
            confidence=item.get("confidence", 1.0),
            status=item.get("status", "active"),
            metadata=metadata,
        )

    def insert(self, memory: Memory) -> Memory:
        self._table.put_item(Item=self._memory_to_item(memory))
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        resp = self._table.get_item(Key={"id": memory_id})
        item = resp.get("Item")
        if item is None:
            return None
        return self._item_to_memory(item)

    def update(self, memory: Memory) -> Memory:
        self._table.put_item(Item=self._memory_to_item(memory))
        return memory

    def delete(self, memory_id: str) -> bool:
        resp = self._table.delete_item(
            Key={"id": memory_id},
            ReturnValues="ALL_OLD",
        )
        return "Attributes" in resp

    def list_memories(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        namespace: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        from boto3.dynamodb.conditions import Key, Attr

        # Use GSI queries when a single partition key filter is available and no namespace filter
        if status and not category and not entity and not namespace:
            return self._query_gsi(
                "status-created_at-index", "status", status,
                after=after, before=before, limit=limit,
            )
        if category and not status and not entity and not namespace:
            return self._query_gsi(
                "category-created_at-index", "category", category,
                after=after, before=before, limit=limit,
            )
        if entity and not status and not category and not namespace:
            return self._query_gsi(
                "entity-created_at-index", "entity", entity,
                after=after, before=before, limit=limit,
            )

        # Fallback: scan with filter
        filter_expr = None
        expr_values: Dict[str, Any] = {}
        expr_names: Dict[str, str] = {}
        conditions = []

        if status:
            conditions.append(Attr("status").eq(status))
        if category:
            conditions.append(Attr("category").eq(category))
        if entity:
            conditions.append(Attr("entity").eq(entity))
        if namespace:
            conditions.append(Attr("namespace").eq(namespace))
        if after:
            conditions.append(Attr("created_at").gte(after))
        if before:
            conditions.append(Attr("created_at").lte(before))

        if conditions:
            combined = conditions[0]
            for c in conditions[1:]:
                combined = combined & c
            filter_expr = combined

        scan_kwargs: Dict[str, Any] = {"Limit": limit}
        if filter_expr is not None:
            scan_kwargs["FilterExpression"] = filter_expr

        items = self._scan_all(scan_kwargs, limit)
        memories = [self._item_to_memory(item) for item in items]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def _query_gsi(
        self,
        index_name: str,
        pk_name: str,
        pk_value: str,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        """Query a GSI by partition key, optionally filtering by sort key range."""
        from boto3.dynamodb.conditions import Key

        key_expr = Key(pk_name).eq(pk_value)
        if after and before:
            key_expr = key_expr & Key("created_at").between(after, before)
        elif after:
            key_expr = key_expr & Key("created_at").gte(after)
        elif before:
            key_expr = key_expr & Key("created_at").lte(before)

        resp = self._table.query(
            IndexName=index_name,
            KeyConditionExpression=key_expr,
            Limit=limit,
            ScanIndexForward=False,
        )
        return [self._item_to_memory(item) for item in resp.get("Items", [])]

    def _scan_all(self, scan_kwargs: Dict[str, Any], limit: int) -> List[Dict[str, Any]]:
        """Scan with pagination, stopping at limit."""
        items: List[Dict[str, Any]] = []
        while True:
            resp = self._table.scan(**scan_kwargs)
            items.extend(resp.get("Items", []))
            if len(items) >= limit:
                break
            last_key = resp.get("LastEvaluatedKey")
            if not last_key:
                break
            scan_kwargs["ExclusiveStartKey"] = last_key
        return items[:limit]

    def tag_search(
        self,
        tags: List[str],
        category: Optional[str] = None,
        namespace: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        """Scan for memories whose tags overlap with the given list.

        DynamoDB doesn't have a native array-overlap operator, so we scan
        with a filter that checks membership of each tag.
        """
        from boto3.dynamodb.conditions import Attr

        # Build: status = active AND valid_until not exists AND (contains(tags, t1) OR ...)
        conditions = [
            Attr("status").eq("active"),
            Attr("valid_until").not_exists(),
        ]

        tag_conditions = [Attr("tags").contains(t) for t in tags]
        if tag_conditions:
            tag_filter = tag_conditions[0]
            for tc in tag_conditions[1:]:
                tag_filter = tag_filter | tc
            conditions.append(tag_filter)

        if category:
            conditions.append(Attr("category").eq(category))
        if namespace:
            conditions.append(Attr("namespace").eq(namespace))

        combined = conditions[0]
        for c in conditions[1:]:
            combined = combined & c

        items = self._scan_all({"FilterExpression": combined, "Limit": limit}, limit)
        memories = [self._item_to_memory(item) for item in items]
        memories.sort(key=lambda m: m.created_at, reverse=True)
        return memories[:limit]

    def execute(
        self, query: str, params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError("Raw SQL not supported on DynamoDB")

    def archive_before(self, date: str) -> int:
        """Archive active memories created before the given date."""
        from boto3.dynamodb.conditions import Attr

        filter_expr = (
            Attr("status").eq("active") & Attr("created_at").lt(date)
        )
        items = self._scan_all({"FilterExpression": filter_expr, "Limit": 10000}, 10000)

        count = 0
        with self._table.batch_writer() as batch:
            for item in items:
                item["status"] = "archived"
                batch.put_item(Item=item)
                count += 1
        return count

    def compact(self) -> int:
        """Delete all archived memories."""
        from boto3.dynamodb.conditions import Attr

        filter_expr = Attr("status").eq("archived")
        items = self._scan_all({"FilterExpression": filter_expr, "Limit": 10000}, 10000)

        count = 0
        with self._table.batch_writer() as batch:
            for item in items:
                batch.delete_item(Key={"id": item["id"]})
                count += 1
        return count

    def stats(self) -> Dict[str, Any]:
        items = self._scan_all({"Limit": 100000}, 100000)

        by_status: Dict[str, int] = {}
        by_category: Dict[str, int] = {}
        for item in items:
            s = item.get("status", "active")
            by_status[s] = by_status.get(s, 0) + 1
            c = item.get("category", "general")
            by_category[c] = by_category.get(c, 0) + 1

        return {
            "total_memories": len(items),
            "by_status": by_status,
            "by_category": by_category,
        }

    # ══════════════════════════════════════════════════════════════════
    # VectorStore — OpenSearch Serverless
    # ══════════════════════════════════════════════════════════════════

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

    def search(self, query_text: str, limit: int = 20, namespace: Optional[str] = None) -> List[Dict[str, Any]]:
        if self._opensearch is None:
            logger.debug("OpenSearch not configured — returning empty results")
            return []

        query_vec = self._embed(query_text)
        knn_clause: Dict[str, Any] = {
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

    # ══════════════════════════════════════════════════════════════════
    # GraphStore — Neptune openCypher
    # ══════════════════════════════════════════════════════════════════

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._neptune_auth:
            logger.debug("Neptune not configured — skipping add_entity")
            return

        key = name.lower()
        attrs = dict(attributes) if attributes else {}

        # MERGE ensures idempotency
        props = ", ".join(
            f"{k}: '{self._escape(str(v))}'"
            for k, v in attrs.items()
        )
        set_clause = f", e.entity_type = '{self._escape(entity_type)}', e.display_name = '{self._escape(name)}'"
        if props:
            set_clause += ", " + ", ".join(
                f"e.{k} = '{self._escape(str(v))}'" for k, v in attrs.items()
            )

        query = (
            f"MERGE (e:Entity {{key: '{self._escape(key)}'}}) "
            f"ON CREATE SET e.display_name = '{self._escape(name)}', "
            f"e.entity_type = '{self._escape(entity_type)}'"
        )
        if attrs:
            on_create_props = ", ".join(
                f"e.{k} = '{self._escape(str(v))}'" for k, v in attrs.items()
            )
            query += f", {on_create_props}"

        query += f" ON MATCH SET e.display_name = '{self._escape(name)}'"
        # Only upgrade entity_type from general
        query += (
            f", e.entity_type = CASE WHEN e.entity_type = 'general' AND '{self._escape(entity_type)}' <> 'general' "
            f"THEN '{self._escape(entity_type)}' ELSE e.entity_type END"
        )
        if attrs:
            match_props = ", ".join(
                f"e.{k} = '{self._escape(str(v))}'" for k, v in attrs.items()
            )
            query += f", {match_props}"

        query += " RETURN e"
        self._neptune_query(query)

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not self._neptune_auth:
            logger.debug("Neptune not configured — skipping add_relation")
            return

        from_key = self._escape(from_entity.lower())
        to_key = self._escape(to_entity.lower())
        from_name = self._escape(from_entity)
        to_name = self._escape(to_entity)
        sanitized = _sanitize_rel_type(relation_type)

        # Ensure both entities exist
        self._neptune_query(
            f"MERGE (e:Entity {{key: '{from_key}'}}) "
            f"ON CREATE SET e.display_name = '{from_name}', e.entity_type = 'general' "
            f"RETURN e"
        )
        self._neptune_query(
            f"MERGE (e:Entity {{key: '{to_key}'}}) "
            f"ON CREATE SET e.display_name = '{to_name}', e.entity_type = 'general' "
            f"RETURN e"
        )

        # Create edge
        meta_props = f"relation_type: '{sanitized}'"
        if metadata:
            for k, v in metadata.items():
                meta_props += f", {k}: '{self._escape(str(v))}'"

        self._neptune_query(
            f"MATCH (a:Entity {{key: '{from_key}'}}), (b:Entity {{key: '{to_key}'}}) "
            f"CREATE (a)-[r:RELATION {{{meta_props}}}]->(b) RETURN r"
        )

    def get_related(self, entity: str, depth: int = 2) -> List[str]:
        if not self._neptune_auth:
            return []

        key = self._escape(entity.lower())

        # Check entity exists
        results = self._neptune_query(
            f"MATCH (e:Entity {{key: '{key}'}}) RETURN e"
        )
        if not results:
            return []

        results = self._neptune_query(
            f"MATCH (start:Entity {{key: '{key}'}})-[*1..{depth}]-(connected:Entity) "
            f"RETURN DISTINCT connected.display_name AS name"
        )
        return [r["name"] for r in results if r.get("name")]

    def get_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        if not self._neptune_auth:
            return {"entity": entity, "nodes": [], "edges": []}

        key = self._escape(entity.lower())

        # Check entity exists
        results = self._neptune_query(
            f"MATCH (e:Entity {{key: '{key}'}}) RETURN e"
        )
        if not results:
            return {"entity": entity, "nodes": [], "edges": []}

        # Get nodes
        node_results = self._neptune_query(
            f"MATCH (start:Entity {{key: '{key}'}})-[*0..{depth}]-(n:Entity) "
            f"RETURN DISTINCT n.key AS key, n.display_name AS name, n.entity_type AS type"
        )
        nodes = []
        node_keys = set()
        for r in node_results:
            nk = r.get("key")
            if nk and nk not in node_keys:
                node_keys.add(nk)
                nodes.append({
                    "name": r.get("name", nk),
                    "type": r.get("type", "general"),
                    "key": nk,
                })

        # Get edges between those nodes
        edges = []
        if len(node_keys) > 1:
            keys_list = ", ".join(f"'{self._escape(k)}'" for k in node_keys)
            edge_results = self._neptune_query(
                f"MATCH (a:Entity)-[r:RELATION]->(b:Entity) "
                f"WHERE a.key IN [{keys_list}] AND b.key IN [{keys_list}] "
                f"RETURN DISTINCT a.key AS source, b.key AS target, r.relation_type AS type"
            )
            for r in edge_results:
                edges.append({
                    "source": r.get("source", ""),
                    "target": r.get("target", ""),
                    "type": r.get("type", "RELATION"),
                })

        return {"entity": entity, "nodes": nodes, "edges": edges}

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if not self._neptune_auth:
            return []

        if entity_type:
            et = self._escape(entity_type)
            results = self._neptune_query(
                f"MATCH (e:Entity) WHERE e.entity_type = '{et}' "
                f"RETURN e.key AS key, e.display_name AS name, e.entity_type AS type"
            )
        else:
            results = self._neptune_query(
                "MATCH (e:Entity) "
                "RETURN e.key AS key, e.display_name AS name, e.entity_type AS type"
            )

        entities = []
        for r in results:
            entities.append({
                "name": r.get("name", r.get("key", "")),
                "type": r.get("type", "general"),
                "key": r.get("key", ""),
                "attributes": {},
            })
        return entities

    def get_edges(self, entity: str) -> List[Dict[str, Any]]:
        if not self._neptune_auth:
            return []

        key = self._escape(entity.lower())

        # Check entity exists
        results = self._neptune_query(
            f"MATCH (e:Entity {{key: '{key}'}}) RETURN e"
        )
        if not results:
            return []

        edge_results = self._neptune_query(
            f"MATCH (a:Entity)-[r:RELATION]-(b:Entity) "
            f"WHERE a.key = '{key}' OR b.key = '{key}' "
            f"RETURN DISTINCT a.display_name AS subject, r.relation_type AS predicate, "
            f"b.display_name AS object, r.event_date AS event_date"
        )

        seen: set = set()
        edges = []
        for r in edge_results:
            triple = (r.get("subject", ""), r.get("predicate", ""), r.get("object", ""))
            if triple not in seen:
                seen.add(triple)
                edges.append({
                    "subject": r.get("subject", ""),
                    "predicate": r.get("predicate", "RELATION"),
                    "object": r.get("object", ""),
                    "event_date": r.get("event_date", ""),
                })
        return edges

    def graph_stats(self) -> Dict[str, Any]:
        if not self._neptune_auth:
            return {"nodes": 0, "edges": 0, "types": {}}

        node_results = self._neptune_query(
            "MATCH (e:Entity) RETURN count(e) AS cnt"
        )
        nodes = node_results[0]["cnt"] if node_results else 0

        edge_results = self._neptune_query(
            "MATCH ()-[r:RELATION]->() RETURN count(r) AS cnt"
        )
        edge_count = edge_results[0]["cnt"] if edge_results else 0

        type_results = self._neptune_query(
            "MATCH (e:Entity) RETURN e.entity_type AS type, count(e) AS cnt"
        )
        types: Dict[str, int] = {}
        for r in type_results:
            et = r.get("type", "general") or "general"
            types[et] = r.get("cnt", 0)

        return {"nodes": nodes, "edges": edge_count, "types": types}

    def save(self) -> None:
        pass  # All AWS services persist automatically

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

    # ══════════════════════════════════════════════════════════════════
    # Helpers
    # ══════════════════════════════════════════════════════════════════

    @staticmethod
    def _escape(value: str) -> str:
        """Escape a string for safe inclusion in a Cypher literal."""
        return value.replace("\\", "\\\\").replace("'", "\\'")
