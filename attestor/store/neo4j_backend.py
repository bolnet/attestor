"""Neo4j backend — graph role only (Layer 0 stack: Postgres+pgvector + Neo4j).

Uses the official Neo4j Python driver (Bolt). Entity nodes are stored as
`(:Entity {key, namespace, display_name, entity_type, ...})`. Relationships
use the sanitized `relation_type` as the Neo4j relationship type. PageRank
is delegated to the Graph Data Science (GDS) plugin when available.

Namespace tenancy (multi-tenant isolation):
    Entity nodes and edges carry a ``namespace`` property; every read
    operation in the recall path filters on it. New writes always carry the
    property. Pre-namespace nodes (without the property) are treated as
    ``namespace="default"`` on read via ``coalesce(e.namespace, 'default')``;
    no migration is required. Closes a cross-tenant graph-affinity leak
    where a recall in one namespace could pull bonuses from another.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Set

from neo4j import GraphDatabase

from attestor.store.base import GraphStore
from attestor.store.connection import CloudConnection

logger = logging.getLogger("attestor")

_REL_TYPE_RE = re.compile(r"[^A-Za-z0-9_]")
_NAME_PUNCT_RE = re.compile(r"[^\w\s-]", re.UNICODE)
_WS_RE = re.compile(r"\s+")

# Namespace assigned to writes when the caller passes ``None`` AND used as
# the read fallback for pre-namespace nodes via ``coalesce(...)``.
_DEFAULT_NAMESPACE = "default"


def _ns(namespace: Optional[str]) -> str:
    """Return the effective namespace, defaulting to ``default``."""
    return namespace or _DEFAULT_NAMESPACE


def _sanitize_rel_type(rel_type: str) -> str:
    """Normalize relation type to a valid Neo4j relationship type."""
    return _REL_TYPE_RE.sub("_", rel_type).upper() or "RELATED_TO"


def _normalize_name(name: str) -> str:
    """Canonicalize an entity name for use as a MERGE key.

    Lowercases, strips punctuation (except ``-`` in hyphenated names),
    collapses internal whitespace, and trims. Prevents "Caroline",
    "caroline.", and "  Caroline  " from producing three distinct nodes.
    """
    if not name:
        return ""
    text = _NAME_PUNCT_RE.sub(" ", name.lower())
    return _WS_RE.sub(" ", text).strip()


class Neo4jBackend(GraphStore):
    """Graph storage backed by Neo4j 5 + GDS."""

    ROLES: Set[str] = {"graph"}

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        conn = CloudConnection.from_config(config, backend_name="neo4j")
        self._conn = conn

        self._driver = GraphDatabase.driver(
            conn.url,
            auth=(conn.auth.username, conn.auth.password),
        )
        self._database = conn.database or "neo4j"

        self._driver.verify_connectivity()
        self._init_schema()
        self._has_gds: Optional[bool] = None

    def _session(self):
        return self._driver.session(database=self._database)

    def _init_schema(self) -> None:
        with self._session() as s:
            # Drop the legacy single-property constraint if present — it
            # blocks the new composite (key, namespace) constraint.
            # ``IF EXISTS`` makes this idempotent on fresh installs.
            try:
                s.run("DROP CONSTRAINT entity_key_unique IF EXISTS")
            except Exception as e:
                logger.debug("legacy constraint drop skipped: %s", e)
            # Composite uniqueness lets the same display-name live in
            # different namespaces without colliding.
            s.run(
                "CREATE CONSTRAINT entity_key_namespace_unique IF NOT EXISTS "
                "FOR (e:Entity) REQUIRE (e.key, e.namespace) IS UNIQUE"
            )

    # ── GraphStore interface ──

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Upsert an entity scoped by namespace.

        New writes always carry ``namespace`` as a node property. Callers
        that pass ``None`` get the ``"default"`` tenant — same bucket the
        read path uses for pre-namespace nodes via ``coalesce``.
        """
        key = _normalize_name(name)
        ns = _ns(namespace)
        attrs = dict(attributes) if attributes else {}
        with self._session() as s:
            s.run(
                """
                MERGE (e:Entity {key: $key, namespace: $namespace})
                ON CREATE SET e.display_name = $name, e.entity_type = $etype
                ON MATCH SET e.display_name = $name,
                             e.entity_type = CASE
                                 WHEN e.entity_type IS NULL OR e.entity_type = 'general'
                                 THEN $etype ELSE e.entity_type END
                SET e += $attrs
                """,
                key=key, namespace=ns, name=name,
                etype=entity_type, attrs=attrs,
            )

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
        namespace: Optional[str] = None,
    ) -> None:
        """Upsert an edge between two namespace-scoped entities.

        Both endpoints AND the edge itself carry ``namespace``. We do not
        support cross-namespace edges by design: if ``from_entity`` and
        ``to_entity`` legitimately belong to different tenants, callers
        must merge them at the document layer first.
        """
        from_key = _normalize_name(from_entity)
        to_key = _normalize_name(to_entity)
        rel = _sanitize_rel_type(relation_type)
        meta = dict(metadata) if metadata else {}
        # Edge namespace == its endpoints' namespace; record explicitly so
        # the read filter doesn't need to traverse to a node to gate.
        ns = _ns(namespace)
        meta["namespace"] = ns
        event_date = meta.get("event_date", "")
        with self._session() as s:
            s.run(
                f"""
                MERGE (a:Entity {{key: $from_key, namespace: $namespace}})
                  ON CREATE SET a.display_name = $from_name, a.entity_type = 'general'
                MERGE (b:Entity {{key: $to_key, namespace: $namespace}})
                  ON CREATE SET b.display_name = $to_name, b.entity_type = 'general'
                MERGE (a)-[r:`{rel}` {{event_date: $event_date, namespace: $namespace}}]->(b)
                SET r += $meta, r.relation_type = $rel
                """,
                from_key=from_key, from_name=from_entity,
                to_key=to_key, to_name=to_entity,
                namespace=ns,
                meta=meta, rel=rel, event_date=event_date,
            )

    def get_related(
        self,
        entity: str,
        depth: int = 2,
        namespace: Optional[str] = None,
    ) -> List[str]:
        """BFS traversal restricted to a single namespace.

        ``coalesce(n.namespace, 'default')`` keeps pre-namespace nodes
        recallable under ``namespace="default"`` without a backfill step.
        """
        start = _normalize_name(entity)
        d = max(1, int(depth))
        ns = _ns(namespace)
        with self._session() as s:
            result = s.run(
                f"""
                MATCH (start:Entity {{key: $start}})
                WHERE coalesce(start.namespace, 'default') = $namespace
                MATCH (start)-[*1..{d}]-(other:Entity)
                WHERE coalesce(other.namespace, 'default') = $namespace
                RETURN DISTINCT other.display_name AS name
                """,
                start=start, namespace=ns,
            )
            return [row["name"] for row in result if row["name"]]

    def get_subgraph(
        self,
        entity: str,
        depth: int = 2,
        namespace: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Fetch a namespace-scoped subgraph rooted at ``entity``."""
        start = _normalize_name(entity)
        d = max(1, int(depth))
        ns = _ns(namespace)
        with self._session() as s:
            nodes_rows = list(s.run(
                f"""
                MATCH (start:Entity {{key: $start}})
                WHERE coalesce(start.namespace, 'default') = $namespace
                OPTIONAL MATCH (start)-[*0..{d}]-(n:Entity)
                WHERE n IS NULL
                   OR coalesce(n.namespace, 'default') = $namespace
                RETURN DISTINCT
                    coalesce(n.key, start.key) AS key,
                    coalesce(n.display_name, start.display_name) AS name,
                    coalesce(n.entity_type, start.entity_type, 'general') AS etype
                """,
                start=start, namespace=ns,
            ))
            if not nodes_rows or all(r["key"] is None for r in nodes_rows):
                return {"entity": entity, "nodes": [], "edges": []}
            nodes = [
                {"key": r["key"], "name": r["name"] or r["key"],
                 "type": r["etype"] or "general"}
                for r in nodes_rows if r["key"]
            ]
            edges_rows = list(s.run(
                f"""
                MATCH (start:Entity {{key: $start}})
                WHERE coalesce(start.namespace, 'default') = $namespace
                MATCH (start)-[*0..{d}]-(n:Entity)
                WHERE coalesce(n.namespace, 'default') = $namespace
                WITH collect(DISTINCT n.key) AS keys
                MATCH (a:Entity)-[r]->(b:Entity)
                WHERE a.key IN keys AND b.key IN keys
                  AND coalesce(a.namespace, 'default') = $namespace
                  AND coalesce(b.namespace, 'default') = $namespace
                RETURN a.key AS source, b.key AS target, type(r) AS type
                """,
                start=start, namespace=ns,
            ))
            edges = [
                {"source": r["source"], "target": r["target"], "type": r["type"]}
                for r in edges_rows
            ]
        return {"entity": entity, "nodes": nodes, "edges": edges}

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        # Admin/UI surface — not on the recall hot path. Returns entities
        # across all namespaces by design (operator visibility). The recall
        # path uses get_related/get_subgraph/get_edges which DO filter.
        if entity_type:
            cypher = "MATCH (n:Entity {entity_type: $etype}) RETURN n"
            params: Dict[str, Any] = {"etype": entity_type}
        else:
            cypher = "MATCH (n:Entity) RETURN n"
            params = {}
        result: List[Dict[str, Any]] = []
        with self._session() as s:
            for row in s.run(cypher, **params):
                node = row["n"]
                props = dict(node)
                key = props.pop("key", None)
                name = props.pop("display_name", key)
                etype = props.pop("entity_type", "general")
                result.append({
                    "key": key,
                    "name": name,
                    "type": etype,
                    "attributes": props,
                })
        return result

    def get_edges(
        self, entity: str, namespace: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return all edges incident to ``entity`` within the namespace."""
        key = _normalize_name(entity)
        ns = _ns(namespace)
        cypher = """
            MATCH (a:Entity {key: $key})-[r]-(b:Entity)
            WHERE coalesce(a.namespace, 'default') = $namespace
              AND coalesce(b.namespace, 'default') = $namespace
            RETURN a.display_name AS subject,
                   type(r) AS predicate,
                   b.display_name AS object,
                   coalesce(r.event_date, '') AS event_date,
                   coalesce(r.source_quote, '') AS source_quote
        """
        seen: set = set()
        result: List[Dict[str, Any]] = []
        with self._session() as s:
            for row in s.run(cypher, key=key, namespace=ns):
                triple = (row["subject"], row["predicate"], row["object"])
                if triple in seen:
                    continue
                seen.add(triple)
                result.append({
                    "subject": row["subject"],
                    "predicate": row["predicate"],
                    "object": row["object"],
                    "event_date": row["event_date"],
                    "source_quote": row["source_quote"],
                })
        return result

    def graph_stats(self) -> Dict[str, Any]:
        with self._session() as s:
            nodes = s.run("MATCH (n:Entity) RETURN count(n) AS c").single()["c"]
            edges = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
            type_rows = s.run(
                "MATCH (n:Entity) "
                "RETURN coalesce(n.entity_type, 'general') AS t, count(*) AS c"
            )
            types = {row["t"]: row["c"] for row in type_rows}
        return {"nodes": nodes, "edges": edges, "types": types}

    # ── Optional: GDS PageRank ──

    def _gds_available(self) -> bool:
        if self._has_gds is not None:
            return self._has_gds
        try:
            with self._session() as s:
                s.run("RETURN gds.version() AS v").single()
            self._has_gds = True
        except Exception:
            self._has_gds = False
        return self._has_gds

    def pagerank(self, alpha: float = 0.85) -> Dict[str, float]:
        """PageRank via GDS. Returns {entity_key: score}; empty if GDS or graph unavailable."""
        if not self._gds_available():
            return {}
        try:
            with self._session() as s:
                rows = s.run(
                    """
                    CALL gds.pageRank.stream({
                        nodeProjection: 'Entity',
                        relationshipProjection: '*',
                        dampingFactor: $alpha
                    })
                    YIELD nodeId, score
                    RETURN gds.util.asNode(nodeId).key AS key, score
                    """,
                    alpha=float(alpha),
                )
                return {row["key"]: float(row["score"]) for row in rows if row["key"]}
        except Exception as e:
            logger.warning("Neo4j GDS PageRank failed: %s", e)
            return {}

    # ── Lifecycle ──

    def save(self) -> None:
        # Neo4j persists automatically.
        pass

    def close(self) -> None:
        try:
            self._driver.close()
        except Exception:
            pass
