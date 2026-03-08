"""Neo4j-based entity relationship graph for memory enrichment.

Requires: pip install neo4j
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from neo4j import GraphDatabase


# Only allow safe chars in relationship types (injected via f-string)
_SAFE_REL_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _sanitize_rel_type(rel_type: str) -> str:
    """Ensure relationship type is safe for Cypher interpolation."""
    normalized = rel_type.upper().replace(" ", "_")
    if not _SAFE_REL_RE.match(normalized):
        normalized = re.sub(r"[^A-Za-z0-9_]", "", normalized)
        if not normalized:
            normalized = "RELATED_TO"
    return normalized


class Neo4jGraph:
    """Entity graph backed by Neo4j."""

    def __init__(
        self,
        uri: str = "bolt://localhost:7687",
        auth: tuple = ("neo4j", "neo4j"),
        database: str = "neo4j",
    ):
        self.uri = uri
        self.auth = auth
        self.database = database
        self._driver = GraphDatabase.driver(uri, auth=auth)
        self._driver.verify_connectivity()
        self._ensure_indexes()

    def _ensure_indexes(self) -> None:
        """Create indexes for fast lookups."""
        self._driver.execute_query(
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            database_=self.database,
        )

    def close(self) -> None:
        self._driver.close()

    def save(self, path: Optional[str] = None) -> None:
        """No-op — Neo4j persists automatically."""
        pass

    def load(self, path: Optional[str] = None) -> None:
        """No-op — Neo4j loads from its own storage."""
        pass

    # ── Entity CRUD ──

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add or update an entity node."""
        attrs = attributes or {}
        self._driver.execute_query(
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.entity_type = $entity_type, e += $attrs "
            "ON MATCH SET e += $attrs",
            name=name.lower(),
            entity_type=entity_type,
            attrs={**attrs, "display_name": name},
            database_=self.database,
        )

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a directed relationship between entities."""
        safe_type = _sanitize_rel_type(relation_type)
        meta = metadata or {}

        # Auto-create nodes if they don't exist
        self._driver.execute_query(
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.entity_type = 'general', e.display_name = $display",
            name=from_entity.lower(),
            display=from_entity,
            database_=self.database,
        )
        self._driver.execute_query(
            "MERGE (e:Entity {name: $name}) "
            "ON CREATE SET e.entity_type = 'general', e.display_name = $display",
            name=to_entity.lower(),
            display=to_entity,
            database_=self.database,
        )

        query = (
            "MATCH (a:Entity {name: $from_name}) "
            "MATCH (b:Entity {name: $to_name}) "
            f"MERGE (a)-[r:{safe_type}]->(b) "
            "SET r += $metadata"
        )
        self._driver.execute_query(
            query,
            from_name=from_entity.lower(),
            to_name=to_entity.lower(),
            metadata=meta,
            database_=self.database,
        )

    # ── Queries ──

    def get_related(self, entity: str, depth: int = 2) -> List[str]:
        """Get entities related to the given entity via traversal.

        Traverses in both directions up to `depth` hops.
        Returns display names (original casing).
        """
        # Neo4j doesn't support parameterized depth in variable-length paths
        d = int(depth)
        records, _, _ = self._driver.execute_query(
            f"MATCH (start:Entity {{name: $name}})-[*1..{d}]-(other:Entity) "
            "RETURN DISTINCT other.display_name AS name",
            name=entity.lower(),
            database_=self.database,
        )
        return [r["name"] for r in records if r["name"]]

    def get_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        """Get entity and all connected nodes/edges as a dict."""
        d = int(depth)
        records, _, _ = self._driver.execute_query(
            f"MATCH path = (start:Entity {{name: $name}})-[*1..{d}]-(other:Entity) "
            "UNWIND nodes(path) AS n "
            "WITH COLLECT(DISTINCT {name: n.display_name, type: n.entity_type, key: n.name}) AS nodes, "
            "     COLLECT(path) AS paths "
            "UNWIND paths AS p "
            "UNWIND relationships(p) AS r "
            "WITH nodes, "
            "     COLLECT(DISTINCT {"
            "       source: startNode(r).name, "
            "       target: endNode(r).name, "
            "       type: type(r)"
            "     }) AS edges "
            "RETURN nodes, edges",
            name=entity.lower(),
            database_=self.database,
        )
        if records:
            return {
                "entity": entity,
                "nodes": records[0]["nodes"],
                "edges": records[0]["edges"],
            }
        return {"entity": entity, "nodes": [], "edges": []}

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all entities, optionally filtered by type."""
        if entity_type:
            records, _, _ = self._driver.execute_query(
                "MATCH (e:Entity) WHERE e.entity_type = $entity_type "
                "RETURN e {.*, key: e.name} AS entity",
                entity_type=entity_type,
                database_=self.database,
            )
        else:
            records, _, _ = self._driver.execute_query(
                "MATCH (e:Entity) RETURN e {.*, key: e.name} AS entity",
                database_=self.database,
            )
        return [dict(r["entity"]) for r in records]

    def stats(self) -> Dict[str, Any]:
        """Graph statistics."""
        def _stats_tx(tx):
            total_nodes = tx.run(
                "MATCH (e:Entity) RETURN count(e) AS c"
            ).single()["c"]

            total_edges = tx.run(
                "MATCH ()-[r]->() RETURN count(r) AS c"
            ).single()["c"]

            type_result = tx.run(
                "MATCH (e:Entity) "
                "RETURN e.entity_type AS t, count(e) AS c"
            )
            types = {r["t"]: r["c"] for r in type_result}

            return {
                "nodes": total_nodes,
                "edges": total_edges,
                "types": types,
            }

        with self._driver.session(database=self.database) as session:
            return session.execute_read(_stats_tx)
