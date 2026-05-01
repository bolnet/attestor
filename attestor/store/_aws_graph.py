"""Neptune graph-role mixin for AWSBackend (split from aws_backend.py).

This module is private — consumers should import ``AWSBackend`` from
``attestor.store.aws_backend``. The mixin is stateless: it operates on
``self._neptune_endpoint`` / ``self._neptune_auth`` / ``self._session``
configured by ``AWSBackend.__init__``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from attestor.store._graph_utils import sanitize_rel_type as _sanitize_rel_type

logger = logging.getLogger("attestor")


class _AWSGraphMixin:
    """Neptune (openCypher over HTTP) graph role for AWSBackend."""

    # ── Neptune query helper ──

    def _neptune_query(self, query: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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

    # ── GraphStore — Neptune openCypher ──

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: dict[str, Any] | None = None,
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
        metadata: dict[str, Any] | None = None,
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

    def get_related(self, entity: str, depth: int = 2) -> list[str]:
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

    def get_subgraph(self, entity: str, depth: int = 2) -> dict[str, Any]:
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

    def get_entities(self, entity_type: str | None = None) -> list[dict[str, Any]]:
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

    def get_edges(self, entity: str) -> list[dict[str, Any]]:
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

    def graph_stats(self) -> dict[str, Any]:
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
        types: dict[str, int] = {}
        for r in type_results:
            et = r.get("type", "general") or "general"
            types[et] = r.get("cnt", 0)

        return {"nodes": nodes, "edges": edge_count, "types": types}

    def save(self) -> None:
        pass  # All AWS services persist automatically

    # ── Helpers ──

    @staticmethod
    def _escape(value: str) -> str:
        """Escape a string for safe inclusion in a Cypher literal."""
        return value.replace("\\", "\\\\").replace("'", "\\'")
