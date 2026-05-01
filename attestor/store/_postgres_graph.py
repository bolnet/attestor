"""Postgres graph-role mixin (split from postgres_backend.py).

This module is private — consumers should import ``PostgresBackend`` from
``attestor.store.postgres_backend``. The mixin is stateless: it operates on
``self._conn`` and ``self._has_age`` configured by ``PostgresBackend.__init__``.

Implements the optional Apache AGE Cypher path. When AGE is not installed
(e.g. Neon, Cloud SQL), every graph method short-circuits via
``_require_age`` to a clean ``NotImplementedError`` so the document path
keeps working.

Cypher security: ALL user-controlled property *values* are bound via AGE's
agtype parameter slot (third ``cypher()`` argument). Identifiers (labels,
property keys, relation types, variable-length path bounds) cannot be
parameterized in Cypher itself, so they are validated against
``_CYPHER_IDENT_RE`` / ``_validate_depth`` before being inlined.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any


logger = logging.getLogger("attestor")


def _escape_cypher(value: str) -> str:
    """Escape a string for safe inclusion in a Cypher literal.

    NOTE: This helper is retained for legacy callers and identifier-only
    contexts. ALL user-controlled property *values* must be passed via the
    AGE agtype-parameter slot (see ``PostgresBackend._age_query``), not
    interpolated into the query string. Quoted-string interpolation is
    fundamentally unsafe — Cypher syntax has additional metacharacters
    (``{``, ``}``, line breaks, etc.) that this function does not escape.
    """
    return value.replace("\\", "\\\\").replace("'", "\\'")


# Cypher identifier allow-list — used for property keys, label fragments, and
# anywhere we must inline an identifier (Cypher cannot parameterize identifiers
# or variable-length path bounds). Pattern matches standard Cypher identifiers.
_CYPHER_IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,255}$")


def _validate_cypher_identifier(value: str, kind: str = "identifier") -> str:
    """Validate that ``value`` is a safe Cypher identifier (label / property
    key / relation type). Identifiers cannot be parameterized in Cypher, so
    they must be statically validated before being inlined into a query.

    Raises ``ValueError`` on mismatch — the caller surfaces this as a normal
    validation error rather than letting an injection slip through.
    """
    if not isinstance(value, str) or not _CYPHER_IDENT_RE.match(value):
        raise ValueError(
            f"Invalid Cypher {kind}: {value!r}. Must match "
            f"[A-Za-z_][A-Za-z0-9_]{{0,255}}."
        )
    return value


def _validate_depth(value: int, max_depth: int = 10) -> int:
    """Validate a Cypher variable-length path bound. AGE does not support
    parameterizing the ``[*1..N]`` bounds, so the value is inlined; we
    must verify it is a small non-negative int."""
    try:
        d = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid depth (must be int): {value!r}")
    if d < 0 or d > max_depth:
        raise ValueError(f"Invalid depth (must be in 0..{max_depth}): {d}")
    return d


def _parse_agtype(raw: str) -> Any:
    """Parse an AGE agtype string into a Python object.

    AGE returns values like:
        {"name": "Alice", "entity_type": "person"}::vertex
        {"id": 123, ...}::edge
        [...]::path

    Strip the ::suffix and parse the JSON.
    """
    if raw is None:
        return None
    s = str(raw)
    # Strip ::vertex, ::edge, ::path, ::numeric etc.
    s = re.sub(r"::(?:vertex|edge|path|numeric)\s*$", "", s)
    # Also handle nested ::vertex inside arrays/paths
    s = re.sub(r"::(?:vertex|edge|path|numeric)", "", s)
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return s


class _PostgresGraphMixin:
    """Apache AGE graph role for PostgresBackend (Cypher over PG)."""

    # ── AGE Helpers ──

    def _require_age(self) -> None:
        """Raise if AGE is not available."""
        if not self._has_age:
            raise NotImplementedError(
                "Graph operations require the Apache AGE extension on this "
                "PostgreSQL instance. Configure a Neo4j backend for the graph "
                "role instead."
            )

    def _age_execute(self, sql: str, params: Any = None) -> None:
        """Execute a non-query AGE/SQL statement."""
        with self._conn.cursor() as cur:
            cur.execute(sql, params)

    def _age_query(
        self,
        cypher: str,
        cols: str = "result agtype",
        params: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Execute a Cypher query via AGE and return parsed results.

        Args:
            cypher: openCypher query string. Reference parameters as
                ``$name`` (e.g. ``MATCH (e:Entity {key: $key})``).
            cols: Column definition for the AS clause
                (default: ``"result agtype"``).
            params: Optional dict of parameters bound via AGE's third
                ``cypher()`` argument (an agtype JSON map). Values must be
                JSON-serializable. Use this for ALL user-controlled property
                values; never interpolate them into ``cypher`` directly.

        Security: AGE's parameter slot binds property *values* safely.
        Identifiers (labels, property keys, edge types, variable-length
        path bounds) cannot be parameterized in Cypher itself — those must
        be validated against ``_CYPHER_IDENT_RE`` (or similar) before being
        inlined.
        """
        with self._conn.cursor() as cur:
            cur.execute("LOAD 'age';")
            cur.execute("SET search_path = ag_catalog, \"$user\", public;")
            if params:
                # AGE's third cypher() arg is an agtype map; pass it as a
                # JSON string and cast in SQL. psycopg2 binds the JSON via
                # the standard %s mechanism, so the Cypher body itself
                # never sees user data.
                sql = (
                    f"SELECT * FROM cypher('memory_graph', $$ {cypher} $$, %s) "
                    f"AS ({cols});"
                )
                cur.execute(sql, (json.dumps(params),))
            else:
                sql = (
                    f"SELECT * FROM cypher('memory_graph', $$ {cypher} $$) "
                    f"AS ({cols});"
                )
                cur.execute(sql)
            rows = cur.fetchall()
        return [_parse_agtype(row[0]) if len(row) == 1 else tuple(_parse_agtype(c) for c in row) for row in rows]

    # ── GraphStore (Apache AGE) ──

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._require_age()
        key = name.lower()

        # Check if entity exists. Property *value* (key) is bound via
        # AGE's agtype parameter slot — never interpolated.
        results = self._age_query(
            "MATCH (e:Entity {key: $key}) RETURN e",
            params={"key": key},
        )

        if results:
            existing = results[0]
            existing_type = existing.get("properties", {}).get("entity_type", "general")
            new_type = entity_type if (existing_type == "general" and entity_type != "general") else existing_type

            # Property *keys* in the SET clause are identifiers and cannot
            # be parameterized in Cypher; validate them against the
            # identifier allow-list before inlining. Property *values*
            # are bound via the agtype params slot.
            set_parts = ["e.display_name = $display_name", "e.entity_type = $entity_type"]
            cy_params: dict[str, Any] = {
                "key": key,
                "display_name": name,
                "entity_type": new_type,
            }
            if attributes:
                for k, v in attributes.items():
                    safe_k = _validate_cypher_identifier(k, kind="property key")
                    pname = f"attr_{safe_k}"
                    set_parts.append(f"e.{safe_k} = ${pname}")
                    cy_params[pname] = v if isinstance(v, (int, float, bool)) else str(v)

            set_clause = ", ".join(set_parts)
            # ``set_clause`` is composed from validated identifiers and
            # ``$param`` references only — no user-controlled string content.
            self._age_query(
                "MATCH (e:Entity {key: $key}) SET " + set_clause + " RETURN e",
                params=cy_params,
            )
        else:
            prop_parts = [
                "key: $key",
                "display_name: $display_name",
                "entity_type: $entity_type",
            ]
            cy_params = {
                "key": key,
                "display_name": name,
                "entity_type": entity_type,
            }
            if attributes:
                for k, v in attributes.items():
                    safe_k = _validate_cypher_identifier(k, kind="property key")
                    pname = f"attr_{safe_k}"
                    prop_parts.append(f"{safe_k}: ${pname}")
                    cy_params[pname] = v if isinstance(v, (int, float, bool)) else str(v)
            props = ", ".join(prop_parts)
            # ``props`` is composed from validated identifiers and ``$param``
            # references only — no user-controlled string content.
            self._age_query(
                "CREATE (e:Entity {" + props + "}) RETURN e",
                params=cy_params,
            )

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._require_age()
        from_key = from_entity.lower()
        to_key = to_entity.lower()

        # ``relation_type`` is stored as a property value (not used as a
        # Cypher identifier here — the edge label is the static "RELATION"),
        # so we still pass it through the param slot. We retain the
        # uppercase normalization for backward compatibility with existing
        # rows / callers.
        sanitized = re.sub(r"[^A-Za-z0-9_]", "_", relation_type).upper()

        # Ensure both entities exist (AGE doesn't support MERGE ... ON CREATE SET)
        for ek, en in [(from_key, from_entity), (to_key, to_entity)]:
            existing = self._age_query(
                "MATCH (e:Entity {key: $key}) RETURN e",
                params={"key": ek},
            )
            if not existing:
                self._age_query(
                    "CREATE (e:Entity {key: $key, display_name: $display_name, "
                    "entity_type: 'general'}) RETURN e",
                    params={"key": ek, "display_name": en},
                )

        # Build the edge property map. Identifier keys are validated;
        # values are bound via the agtype param slot.
        edge_prop_parts = ["relation_type: $relation_type"]
        cy_params: dict[str, Any] = {
            "from_key": from_key,
            "to_key": to_key,
            "relation_type": sanitized,
        }
        if metadata:
            for k, v in metadata.items():
                safe_k = _validate_cypher_identifier(k, kind="metadata key")
                pname = f"meta_{safe_k}"
                edge_prop_parts.append(f"{safe_k}: ${pname}")
                cy_params[pname] = v if isinstance(v, (int, float, bool)) else str(v)
        edge_props = ", ".join(edge_prop_parts)

        # ``edge_props`` is composed from validated identifiers and ``$param``
        # references only — no user-controlled string content.
        self._age_query(
            "MATCH (a:Entity {key: $from_key}), (b:Entity {key: $to_key}) "
            "CREATE (a)-[r:RELATION {" + edge_props + "}]->(b) RETURN r",
            params=cy_params,
        )

    def get_related(self, entity: str, depth: int = 2) -> list[str]:
        self._require_age()
        key = entity.lower()
        # AGE cannot parameterize variable-length path bounds (``[*1..N]``);
        # validate as a small int to keep the inlined value safe.
        safe_depth = _validate_depth(depth)

        results = self._age_query(
            "MATCH (e:Entity {key: $key}) RETURN e",
            params={"key": key},
        )
        if not results:
            return []

        # ``safe_depth`` is a validated int (see ``_validate_depth``); AGE
        # cannot parameterize variable-length path bounds, so it is
        # concatenated as a plain int — no string interpolation of
        # user-controlled data is involved.
        path_bound = "[*1.." + str(safe_depth) + "]"
        results = self._age_query(
            "MATCH (start:Entity {key: $key})"
            "-" + path_bound + "-(connected:Entity) "
            "RETURN DISTINCT connected",
            params={"key": key},
        )
        names = []
        for r in results:
            if isinstance(r, dict):
                name = r.get("properties", {}).get("display_name")
                if name:
                    names.append(name)
        return names

    def get_subgraph(self, entity: str, depth: int = 2) -> dict[str, Any]:
        self._require_age()
        key = entity.lower()
        safe_depth = _validate_depth(depth)

        # Check entity exists
        results = self._age_query(
            "MATCH (e:Entity {key: $key}) RETURN e",
            params={"key": key},
        )
        if not results:
            return {"entity": entity, "nodes": [], "edges": []}

        # Get nodes — ``safe_depth`` is validated int; concatenation is safe.
        path_bound = "[*0.." + str(safe_depth) + "]"
        node_results = self._age_query(
            "MATCH (start:Entity {key: $key})"
            "-" + path_bound + "-(n:Entity) "
            "RETURN DISTINCT n",
            params={"key": key},
        )

        nodes = []
        node_keys = set()
        for r in node_results:
            if isinstance(r, dict):
                props = r.get("properties", {})
                nk = props.get("key")
                if nk and nk not in node_keys:
                    node_keys.add(nk)
                    nodes.append({
                        "name": props.get("display_name", nk),
                        "type": props.get("entity_type", "general"),
                        "key": nk,
                    })

        # Get edges between those nodes
        edges = []
        if len(node_keys) > 1:
            edge_path_bound = "[*0.." + str(safe_depth) + "]"
            edge_results = self._age_query(
                "MATCH (start:Entity {key: $key})"
                "-" + edge_path_bound + "-(a:Entity)-[r:RELATION]->(b:Entity) "
                "RETURN DISTINCT a, r, b",
                cols="a agtype, r agtype, b agtype",
                params={"key": key},
            )
            for row in edge_results:
                if isinstance(row, tuple) and len(row) == 3:
                    a_props = row[0].get("properties", {}) if isinstance(row[0], dict) else {}
                    r_props = row[1].get("properties", {}) if isinstance(row[1], dict) else {}
                    b_props = row[2].get("properties", {}) if isinstance(row[2], dict) else {}
                    a_key = a_props.get("key")
                    b_key = b_props.get("key")
                    if a_key in node_keys and b_key in node_keys:
                        edges.append({
                            "source": a_key,
                            "target": b_key,
                            "type": r_props.get("relation_type", "RELATION"),
                        })

        return {"entity": entity, "nodes": nodes, "edges": edges}

    def get_entities(self, entity_type: str | None = None) -> list[dict[str, Any]]:
        self._require_age()
        if entity_type:
            results = self._age_query(
                "MATCH (e:Entity) WHERE e.entity_type = $entity_type RETURN e",
                params={"entity_type": entity_type},
            )
        else:
            results = self._age_query("MATCH (e:Entity) RETURN e")

        entities = []
        for r in results:
            if isinstance(r, dict):
                props = r.get("properties", {})
                key = props.get("key", "")
                attrs = {
                    k: v for k, v in props.items()
                    if k not in ("key", "display_name", "entity_type")
                }
                entities.append({
                    "name": props.get("display_name", key),
                    "type": props.get("entity_type", "general"),
                    "key": key,
                    "attributes": attrs,
                })
        return entities

    def get_edges(self, entity: str) -> list[dict[str, Any]]:
        self._require_age()
        key = entity.lower()

        results = self._age_query(
            "MATCH (e:Entity {key: $key}) RETURN e",
            params={"key": key},
        )
        if not results:
            return []

        edge_results = self._age_query(
            "MATCH (a:Entity)-[r:RELATION]-(b:Entity) "
            "WHERE a.key = $key OR b.key = $key "
            "RETURN a, r, b",
            cols="a agtype, r agtype, b agtype",
            params={"key": key},
        )

        seen = set()
        edges = []
        for row in edge_results:
            if isinstance(row, tuple) and len(row) == 3:
                a_props = row[0].get("properties", {}) if isinstance(row[0], dict) else {}
                r_props = row[1].get("properties", {}) if isinstance(row[1], dict) else {}
                b_props = row[2].get("properties", {}) if isinstance(row[2], dict) else {}

                subject = a_props.get("display_name", "")
                predicate = r_props.get("relation_type", "RELATION")
                obj = b_props.get("display_name", "")
                event_date = r_props.get("event_date", "")

                triple = (subject, predicate, obj)
                if triple not in seen:
                    seen.add(triple)
                    edges.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": obj,
                        "event_date": event_date,
                    })
        return edges

    def graph_stats(self) -> dict[str, Any]:
        """Graph statistics: node/edge counts, entity types."""
        self._require_age()
        node_results = self._age_query(
            "MATCH (e:Entity) RETURN count(e)",
        )
        nodes = node_results[0] if node_results else 0

        edge_results = self._age_query(
            "MATCH ()-[r:RELATION]->() RETURN count(r)",
        )
        edge_count = edge_results[0] if edge_results else 0

        type_results = self._age_query(
            "MATCH (e:Entity) RETURN e.entity_type, count(e)",
            cols="et agtype, cnt agtype",
        )
        types: dict[str, int] = {}
        for row in type_results:
            if isinstance(row, tuple) and len(row) == 2:
                et = str(row[0]).strip('"') if row[0] else "general"
                types[et] = int(row[1]) if row[1] else 0

        return {"nodes": nodes, "edges": edge_count, "types": types}
