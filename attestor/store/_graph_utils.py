"""Shared graph-layer string utilities for backend implementations."""

from __future__ import annotations

import re

_REL_TYPE_RE = re.compile(r"[^A-Za-z0-9_]")


def sanitize_rel_type(rel_type: str) -> str:
    """Normalize a graph-edge relationship type for Cypher / openCypher / Gremlin.

    Replaces non-word characters with ``_``, uppercases the result, and falls
    back to ``"RELATED_TO"`` when the input is empty or sanitizes to empty
    (e.g. ``""``, ``"!!!"``). The fallback was previously only applied by the
    Neo4j backend; AWS/Azure/Arango used the same regex without a fallback and
    would return ``""``, which is invalid Cypher and crashes at write time.
    Unifying on the fallback makes this safe across every backend.
    """
    return _REL_TYPE_RE.sub("_", rel_type).upper() or "RELATED_TO"
