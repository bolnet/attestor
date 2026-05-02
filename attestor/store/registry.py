"""Backend registry — maps backend names to implementations and resolves role assignments."""

from __future__ import annotations

import importlib
import logging
from typing import Any

from attestor.store.connection import (
    BACKEND_DEFAULTS,
    CLOUD_DEFAULTS,
    merge_config_layers,
)

logger = logging.getLogger("attestor")


class BackendConflictError(Exception):
    """Raised when two backends claim the same role."""


BACKEND_REGISTRY: dict[str, dict[str, Any]] = {
    "postgres": {
        "module": "attestor.store.postgres_backend",
        "class": "PostgresBackend",
        # Document role only — vector belongs to Pinecone and graph to
        # Neo4j in the canonical (and only) supported stack.
        "roles": {"document"},
        "init_style": "config",
    },
    "pinecone": {
        "module": "attestor.store.pinecone_backend",
        "class": "PineconeBackend",
        # Vector role. Paired with `postgres` (doc) + `neo4j` (graph).
        "roles": {"vector"},
        "init_style": "config",
    },
    "neo4j": {
        "module": "attestor.store.neo4j_backend",
        "class": "Neo4jBackend",
        "roles": {"graph"},
        "init_style": "config",
    },
}

DEFAULT_BACKENDS = ["postgres", "pinecone", "neo4j"]


def resolve_backends(
    backends: list[str] | None = None,
) -> dict[str, str]:
    """Resolve which backend fills each role.

    Args:
        backends: Ordered list of backend names. Defaults to DEFAULT_BACKENDS.

    Returns:
        Dict mapping role -> backend_name (e.g., {"document": "postgres", "vector": "postgres", "graph": "neo4j"})

    Raises:
        BackendConflictError: If two backends claim the same role.
        ValueError: If a backend name is not in the registry.
    """
    if backends is None:
        backends = DEFAULT_BACKENDS

    role_assignments: dict[str, str] = {}

    for backend_name in backends:
        if backend_name not in BACKEND_REGISTRY:
            raise ValueError(
                f"Unknown backend: {backend_name!r}. "
                f"Known: {sorted(BACKEND_REGISTRY.keys())}"
            )

        entry = BACKEND_REGISTRY[backend_name]
        roles = entry.get("roles") or set()
        if not roles:
            raise ValueError(
                f"Backend {backend_name!r} has no roles declared in "
                f"BACKEND_REGISTRY. Every backend must claim at least "
                f"one of {{'document', 'vector', 'graph'}}."
            )
        for role in roles:
            if role in role_assignments:
                raise BackendConflictError(
                    f"Role {role!r} claimed by both "
                    f"{role_assignments[role]!r} and {backend_name!r}"
                )
            role_assignments[role] = backend_name

    return role_assignments


def instantiate_backend(
    backend_name: str,
    store_path: Any,
    backend_config: dict[str, Any] | None = None,
) -> Any:
    """Import and instantiate a backend class.

    Applies layered config resolution:
        Layer 1: CLOUD_DEFAULTS
        Layer 2: BACKEND_DEFAULTS[backend_name]
        Layer 3+: backend_config (project config + CLI overrides)

    Args:
        backend_name: Registry key — one of "postgres", "pinecone", "neo4j".
        store_path: Path Attestor uses for side-car files (certs, etc.).
        backend_config: Config dict for backend (layers 3-5 merged).

    Returns:
        Instantiated backend object.
    """
    # Intentionally lets KeyError surface for unknown backend names —
    # callers (CLI, registry tests) rely on this concrete type.
    entry = BACKEND_REGISTRY[backend_name]
    try:
        module = importlib.import_module(entry["module"])
    except ImportError as e:
        raise ImportError(
            f"Backend module {entry['module']!r} not installed; "
            f"install with `pip install attestor[{backend_name}]`."
        ) from e
    try:
        cls = getattr(module, entry["class"])
    except AttributeError as e:
        raise AttributeError(
            f"Backend class {entry['class']!r} not found in "
            f"module {entry['module']!r}; registry entry is stale."
        ) from e

    merged = merge_config_layers(
        CLOUD_DEFAULTS,
        BACKEND_DEFAULTS.get(backend_name, {}),
        backend_config or {},
    )
    merged["_store_path"] = str(store_path)
    return cls(merged)
