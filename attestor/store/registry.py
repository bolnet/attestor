"""Backend registry — maps backend names to implementations and resolves role assignments."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional

from attestor.store.connection import (
    BACKEND_DEFAULTS,
    CLOUD_DEFAULTS,
    merge_config_layers,
)

logger = logging.getLogger("attestor")


class BackendConflictError(Exception):
    """Raised when two backends claim the same role."""


BACKEND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "arangodb": {
        "module": "attestor.store.arango_backend",
        "class": "ArangoBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
    "postgres": {
        "module": "attestor.store.postgres_backend",
        "class": "PostgresBackend",
        # AGE removed from default Postgres image; graph role belongs to Neo4j.
        "roles": {"document", "vector"},
        "init_style": "config",
    },
    "neo4j": {
        "module": "attestor.store.neo4j_backend",
        "class": "Neo4jBackend",
        "roles": {"graph"},
        "init_style": "config",
    },
    "gcp": {
        "module": "attestor.store.gcp_backend",
        "class": "GCPBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
    "azure": {
        "module": "attestor.store.azure_backend",
        "class": "AzureBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
    "aws": {
        "module": "attestor.store.aws_backend",
        "class": "AWSBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
}

DEFAULT_BACKENDS = ["postgres", "neo4j"]


def resolve_backends(
    backends: Optional[List[str]] = None,
) -> Dict[str, str]:
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

    role_assignments: Dict[str, str] = {}

    for backend_name in backends:
        if backend_name not in BACKEND_REGISTRY:
            raise ValueError(f"Unknown backend: {backend_name!r}. Known: {sorted(BACKEND_REGISTRY.keys())}")

        entry = BACKEND_REGISTRY[backend_name]
        for role in entry["roles"]:
            if role in role_assignments:
                raise BackendConflictError(
                    f"Role {role!r} claimed by both {role_assignments[role]!r} and {backend_name!r}"
                )
            role_assignments[role] = backend_name

    return role_assignments


def instantiate_backend(
    backend_name: str,
    store_path: Any,
    backend_config: Optional[Dict[str, Any]] = None,
) -> Any:
    """Import and instantiate a backend class.

    Applies layered config resolution:
        Layer 1: CLOUD_DEFAULTS
        Layer 2: BACKEND_DEFAULTS[backend_name]
        Layer 3+: backend_config (project config + CLI overrides)

    Args:
        backend_name: Registry key (e.g., "postgres", "neo4j", "arangodb").
        store_path: Path Attestor uses for side-car files (certs, etc.).
        backend_config: Config dict for backend (layers 3-5 merged).

    Returns:
        Instantiated backend object.
    """
    entry = BACKEND_REGISTRY[backend_name]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])

    merged = merge_config_layers(
        CLOUD_DEFAULTS,
        BACKEND_DEFAULTS.get(backend_name, {}),
        backend_config or {},
    )
    merged["_store_path"] = str(store_path)
    return cls(merged)
