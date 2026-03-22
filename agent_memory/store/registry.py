"""Backend registry — maps backend names to implementations and resolves role assignments."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("agent_memory")


class BackendConflictError(Exception):
    """Raised when two backends claim the same role."""


BACKEND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sqlite": {
        "module": "agent_memory.store.sqlite_store",
        "class": "SQLiteStore",
        "roles": {"document"},
        "init_style": "path",
    },
    "chroma": {
        "module": "agent_memory.store.chroma_store",
        "class": "ChromaStore",
        "roles": {"vector"},
        "init_style": "path",
    },
    "networkx": {
        "module": "agent_memory.graph.networkx_graph",
        "class": "NetworkXGraph",
        "roles": {"graph"},
        "init_style": "path",
    },
    "arangodb": {
        "module": "agent_memory.store.arango_backend",
        "class": "ArangoBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
}

DEFAULT_BACKENDS = ["sqlite", "chroma", "networkx"]


def resolve_backends(
    backends: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Resolve which backend fills each role.

    Args:
        backends: Ordered list of backend names. Defaults to DEFAULT_BACKENDS.

    Returns:
        Dict mapping role -> backend_name (e.g., {"document": "sqlite", "vector": "chroma", "graph": "networkx"})

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

    Args:
        backend_name: Registry key (e.g., "sqlite", "arangodb").
        store_path: Path for path-based backends.
        backend_config: Config dict for config-based backends.

    Returns:
        Instantiated backend object.
    """
    entry = BACKEND_REGISTRY[backend_name]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])

    if entry["init_style"] == "path":
        return cls(store_path)
    elif entry["init_style"] == "config":
        config = backend_config or {}
        return cls(config)
    else:
        raise ValueError(f"Unknown init_style: {entry['init_style']!r}")
