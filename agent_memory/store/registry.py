"""Backend registry — maps backend names to implementations and resolves role assignments."""

from __future__ import annotations

import importlib
import logging
from typing import Any, Dict, List, Optional, Set

from agent_memory.store.connection import (
    BACKEND_DEFAULTS,
    CLOUD_DEFAULTS,
    merge_config_layers,
)

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
    "postgres": {
        "module": "agent_memory.store.postgres_backend",
        "class": "PostgresBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
    "gcp": {
        "module": "agent_memory.store.gcp_backend",
        "class": "GCPBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
    "azure": {
        "module": "agent_memory.store.azure_backend",
        "class": "AzureBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
    "aws": {
        "module": "agent_memory.store.aws_backend",
        "class": "AWSBackend",
        "roles": {"document", "vector", "graph"},
        "init_style": "config",
    },
}

DEFAULT_BACKENDS = ["sqlite", "chroma", "networkx"]


# Module-level constants referencing built-in registry entries.
# These are the targets of the entry-point declarations in pyproject.toml —
# they share the same underlying dict objects with BACKEND_REGISTRY (aliasing is intentional).
_BUILTIN_SQLITE = BACKEND_REGISTRY["sqlite"]
_BUILTIN_CHROMA = BACKEND_REGISTRY["chroma"]
_BUILTIN_NETWORKX = BACKEND_REGISTRY["networkx"]
_BUILTIN_ARANGO = BACKEND_REGISTRY["arangodb"]
_BUILTIN_POSTGRES = BACKEND_REGISTRY["postgres"]
_BUILTIN_AWS = BACKEND_REGISTRY["aws"]
_BUILTIN_AZURE = BACKEND_REGISTRY["azure"]
_BUILTIN_GCP = BACKEND_REGISTRY["gcp"]


def discover_backends() -> None:
    """Populate BACKEND_REGISTRY from importlib.metadata entry points.

    Entry-point group: 'memwright.backends'
    Each entry point should resolve to a dict with the same shape as the static registry entries:
        { "module": "...", "class": "...", "roles": {...}, "init_style": "..." }

    Static registry entries take precedence: if an entry point's name is already
    present in BACKEND_REGISTRY, the loader skips it. Failures to load a plugin
    are logged as warnings and do not prevent other plugins from loading.
    """
    from importlib.metadata import entry_points

    for ep in entry_points(group="memwright.backends"):
        if ep.name in BACKEND_REGISTRY:
            continue
        try:
            entry = ep.load()
        except Exception as e:
            logger.warning("Failed to load backend plugin %r: %s", ep.name, e)
            continue
        BACKEND_REGISTRY[ep.name] = entry


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
    discover_backends()

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

    For config-based backends, applies layered config resolution:
        Layer 1: CLOUD_DEFAULTS
        Layer 2: BACKEND_DEFAULTS[backend_name]
        Layer 3+: backend_config (project config + CLI overrides)

    Args:
        backend_name: Registry key (e.g., "sqlite", "arangodb").
        store_path: Path for path-based backends.
        backend_config: Config dict for config-based backends (layers 3-5 merged).

    Returns:
        Instantiated backend object.
    """
    entry = BACKEND_REGISTRY[backend_name]
    module = importlib.import_module(entry["module"])
    cls = getattr(module, entry["class"])

    if entry["init_style"] == "path":
        return cls(store_path)
    elif entry["init_style"] == "config":
        merged = merge_config_layers(
            CLOUD_DEFAULTS,
            BACKEND_DEFAULTS.get(backend_name, {}),
            backend_config or {},
        )
        # Pass store_path so backends can write cert files etc.
        merged["_store_path"] = str(store_path)
        return cls(merged)
    else:
        raise ValueError(f"Unknown init_style: {entry['init_style']!r}")
