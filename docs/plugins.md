# Writing a memwright Backend Plugin

Third parties can ship a backend (e.g., `memwright-redis`) as a separate PyPI package and have it auto-register when installed alongside memwright. Memwright discovers plugins via Python's `importlib.metadata` entry-point mechanism; no monkey-patching, no manual registration, no imports from user code.

## Minimal plugin layout

```
memwright_redis/
  pyproject.toml
  src/memwright_redis/
    __init__.py
    backend.py
```

## `pyproject.toml`

Declare your package and register it under the `memwright.backends` entry-point group. The right-hand side is a `module:attribute` path to the registry entry (a dict — see below).

```toml
[project]
name = "memwright-redis"
version = "0.1.0"
dependencies = ["memwright>=2.1", "redis>=5.0"]

[project.entry-points."memwright.backends"]
redis = "memwright_redis:REGISTRY_ENTRY"
```

The entry-point key (`redis` above) is the name users will put in `config.toml` under `backends = [...]`.

## `__init__.py`

Expose a dict with the four required keys: `module`, `class`, `roles`, and `init_style`. Memwright validates this shape at discovery time and skips malformed entries with a warning (the whole registry does NOT crash).

```python
REGISTRY_ENTRY = {
    "module": "memwright_redis.backend",
    "class": "RedisBackend",
    "roles": {"document"},
    "init_style": "config",
}
```

### Entry fields

| Key          | Type         | Meaning                                                                         |
| ------------ | ------------ | ------------------------------------------------------------------------------- |
| `module`     | `str`        | Importable Python path to the module containing your backend class.             |
| `class`      | `str`        | Class name inside that module. Must implement the store protocol(s) below.      |
| `roles`      | `set[str]`   | Which roles this backend can fill. One or more of: `"document"`, `"vector"`, `"graph"`. |
| `init_style` | `str`        | `"path"` (backend is constructed with a filesystem path) or `"config"` (backend receives a config dict). |

### Name-collision rules

Built-in backend names (`sqlite`, `chroma`, `networkx`, `arangodb`, `postgres`, `aws`, `azure`, `gcp`) are protected — a third-party entry point using any of these names is ignored. Pick a unique name for your backend.

## `backend.py`

Implement at least one of the store protocols defined in `agent_memory.store.base` (or `agent_memory.graph.*` for graph backends). For example, a document store backend must implement `DocumentStore`. Multi-role backends (like ArangoDB, which serves `document + vector + graph`) implement each protocol.

Constructor signature depends on `init_style`:

```python
# init_style = "path"
class MyBackend:
    def __init__(self, path: pathlib.Path) -> None:
        ...

# init_style = "config"
class MyBackend:
    def __init__(self, config: dict) -> None:
        ...
```

## Selecting your backend

Users add it to their `config.toml`:

```toml
backends = ["redis", "chroma", "networkx"]

[redis]
url = "redis://localhost:6379"
database = 0
```

When `AgentMemory(path)` is constructed, memwright calls `discover_backends()`, which reads all `memwright.backends` entry points exactly once (idempotent on subsequent calls), validates each, merges the valid ones into the in-process registry, and resolves the user's requested backends against it.

## Error-handling contract

- If your plugin's `ep.load()` raises any `Exception`, memwright logs a warning (via the `agent_memory` logger) and continues without your backend registered. The user's `AgentMemory` init will then fail when resolving the unknown backend name — so any load-time exception surfaces to the user as a clear configuration error rather than as a stack trace from deep inside the plugin.
- If your registry entry is malformed (not a dict, or missing one of the four required keys), memwright logs a warning with `"invalid shape"` and skips the entry.
- Your plugin must not mutate `agent_memory.store.registry.BACKEND_REGISTRY` directly. Declare everything via the entry point.

## Testing your plugin

```python
import importlib.metadata
import agent_memory.store.registry as registry

registry._backends_discovered = False  # force re-scan
registry.discover_backends()
assert "redis" in registry.BACKEND_REGISTRY
```

If your backend does not appear after `discover_backends()`, check:
1. Did you `pip install -e .` (or `pip install memwright-redis`) so entry-point metadata is registered?
2. Does `python -c "from importlib.metadata import entry_points; print(list(entry_points(group='memwright.backends')))"` list your entry?
3. Is your `REGISTRY_ENTRY` a dict with exactly the four required keys?

## Reference implementation

See `agent_memory/store/registry.py` for the built-in backend entries — they use the same shape your plugin must produce.
