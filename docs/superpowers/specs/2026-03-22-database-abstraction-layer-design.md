# Database Abstraction Layer Design

## Problem

Memwright hardcodes SQLite + ChromaDB + NetworkX. Users cannot swap backends. Some databases (ArangoDB) cover all three roles in one product; others (PostgreSQL + Neo4j) split across two. The abstraction must handle both cases cleanly.

## Design

### Three Abstract Interfaces

Located in `agent_memory/store/base.py`:

**DocumentStore** — CRUD + search for Memory objects:
- `insert(memory) -> Memory`
- `get(memory_id) -> Optional[Memory]`
- `update(memory) -> Memory`
- `delete(memory_id) -> bool`
- `list_memories(status, category, entity, after, before, limit) -> List[Memory]`
- `tag_search(tags, category, limit) -> List[Memory]`
- `execute(sql, params) -> List[Dict]` (raw query passthrough)
- `archive_before(date) -> int`
- `compact() -> int`
- `stats() -> Dict`
- `close()`

**VectorStore** — Embedding storage + similarity search:
- `add(memory_id, content) -> None`
- `search(query_text, limit) -> List[Dict]`
- `delete(memory_id) -> bool`
- `count() -> int`
- `close()`

**GraphStore** — Entity graph with traversal:
- `add_entity(name, entity_type, attributes) -> None`
- `add_relation(from_entity, to_entity, relation_type, metadata) -> None`
- `get_related(entity, depth) -> List[str]`
- `get_subgraph(entity, depth) -> Dict`
- `get_entities(entity_type) -> List[Dict]`
- `get_edges(entity) -> List[Dict]`
- `stats() -> Dict`
- `save()`
- `close()`

Each interface is an ABC. Backends declare which roles they support via a `ROLES` class attribute.

### Backend Registry

Located in `agent_memory/store/registry.py`:

```python
BACKEND_REGISTRY = {
    "sqlite":   {"module": "agent_memory.store.sqlite_store",   "class": "SQLiteStore",    "roles": {"document"}},
    "chroma":   {"module": "agent_memory.store.chroma_store",   "class": "ChromaStore",    "roles": {"vector"}},
    "networkx": {"module": "agent_memory.store.networkx_graph", "class": "NetworkXGraph",  "roles": {"graph"}},
    "arangodb": {"module": "agent_memory.store.arango_backend", "class": "ArangoBackend",  "roles": {"document", "vector", "graph"}},
}
```

**Resolution logic**:
1. Iterate `backends` list in order
2. For each backend, check which roles it claims
3. If a role is already claimed by a previous backend, raise `BackendConflictError`
4. After all backends processed, check all 3 roles are filled (document, vector, graph are required; vector and graph degrade gracefully if unfilled — matching current behavior)

### Configuration

Extends `MemoryConfig` in `agent_memory/utils/config.py`:

```json
{
  "default_token_budget": 2000,
  "min_results": 3,
  "backends": ["arangodb"],
  "arangodb": {
    "mode": "local",
    "port": 8529
  }
}
```

- **Default**: when `backends` is omitted, falls back to `["sqlite", "chroma", "networkx"]` (current zero-config behavior, no Docker)
- **`mode: "local"`**: auto-manages Docker container
- **`mode: "cloud"`**: requires `url` + auth credentials
- **Environment variable resolution**: values starting with `$` are resolved from `os.environ` at runtime (e.g., `"password": "$ARANGO_PASSWORD"`)

Backend-specific config keys match registry names. Each backend receives its own config dict on init.

### Docker Manager

Located in `agent_memory/infra/docker.py`:

```python
class DockerManager:
    ensure_running(backend_name, image, port, env) -> ContainerInfo
    stop(backend_name) -> None
    health_check(backend_name) -> bool
    cleanup() -> None
```

- Uses `docker` CLI via subprocess — no `docker-py` dependency
- Container names prefixed: `memwright-arangodb`, `memwright-postgres`, etc.
- Waits for health check with timeout before returning
- Containers persist across sessions (reusable)
- Only invoked when `mode: "local"` — never for default SQLite+ChromaDB+NetworkX setup
- CLI command `agent-memory docker stop` for explicit cleanup

### ArangoDB Backend

Single class in `agent_memory/store/arango_backend.py`:

```python
class ArangoBackend(DocumentStore, VectorStore, GraphStore):
    ROLES = {"document", "vector", "graph"}
```

**Document role** — `memories` collection:
- CRUD via python-arango client methods
- Persistent indexes on `category`, `entity`, `status`, `created_at`
- Tag search via AQL with `LIKE` operator

**Vector role** — same `memories` collection:
- Embeddings stored as `vector_data` field on memory documents
- Local sentence-transformers (`all-MiniLM-L6-v2`, 384D) generates embeddings — ArangoDB stores them
- Vector index: `collection.add_index(type="vector", fields=["vector_data"], params={"metric": "cosine", "dimension": 384, "nLists": 100})`
- Search via AQL: `APPROX_NEAR_COSINE(doc.vector_data, @query_vector)`
- Index created after first batch of documents have embeddings (ArangoDB requires data before index)

**Graph role** — named graph `memory_graph`:
- Vertex collection: `entities`
- Edge collection: `relations` (edge definition: entities -> entities)
- BFS traversal via AQL: `FOR v, e, p IN 1..@depth ANY @start GRAPH 'memory_graph'`
- Native graph engine — faster than NetworkX at scale

**Single connection**: one `ArangoClient` instance, one database. Writes across all three concerns can be transactional.

**Docker config** (local mode):
- Image: `arangodb/arangodb:latest`
- Port: 8529 (configurable)
- Auto-creates database `memwright` and all collections/indexes on first use

### core.py Changes

`AgentMemory.__init__` updated to:
1. Load config (unchanged)
2. Resolve backends from config via registry
3. Instantiate backend classes, passing their config dicts
4. Assign to `self._store` (DocumentStore), `self._vector_store` (VectorStore), `self._graph` (GraphStore)
5. These may all point to the same object (ArangoDB) or different objects (SQLite+ChromaDB+NetworkX)

Rest of core.py is unchanged — it already uses these three references.

### Current Backends Refactored

- `SQLiteStore` adds `class ROLES = {"document"}` and inherits from `DocumentStore`
- `ChromaStore` adds `class ROLES = {"vector"}` and inherits from `VectorStore`
- `NetworkXGraph` adds `class ROLES = {"graph"}` and inherits from `GraphStore`
- No behavior changes — just interface conformance

### File Structure (new/modified)

```
agent_memory/
  store/
    base.py              -- NEW: DocumentStore, VectorStore, GraphStore ABCs
    registry.py          -- NEW: Backend registry + resolver
    sqlite_store.py      -- MODIFIED: inherits DocumentStore
    chroma_store.py      -- MODIFIED: inherits VectorStore
    arango_backend.py    -- NEW: ArangoBackend (all 3 roles)
  graph/
    networkx_graph.py    -- MODIFIED: inherits GraphStore
  infra/
    __init__.py          -- NEW
    docker.py            -- NEW: DockerManager
  utils/
    config.py            -- MODIFIED: backends + per-backend config
  core.py                -- MODIFIED: resolve backends from registry
```

### Dependencies

- `python-arango` — added as optional dependency (`pip install memwright[arangodb]`)
- `sentence-transformers` — already required (used by ChromaDB and ArangoDB vector)
- No new required dependencies for default setup

### Migration Path

- Existing users: zero change. No config = SQLite+ChromaDB+NetworkX (unchanged)
- New ArangoDB users: add `backends: ["arangodb"]` to config, optionally set mode
- Future backends (Postgres, Neo4j): implement the interfaces, add to registry

## Non-Goals

- Data migration between backends (future work)
- Multi-region / replication config
- Backend-specific query passthrough beyond `execute()`
