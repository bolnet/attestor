# Phase 1: Zero-Infrastructure Backends - Context

**Gathered:** 2026-03-15
**Status:** Ready for planning

<domain>
## Phase Boundary

Replace pgvector (PostgreSQL) and Neo4j with ChromaDB (via MCP stdio subprocess) and NetworkX (in-memory, JSON-persisted). Remove all Docker dependencies. Auto-provision all storage on first use. Result: `AgentMemory("./path")` works with zero config, zero env vars, zero Docker.

</domain>

<decisions>
## Implementation Decisions

### ChromaDB integration
- Use ChromaDB in-process via `chromadb.PersistentClient` (simpler than MCP stdio, still zero-config)
- ChromaDB handles embeddings internally via built-in sentence-transformers
- Data stored inside store path: `{store_path}/chroma/` — everything in one directory
- Note: Originally planned as MCP stdio subprocess (like claude-mem), changed to in-process for simplicity

### Deletion strategy
- Claude's discretion on ordering (delete-first vs build-alongside-then-swap)
- Claude's discretion on Docker file handling
- Claude's discretion on test migration strategy (rewrite vs fresh)
- Benchmarks (mab.py, locomo.py): Claude's discretion on handling broken imports

### Graph persistence timing
- Claude's discretion on save timing (after-every-write vs batch vs periodic)
- Claude's discretion on serialization format (node_link_data JSON vs pickle)
- Claude's discretion on size limits
- Claude's discretion on corruption recovery

### Embedding model
- Claude's discretion on model choice (all-MiniLM-L6-v2 default vs all-mpnet-base-v2)
- Claude's discretion on first-run download UX
- Existing data: clean slate by default + provide `memwright re-embed` command for migration
- Claude's discretion on offline behavior

### Claude's Discretion
Broad discretion granted on implementation details for this phase:
- Deletion ordering and test strategy
- Graph persistence mechanics (timing, format, recovery)
- Embedding model selection and download UX
- Offline/failure handling for ChromaDB subprocess and embeddings
- Exception handling approach (user noted broad `except Exception:` is a known concern)

</decisions>

<specifics>
## Specific Ideas

- Match claude-mem's ChromaDB integration pattern exactly (chroma-mcp via stdio)
- `{store_path}/chroma/` keeps everything portable — one directory = one agent's memory
- Auto-restart on ChromaDB crash — resilience over simplicity
- Re-embed command for users migrating from pgvector (1536-dim OpenAI → smaller local model)

</specifics>

<code_context>
## Existing Code Insights

### Reusable Assets
- `agent_memory/store/sqlite_store.py`: Carries forward entirely — SQLite is unchanged
- `agent_memory/retrieval/orchestrator.py`: Duck-typed, uses null guards — backend swap is init-only
- `agent_memory/temporal/manager.py`: No changes needed — depends only on SQLiteStore
- `agent_memory/retrieval/scorer.py`: RRF fusion, temporal/entity boosts — carries forward
- `agent_memory/retrieval/tag_matcher.py`: Tag extraction — carries forward
- `agent_memory/models.py`: Memory and RetrievalResult dataclasses — carries forward

### Established Patterns
- Optional service init with try/except in `core.py` lines 48-93 (pattern to follow for new backends)
- Duck typing in orchestrator: `if self.vector_store:` / `if self.graph:` guards
- Config via `MemoryConfig` dataclass with env var overrides in `__post_init__`
- Section markers in core.py: `# ── Write ──`, `# ── Read ──`

### Integration Points
- `core.py.__init__()` — main rewrite target (lines 48-93)
- `core.py.add()` — lines 165-191 where vector_store.add() and graph operations happen
- `core.py.health()` — lines 360-532, entire Docker/container checking section replaced
- `pyproject.toml` — dependency swap (remove psycopg/neo4j, add chromadb/networkx)
- `utils/config.py` — strip pg_connection_string, neo4j_* fields

### Known Concerns (from codebase analysis)
- Broad `except Exception:` blocks mask real bugs (14 locations in core.py)
- Memory objects mutated in-place during supersession (violates immutability)
- Tag search uses LIKE without anchoring (substring false positives)
- These are pre-existing — fix opportunistically during rewrite, not scope of this phase

</code_context>

<deferred>
## Deferred Ideas

None — discussion stayed within phase scope

</deferred>

---

*Phase: 01-zero-infrastructure-backends*
*Context gathered: 2026-03-15*
