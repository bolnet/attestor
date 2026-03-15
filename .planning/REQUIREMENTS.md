# Requirements: Memwright v2

**Defined:** 2026-03-15
**Core Value:** Zero-config automatic memory that just works — install and forget

## v1 Requirements

### Backend Cleanup

- [ ] **CLEAN-01**: All pgvector/PostgreSQL code removed (vector_store.py, schema_pg.sql, psycopg deps)
- [ ] **CLEAN-02**: All Neo4j code removed (neo4j_graph.py, neo4j driver dep)
- [ ] **CLEAN-03**: Docker compose file and Docker health checks removed
- [ ] **CLEAN-04**: OpenAI/OpenRouter embedding client removed (embeddings.py rewritten or deleted)
- [ ] **CLEAN-05**: Config dataclass stripped of pg_connection_string, neo4j_* fields

### Vector Search

- [ ] **VEC-01**: ChromaDB runs as MCP stdio subprocess (not in-process)
- [ ] **VEC-02**: ChromaDB uses built-in sentence-transformers embeddings (no API key)
- [ ] **VEC-03**: ChromaDB persistent storage at `{store_path}/chroma/`
- [ ] **VEC-04**: Vector search returns results compatible with retrieval orchestrator (memory_id, content, distance)
- [ ] **VEC-05**: ChromaDB auto-provisions on first use (collection created automatically)

### Entity Graph

- [ ] **GRAPH-01**: NetworkX MultiDiGraph replaces Neo4j for entity storage
- [ ] **GRAPH-02**: Graph persisted as JSON at `{store_path}/graph.json`
- [ ] **GRAPH-03**: Multi-hop traversal (get_related with depth parameter) works identically to Neo4j version
- [ ] **GRAPH-04**: Entity/relation extraction populates NetworkX graph on memory add
- [ ] **GRAPH-05**: Graph auto-provisions on first use (empty graph if no file exists)

### Auto-Provisioning

- [ ] **PROV-01**: `AgentMemory("./path")` creates all storage (SQLite, ChromaDB, NetworkX) with zero config
- [ ] **PROV-02**: No environment variables required by default
- [ ] **PROV-03**: No Docker required
- [ ] **PROV-04**: Health check updated to reflect new stack (no Docker/container checks)

### Plugin System

- [ ] **PLUG-01**: Plugin manifest at `.claude-plugin/plugin.json` with marketplace metadata
- [ ] **PLUG-02**: Marketplace catalog at `.claude-plugin/marketplace.json`
- [ ] **PLUG-03**: Users can install via `/plugin marketplace add bolnet/agent-memory`
- [ ] **PLUG-04**: Plugin bundles MCP server via `.mcp.json`
- [ ] **PLUG-05**: Plugin includes CLAUDE.md with usage instructions

### Auto-Capture Hooks

- [ ] **HOOK-01**: SessionStart hook injects relevant memories into conversation context
- [ ] **HOOK-02**: PostToolUse hook captures observations from Write/Edit/Bash tool use
- [ ] **HOOK-03**: Stop hook summarizes session and stores key decisions as memories
- [ ] **HOOK-04**: Hooks registered in `hooks/hooks.json` within plugin structure
- [ ] **HOOK-05**: Hook scripts callable as shell commands pointing to Python entry points

### Skills

- [ ] **SKILL-01**: `/memwright:mem-recall` skill for natural language memory search
- [ ] **SKILL-02**: `/memwright:mem-timeline` skill for entity timeline queries
- [ ] **SKILL-03**: `/memwright:mem-health` skill for component health check

### MCP Enhancements

- [ ] **MCP-01**: MCP resources expose entities as @-mentionable (`@memwright:entity://name`)
- [ ] **MCP-02**: MCP resources expose recent memories as @-mentionable
- [ ] **MCP-03**: MCP prompts register `recall` as native `/mcp__memwright__recall` command
- [ ] **MCP-04**: MCP prompts register `timeline` as native `/mcp__memwright__timeline` command
- [ ] **MCP-05**: Existing 8 MCP tools updated to work with new backends

### Distribution

- [ ] **DIST-01**: `pip install memwright` works for any MCP client (zero-config)
- [ ] **DIST-02**: `/plugin marketplace add bolnet/agent-memory` works for Claude Code
- [ ] **DIST-03**: PyPI package updated with new dependencies (chromadb, networkx, no psycopg/neo4j)
- [ ] **DIST-04**: MCP Registry listing updated

## v2 Requirements

### Web Viewer

- **WEB-01**: React dashboard for browsing memories
- **WEB-02**: Real-time observation viewing during sessions

### Benchmarks

- **BENCH-01**: MAB benchmark updated for ChromaDB + NetworkX backends
- **BENCH-02**: LOCOMO benchmark updated for new backends

## Out of Scope

| Feature | Reason |
|---------|--------|
| pgvector/Neo4j backends | Clean break — removed entirely, no optional extras |
| OpenAI/OpenRouter embeddings | Using ChromaDB built-in, no API key dependency |
| Docker support | Zero-infrastructure is the whole point |
| Localized modes | Not a priority for dev tool |
| Explicit memory_add in default flow | Auto-capture via hooks is the primary path |
| Mobile/desktop app | CLI + plugin is the interface |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| CLEAN-01 | — | Pending |
| CLEAN-02 | — | Pending |
| CLEAN-03 | — | Pending |
| CLEAN-04 | — | Pending |
| CLEAN-05 | — | Pending |
| VEC-01 | — | Pending |
| VEC-02 | — | Pending |
| VEC-03 | — | Pending |
| VEC-04 | — | Pending |
| VEC-05 | — | Pending |
| GRAPH-01 | — | Pending |
| GRAPH-02 | — | Pending |
| GRAPH-03 | — | Pending |
| GRAPH-04 | — | Pending |
| GRAPH-05 | — | Pending |
| PROV-01 | — | Pending |
| PROV-02 | — | Pending |
| PROV-03 | — | Pending |
| PROV-04 | — | Pending |
| PLUG-01 | — | Pending |
| PLUG-02 | — | Pending |
| PLUG-03 | — | Pending |
| PLUG-04 | — | Pending |
| PLUG-05 | — | Pending |
| HOOK-01 | — | Pending |
| HOOK-02 | — | Pending |
| HOOK-03 | — | Pending |
| HOOK-04 | — | Pending |
| HOOK-05 | — | Pending |
| SKILL-01 | — | Pending |
| SKILL-02 | — | Pending |
| SKILL-03 | — | Pending |
| MCP-01 | — | Pending |
| MCP-02 | — | Pending |
| MCP-03 | — | Pending |
| MCP-04 | — | Pending |
| MCP-05 | — | Pending |
| DIST-01 | — | Pending |
| DIST-02 | — | Pending |
| DIST-03 | — | Pending |
| DIST-04 | — | Pending |

**Coverage:**
- v1 requirements: 40 total
- Mapped to phases: 0
- Unmapped: 40

---
*Requirements defined: 2026-03-15*
*Last updated: 2026-03-15 after initial definition*
