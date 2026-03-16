---
phase: 01-zero-infrastructure-backends
plan: 01
subsystem: infra
tags: [cleanup, pgvector, neo4j, docker, chromadb, networkx]

requires: []
provides:
  - Clean codebase with no pgvector/Neo4j/Docker/OpenAI embedding code
  - Stripped MemoryConfig with only default_token_budget and min_results
  - Updated pyproject.toml with chromadb and networkx as required deps
affects: [01-02, 01-03]

tech-stack:
  added: [chromadb, networkx]
  patterns: [stubbed backends with TODO markers for Plan 03 rewiring]

key-files:
  created: []
  modified:
    - agent_memory/core.py
    - agent_memory/cli.py
    - agent_memory/retrieval/orchestrator.py
    - agent_memory/utils/config.py
    - pyproject.toml

key-decisions:
  - "Stubbed vector_store and graph to None with TODO markers rather than removing init paths entirely"
  - "Simplified doctor command to SQLite-only check rather than removing it"
  - "Kept close() method graph/vector cleanup paths for forward compatibility"

patterns-established:
  - "TODO rewire pattern: # TODO: rewire in Plan 03 marks all stubbed-out integration points"

requirements-completed: [CLEAN-01, CLEAN-02, CLEAN-03, CLEAN-04, CLEAN-05]

duration: 3min
completed: 2026-03-16
---

# Phase 1 Plan 01: Delete Old Backends Summary

**Removed all pgvector, Neo4j, Docker, and OpenAI embedding code; stripped config to 2 fields; added chromadb+networkx deps**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-16T00:43:29Z
- **Completed:** 2026-03-16T00:46:31Z
- **Tasks:** 2
- **Files modified:** 5 (plus 8 deleted)

## Accomplishments
- Deleted 8 files: vector_store.py, schema_pg.sql, neo4j_graph.py, embeddings.py, docker-compose.yml, and 3 test files
- Stubbed all vector/graph integration points in core.py with TODO markers for Plan 03
- Stripped MemoryConfig from 7 fields to 2 (default_token_budget, min_results)
- Updated pyproject.toml: chromadb and networkx as required deps, removed psycopg/neo4j/openai groups

## Task Commits

Each task was committed atomically:

1. **Task 1: Delete old backend files and their tests** - `82e43f7` (chore)
2. **Task 2: Strip config and update dependencies** - `7f3eb7e` (chore)

## Files Created/Modified
- `agent_memory/core.py` - Stubbed vector_store/graph init, commented out embedding/graph blocks in add/search
- `agent_memory/cli.py` - Removed _init_docker(), --no-docker, stripped doctor to SQLite-only
- `agent_memory/retrieval/orchestrator.py` - Commented out vector similarity layer
- `agent_memory/utils/config.py` - Stripped to 2 fields, removed env var overrides
- `pyproject.toml` - Updated deps and description

## Files Deleted
- `agent_memory/store/vector_store.py` - pgvector store
- `agent_memory/store/schema_pg.sql` - PostgreSQL schema
- `agent_memory/graph/neo4j_graph.py` - Neo4j graph
- `agent_memory/embeddings.py` - OpenAI/OpenRouter embedding client
- `docker-compose.yml` - Docker compose for PostgreSQL + Neo4j
- `tests/test_vector_store.py` - pgvector tests
- `tests/test_neo4j_graph.py` - Neo4j tests
- `tests/test_embeddings.py` - OpenAI embedding tests

## Decisions Made
- Stubbed vector_store and graph to None with TODO markers rather than removing init paths entirely -- preserves structure for Plan 03 rewiring
- Simplified doctor command to SQLite-only check rather than removing it entirely
- Kept close() method graph/vector cleanup paths for forward compatibility with new backends

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Next Phase Readiness
- Codebase is clean of all pgvector/Neo4j/Docker/OpenAI embedding code
- Ready for Plan 02 (ChromaDB vector backend) and Plan 03 (NetworkX graph + rewiring)
- All stubbed integration points marked with TODO comments

---
*Phase: 01-zero-infrastructure-backends*
*Completed: 2026-03-16*

## Self-Check: PASSED
