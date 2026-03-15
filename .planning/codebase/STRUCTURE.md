# Codebase Structure

**Analysis Date:** 2025-03-15

## Directory Layout

```
agent-memory/
├── agent_memory/             # Main package
│   ├── __init__.py           # Package exports: AgentMemory, Memory, RetrievalResult
│   ├── core.py               # AgentMemory class (public API)
│   ├── models.py             # Memory, RetrievalResult dataclasses
│   ├── embeddings.py         # Embedding computation (OpenAI/OpenRouter)
│   ├── cli.py                # CLI entry point with argparse
│   ├── mab.py                # MemoryAgentBench benchmark
│   ├── locomo.py             # LOCOMO benchmark
│   │
│   ├── store/                # Storage layer
│   │   ├── __init__.py
│   │   ├── sqlite_store.py   # SQLite persistence (always available)
│   │   ├── vector_store.py   # pgvector semantic search (optional)
│   │   ├── schema.sql        # SQLite schema
│   │   └── schema_pg.sql     # PostgreSQL pgvector schema
│   │
│   ├── graph/                # Graph layer
│   │   ├── __init__.py
│   │   ├── neo4j_graph.py    # Neo4j entity/relation store (optional)
│   │   └── extractor.py      # Entity/relation extraction patterns
│   │
│   ├── retrieval/            # Retrieval orchestration
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # 3-layer cascade + RRF fusion
│   │   ├── tag_matcher.py    # Tag extraction from natural language
│   │   └── scorer.py         # Scoring boosts, dedup, budget assembly
│   │
│   ├── temporal/             # Temporal semantics
│   │   ├── __init__.py
│   │   └── manager.py        # Contradiction detection, supersession, timelines
│   │
│   ├── extraction/           # Memory extraction from conversations
│   │   ├── __init__.py
│   │   ├── extractor.py      # extract_memories() dispatcher
│   │   ├── llm_extractor.py  # LLM-based extraction
│   │   └── rule_based.py     # Rule-based extraction
│   │
│   ├── mcp/                  # Claude Code / MCP integration
│   │   ├── __init__.py
│   │   └── server.py         # MCP server factory with 8+ tools
│   │
│   └── utils/                # Utilities
│       ├── __init__.py
│       ├── config.py         # MemoryConfig dataclass, env loading
│       └── tokens.py         # Token estimation for budget assembly
│
├── tests/                    # Test suite
│   ├── __init__.py
│   ├── conftest.py           # pytest fixtures (Docker, agents)
│   ├── test_core.py          # AgentMemory API tests
│   ├── test_cli.py           # CLI command tests
│   ├── test_embeddings.py    # Embedding API tests
│   ├── test_extraction.py    # Memory extraction tests
│   ├── test_graph.py         # Neo4j graph tests
│   ├── test_mab.py           # MemoryAgentBench tests
│   ├── test_retrieval.py     # Orchestrator + layer tests
│   ├── test_sqlite.py        # SQLiteStore tests
│   ├── test_temporal.py      # Temporal semantics tests
│   └── test_vector.py        # pgvector tests
│
├── docs/                     # Documentation
│   ├── API.md                # API reference
│   └── ... (other docs)
│
├── examples/                 # Example scripts
│   └── ... (usage examples)
│
├── scripts/                  # Utility scripts
│   └── ... (setup, benchmarking)
│
├── pyproject.toml            # Project metadata, dependencies, entry points
├── README.md                 # Main documentation
├── docker-compose.yml        # Docker services (PostgreSQL, Neo4j)
├── .env.example              # Example environment variables
└── LICENSE                   # Apache 2.0

```

## Directory Purposes

**agent_memory/:**
- Purpose: Main package containing all source code
- Contains: Core API, storage backends, retrieval layers, extraction, CLI, MCP server
- Key files: `core.py` (main entry point), `models.py` (data structures)

**agent_memory/store/:**
- Purpose: Data persistence abstraction
- Contains: SQLiteStore (always available), VectorStore (optional pgvector)
- Key files: `sqlite_store.py` (repository pattern), `schema.sql` (SQLite DDL)
- Note: VectorStore is optional — failures don't break core functionality

**agent_memory/graph/:**
- Purpose: Entity relationship graph for multi-hop retrieval expansion
- Contains: Neo4j client, entity/relation extraction patterns
- Key files: `neo4j_graph.py` (Neo4j operations), `extractor.py` (pattern-based extraction)
- Note: Optional — if unavailable, 3-layer retrieval falls back to tag + vector

**agent_memory/retrieval/:**
- Purpose: Smart retrieval orchestration with multi-source fusion
- Contains: 3-layer cascade (tag → graph → vector), RRF fusion, scoring boosts
- Key files: `orchestrator.py` (recall pipeline), `scorer.py` (temporal/entity/RRF logic)

**agent_memory/temporal/:**
- Purpose: Time-aware memory semantics
- Contains: Contradiction detection, supersession, timeline queries
- Key files: `manager.py` (all temporal operations)
- Pattern: Rule-based contradiction (same entity + category + different content)

**agent_memory/extraction/:**
- Purpose: Convert unstructured conversation data to memories
- Contains: Rule-based and LLM-based extraction strategies
- Key files: `extractor.py` (dispatcher), `llm_extractor.py` (OpenAI), `rule_based.py` (patterns)

**agent_memory/mcp/:**
- Purpose: Claude Code integration via Model Context Protocol
- Contains: MCP server exposing 8+ tools
- Key files: `server.py` (server factory)

**agent_memory/utils/:**
- Purpose: Shared utilities
- Contains: Config management, token estimation
- Key files: `config.py` (MemoryConfig dataclass)

**tests/:**
- Purpose: Test coverage for all layers
- Contains: Unit tests for each module, integration tests with Docker services
- Pattern: Fixtures in conftest.py, tests use `mem` fixture with real SQLite + PostgreSQL + Neo4j

## Key File Locations

**Entry Points:**
- `agent_memory/core.py`: Main AgentMemory class, public API (add, recall, search, timeline, extract, health)
- `agent_memory/cli.py`: CLI dispatcher with 12+ commands (add, recall, search, list, timeline, stats, doctor, etc.)
- `agent_memory/mcp/server.py`: MCP server factory for Claude Code integration

**Configuration:**
- `agent_memory/utils/config.py`: MemoryConfig dataclass with env var loading
- `pyproject.toml`: Project metadata, optional dependencies (vectors, neo4j, extraction, mcp, production, all)
- `.env.example`: Template for required env vars (PG_CONNECTION_STRING, NEO4J_PASSWORD, OPENAI_API_KEY, etc.)

**Core Logic:**
- `agent_memory/store/sqlite_store.py`: SQLite CRUD, schema management (always available)
- `agent_memory/store/vector_store.py`: pgvector semantic search (optional)
- `agent_memory/graph/neo4j_graph.py`: Entity graph operations (optional)
- `agent_memory/retrieval/orchestrator.py`: 3-layer retrieval cascade with RRF fusion
- `agent_memory/temporal/manager.py`: Contradiction detection, supersession, timeline queries

**Testing:**
- `tests/conftest.py`: pytest fixtures (Docker cleanup, AgentMemory instances)
- `tests/test_core.py`: AgentMemory API integration tests
- `tests/test_retrieval.py`: Orchestrator and retrieval layer tests
- `tests/test_temporal.py`: Contradiction detection and supersession tests

## Naming Conventions

**Files:**
- Module files: lowercase_with_underscores.py (e.g., `sqlite_store.py`, `neo4j_graph.py`)
- Test files: `test_*.py` (e.g., `test_core.py`, `test_retrieval.py`)
- Schema files: `schema.sql`, `schema_pg.sql`

**Directories:**
- Package dirs: lowercase plural when grouping related code (store/, graph/, retrieval/, temporal/, extraction/, utils/, mcp/)

**Functions/Methods:**
- Public methods: lowercase_with_underscores (e.g., `add()`, `recall()`, `tag_search()`)
- Private methods: _leading_underscore (e.g., `_init_schema()`, `_llm_extract()`)
- Dataclass fields: lowercase_with_underscores (e.g., `created_at`, `valid_from`, `superseded_by`)

**Classes:**
- PascalCase (e.g., `AgentMemory`, `SQLiteStore`, `RetrievalOrchestrator`, `TemporalManager`)

**Variables:**
- Simple: lowercase (e.g., `results`, `memory`, `embedding`)
- Constants: UPPERCASE_WITH_UNDERSCORES (e.g., `_STOP_WORDS`, `_SCHEMA_PATH`)

## Where to Add New Code

**New Feature:**
- Primary code: Add module to appropriate layer directory
  - Retrieval enhancement: `agent_memory/retrieval/new_matcher.py`
  - Storage backend: `agent_memory/store/new_backend.py`
  - Extraction strategy: `agent_memory/extraction/new_extractor.py`
- Tests: Add corresponding `tests/test_new_feature.py`
- Integration: Wire into `core.py` or relevant orchestrator

**New Component/Module:**
- Implementation: Create new file in appropriate package directory (e.g., `agent_memory/retrieval/new_module.py`)
- Pattern: Encapsulate related logic in a class (repository pattern for data access, manager pattern for orchestration)
- Dependencies: Import only from same layer or lower layers (store depends only on models, retrieval depends on store and models)

**Utilities:**
- Shared helpers: Add to `agent_memory/utils/` (e.g., `agent_memory/utils/new_helper.py`)
- Import pattern: Exposed via `__all__` in utils/__init__.py for clean imports

**Database Schema:**
- SQLite schema: Modify `agent_memory/store/schema.sql` only
- PostgreSQL schema: Modify `agent_memory/store/schema_pg.sql` only
- Migration: No migration system — schema changes are manual

## Special Directories

**store/schema.sql:**
- Purpose: SQLite DDL defining memories table and indexes
- Generated: No (hand-written)
- Committed: Yes (version controlled)
- Note: Applied on first SQLiteStore initialization via _init_schema()

**store/schema_pg.sql:**
- Purpose: PostgreSQL DDL for pgvector extension and memories_vectors table
- Generated: No (hand-written)
- Committed: Yes (version controlled)
- Note: Applied on first VectorStore initialization

**tests/:**
- Purpose: Test suite requiring Docker containers (PostgreSQL, Neo4j)
- Docker requirements: postgres:15 with pgvector extension, neo4j:latest
- Run: `.venv/bin/pytest tests/ -v` (Docker must be running)
- Generated: No (test files are source code)
- Committed: Yes (all test files committed)

**dist/:**
- Purpose: Distribution artifacts (wheels, sdists)
- Generated: Yes (via `python -m build`)
- Committed: No (only source committed)

**benchmark-logs/:**
- Purpose: Benchmark output logs (MAB, LOCOMO)
- Generated: Yes (created during benchmark runs)
- Committed: No (ignored in .gitignore)

## Dependency Layers

**No circular dependencies.** Import hierarchy:

```
models.py (no imports from agent_memory)
    ↑
utils/config.py, utils/tokens.py (import models)
    ↑
store/sqlite_store.py (imports models, utils)
    ↑
store/vector_store.py (imports models, utils)
graph/extractor.py, graph/neo4j_graph.py (import models, utils)
temporal/manager.py (imports models, store)
retrieval/tag_matcher.py, retrieval/scorer.py, retrieval/orchestrator.py (import models, store, temporal, etc.)
extraction/extractor.py, extraction/llm_extractor.py, extraction/rule_based.py (import models)
    ↑
core.py (imports all above)
    ↑
cli.py, mcp/server.py (import core)
```

---

*Structure analysis: 2025-03-15*
