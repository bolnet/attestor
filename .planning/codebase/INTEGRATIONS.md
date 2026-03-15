# External Integrations

**Analysis Date:** 2026-03-15

## APIs & External Services

**Embeddings (Semantic Search):**
- OpenRouter (preferred) - Text embedding generation via `openai/text-embedding-3-small` model
  - SDK/Client: `openai` Python package
  - Auth: `OPENROUTER_API_KEY` environment variable
  - Base URL: `https://openrouter.ai/api/v1`

- OpenAI (fallback) - Text embedding generation via `text-embedding-3-small` model
  - SDK/Client: `openai` Python package
  - Auth: `OPENAI_API_KEY` environment variable
  - Provider priority: OpenRouter checked first, falls back to OpenAI if key is available

**LLM Services (Optional/Future):**
- Anthropic Claude - Optional integration for advanced memory extraction/contradiction detection
  - Auth: `ANTHROPIC_API_KEY` environment variable
  - Status: Currently optional (not used in core retrieval)

## Data Storage

**Databases:**

**SQLite (Core - Always Required):**
- Type: Embedded relational database
- Location: `{memory_path}/memory.db`
- Client: Python `sqlite3` (built-in)
- Schemas: `agent_memory/store/schema.sql`
- Features: WAL mode (write-ahead logging), foreign key constraints enabled
- Contents: Memory records (content, tags, category, entity, timestamps, confidence, status, metadata)

**PostgreSQL 16 + pgvector (Required - Docker):**
- Type: PostgreSQL with pgvector extension for vector similarity search
- Connection: Environment variable `PG_CONNECTION_STRING`
- Default: `postgresql://memwright:memwright@localhost:5432/memwright`
- Client: `psycopg` (PostgreSQL async/sync driver) with `psycopg-pool` for connection pooling
- Schemas: `agent_memory/store/schema_pg.sql`
- Contents: Memory embeddings (1536-dimensional vectors via text-embedding-3-small model)
- Search: IVFFlat indexing for approximate nearest neighbor queries
- Integration point: `agent_memory.store.vector_store.VectorStore` class

**Neo4j 5 Community (Required - Docker):**
- Type: Property graph database for entity relationships
- Connection: Bolt protocol at `NEO4J_URI` (default: `bolt://localhost:7687`)
- Auth: `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE` environment variables
- Client: `neo4j` Python driver
- Schemas: Defined in code via Cypher (no SQL schema files)
- Contents: Entity nodes with relationships (RELATED_TO, MENTIONS, etc.)
- Indexes: Entity name indexing for fast lookups
- Integration point: `agent_memory.graph.neo4j_graph.Neo4jGraph` class

**File Storage:**
- Local filesystem only - No cloud file storage
- Config files: `{memory_path}/config.json` (JSON format)
- Exported data: JSON export via CLI command `agent-memory export`

**Caching:**
- Embedding LRU cache - In-memory cache of 256 most recent embeddings (avoids redundant API calls)
- Cache implementation: `agent_memory/embeddings.py` using OrderedDict
- Invalidated on `embeddings.reset()` (test cleanup)

## Authentication & Identity

**Auth Provider:**
- None for internal operations
- API key authentication for external embedding services (OpenRouter/OpenAI)
- Database credentials for PostgreSQL and Neo4j (environment variables)

**Security:**
- No hardcoded credentials - All secrets via environment variables
- Docker credentials: Default test credentials in docker-compose.yml (for development only)
  - PostgreSQL: `memwright:memwright`
  - Neo4j: `neo4j:memwright`

## Monitoring & Observability

**Error Tracking:**
- None configured - Errors logged to Python logging module
- Logger: `logging.getLogger("agent_memory")` in `agent_memory/core.py`

**Logs:**
- Python logging - Log errors when services fail to initialize
- Example: Database connection failures logged as errors, non-fatal to SQLite core
- Health check: `agent_memory.cli.main` with `doctor` command
- Health tools: `AgentMemory.health()` method and MCP `memory_health` tool

**Observability Details:**
- Non-fatal errors in pgvector/Neo4j initialization - Core SQLite path always works
- Graceful degradation: If pgvector unavailable, vector search disabled; if Neo4j unavailable, graph search disabled
- No distributed tracing or metrics collection

## CI/CD & Deployment

**Hosting:**
- PyPI - Python Package Index for pip distribution
- GitHub Releases - Source for triggering PyPI publish workflow
- MCP Registry - Model Context Protocol registry (`io.github.bolnet/memwright`)

**CI Pipeline:**
- GitHub Actions - `.github/workflows/workflow.yml`
- Trigger: On GitHub release publication
- Steps:
  1. Checkout code
  2. Set up Python 3.12
  3. Install build tools (`pip install build`)
  4. Build wheel distribution (`python -m build`)
  5. Upload artifacts
  6. Publish to PyPI using OIDC trusted publisher (`pypa/gh-action-pypi-publish@release/v1`)
- Environment: `release` (requires OIDC token permissions)
- No automated tests in CI (tests require Docker containers running locally)

**Docker Services:**
- Via `docker-compose.yml` at project root
- PostgreSQL 16 with pgvector extension (image: `pgvector/pgvector:pg16`)
  - Container: `memwright-postgres`
  - Ports: 5432
  - Health check: `pg_isready -U memwright -d memwright`
  - Volume: `pgdata` persistent volume
- Neo4j 5 Community (image: `neo4j:5-community`)
  - Container: `memwright-neo4j`
  - Ports: 7474 (HTTP), 7687 (Bolt)
  - Health check: `neo4j status` command
  - Volume: `neo4jdata` persistent volume
- Neo4j Test Instance
  - Container: `memwright-neo4j-test`
  - Ports: 7475 (HTTP), 7688 (Bolt)
  - Separate volume: `neo4jtestdata` for test isolation

## Environment Configuration

**Required Environment Variables:**
- At least one embedding API key:
  - `OPENROUTER_API_KEY` (preferred)
  - `OPENAI_API_KEY` (fallback)

**Optional Environment Variables:**
- `PG_CONNECTION_STRING` - PostgreSQL connection (default: `postgresql://memwright:memwright@localhost:5432/memwright`)
- `NEO4J_URI` - Neo4j Bolt URI (default: `bolt://localhost:7687`)
- `NEO4J_USER` - Neo4j username (default: `neo4j`)
- `NEO4J_PASSWORD` - Neo4j password (default: `memwright`)
- `NEO4J_DATABASE` - Neo4j database (default: `neo4j`)
- `ANTHROPIC_API_KEY` - Anthropic API (optional, for future use)

**Configuration File:**
- `.env` - Git-ignored environment variable file (created from `.env.example`)
- `config.json` - Per-memory-store configuration (auto-generated at `{store_path}/config.json`)

**Precedence:**
1. Environment variables (highest priority)
2. `config.json` file
3. Hardcoded defaults (lowest priority)

## Webhooks & Callbacks

**Incoming:**
- None - This is a library/CLI tool, not a web service

**Outgoing:**
- None - No external callbacks or webhook notifications

## MCP (Model Context Protocol) Integration

**MCP Server:**
- Implementation: `agent_memory/mcp/server.py`
- Server name: `"agent-memory"`
- Exposed tools (via `memory_add`, `memory_get`, `memory_search`, `memory_recall`, `memory_health`):
  - `memory_add` - Add/store a memory with content, tags, category, entity, event_date, confidence
  - `memory_get` - Retrieve a single memory by ID
  - `memory_search` - Search memories with filters (query, category, entity, status, limit)
  - `memory_recall` - Retrieve relevant memories for a natural language query with token budgeting
  - `memory_health` - Health check across all services (SQLite, PostgreSQL, Neo4j, embeddings)

**Execution:**
- Runs via stdio transport (MCP protocol)
- Compatible with Claude Code and Cursor
- Configuration: Added to Claude Code `config.json` via CLI: `agent-memory init {path} --tool claude-code`

## Benchmarks & Performance

**Benchmarks Integrated:**
- MemoryAgentBench (MAB) - `agent_memory/mab.py`
  - CLI: `agent-memory mab`
  - Metrics: Answer Recall (AR), Contradictions Recall (CR), Overall score
  - Latest scores: AR 55%, CR 62%, Overall 58.5%

- LOCOMO (Long Conversation Memory) - `agent_memory/locomo.py`
  - CLI: `agent-memory locomo`
  - Metrics: Multi-turn conversation memory retention
  - Uses LLM for memory extraction and entity graph traversal

## API Rate Limits

**OpenRouter/OpenAI Embeddings:**
- Rate limits vary by API tier and plan
- Caching mitigates repeated calls (256 embedding LRU cache)
- Batch embedding function `get_embeddings_batch()` chunks requests (50 texts per batch)

**PostgreSQL & Neo4j:**
- No explicit rate limiting (local Docker instances)
- Connection pooling via `psycopg-pool` for PostgreSQL

---

*Integration audit: 2026-03-15*
