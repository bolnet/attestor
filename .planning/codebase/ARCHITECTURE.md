# Architecture

**Analysis Date:** 2025-03-15

## Pattern Overview

**Overall:** Layered retrieval cascade with pluggable multi-backend storage

**Key Characteristics:**
- 3-layer retrieval pipeline: tag matching → graph expansion → vector similarity with Reciprocal Rank Fusion
- Temporal semantics built-in: contradictions detected automatically, memories can be superseded
- Non-fatal errors for optional services (pgvector, Neo4j): core SQLite path always works
- Immutable memory model: mutations handled by storing new memories and marking old ones superseded

## Layers

**Core Storage Layer (Always Present):**
- Purpose: Persistent memory storage with ACID guarantees
- Location: `agent_memory/store/sqlite_store.py`
- Contains: SQLiteStore class managing schema initialization, CRUD operations, raw queries
- Depends on: Python sqlite3 (stdlib)
- Used by: All other layers; foundation for everything

**Vector Search Layer (Optional):**
- Purpose: Semantic similarity search via pgvector PostgreSQL extension
- Location: `agent_memory/store/vector_store.py`
- Contains: VectorStore class wrapping psycopg client connection
- Depends on: PostgreSQL with pgvector extension, embeddings API (OpenAI/OpenRouter)
- Used by: RetrievalOrchestrator; failures are caught silently, non-fatal

**Graph Layer (Optional):**
- Purpose: Entity relationship traversal for multi-hop memory expansion
- Location: `agent_memory/graph/neo4j_graph.py`, `agent_memory/graph/extractor.py`
- Contains: Neo4jGraph class managing node/edge operations; entity/relation extraction patterns
- Depends on: Neo4j server, LLM for extraction (optional rule-based fallback)
- Used by: RetrievalOrchestrator for expanding query context via graph relations

**Temporal Layer:**
- Purpose: Timeline queries, contradiction detection, supersession logic
- Location: `agent_memory/temporal/manager.py`
- Contains: TemporalManager class handling check_contradictions, supersede, timeline, current_facts
- Depends on: SQLiteStore
- Used by: AgentMemory.add() for automatic contradiction detection

**Retrieval Orchestrator (Fusion Layer):**
- Purpose: Orchestrate all 3 retrieval channels and combine results
- Location: `agent_memory/retrieval/orchestrator.py`
- Contains: RetrievalOrchestrator class managing recall pipeline and Reciprocal Rank Fusion
- Depends on: SQLiteStore, VectorStore (optional), Neo4jGraph (optional), tag_matcher, scorer modules
- Used by: AgentMemory.recall() as primary recall interface

**Extraction Layer:**
- Purpose: Convert conversations to memories
- Location: `agent_memory/extraction/extractor.py`, `agent_memory/extraction/llm_extractor.py`, `agent_memory/extraction/rule_based.py`
- Contains: extract_memories() dispatcher; rule-based and LLM extraction strategies
- Depends on: OpenAI/OpenRouter API (optional for LLM mode, fallback to rule-based)
- Used by: AgentMemory.extract()

**MCP Layer:**
- Purpose: Expose AgentMemory as Claude Code / Claude Desktop tools
- Location: `agent_memory/mcp/server.py`
- Contains: MCP server factory with 8+ tools (memory_add, memory_recall, memory_search, etc.)
- Depends on: mcp>=1.0.0, AgentMemory
- Used by: Claude Code via MCP protocol

## Data Flow

**Write Flow (add):**

1. User calls `AgentMemory.add(content, tags, category, entity, ...)`
2. Create Memory object with generated UUID
3. Check contradictions via `TemporalManager.check_contradictions()` (rule-based: same entity+category+different content)
4. Insert into SQLite via `SQLiteStore.insert()` (ACID guaranteed)
5. If vector store available: compute embedding via OpenAI/OpenRouter, store in pgvector
6. If Neo4j available: extract entities/relations via pattern matching + optional LLM, add nodes/edges to graph
7. Return Memory object with assigned ID

**Recall Flow (recall):**

1. User calls `AgentMemory.recall(query, budget=2000)`
2. Pass query to `RetrievalOrchestrator.recall()`
3. **Layer 0 - Graph Expansion:** If graph available, extract tags from query, find related entities at depth=2, collect relationship triples
4. **Layer 1 - Tag Match:** Extract tags from query via simple keyword extraction, tag_search() in SQLite
5. **Layer 2 - Graph Expansion Results:** For related entities from step 3, list_memories by entity field
6. **Layer 3 - Vector Search:** Compute query embedding, search pgvector for top-20 cosine-similar results, look up full Memory from SQLite
7. **Synthetic Graph Relations:** Inject relationship triples as synthetic Memory objects (content=triple string)
8. **RRF Fusion:** Combine all sources using Reciprocal Rank Fusion (score = sum of 1/(k+rank) per source)
9. **Apply Boosts:** Temporal boost (recent memories score higher), entity boost (proper nouns in query)
10. **Fit to Budget:** Select highest-scored results until token count ≤ budget (minimum 1 result always)
11. Return List[RetrievalResult]

**State Management:**

Memories are immutable by design:
- New facts create new Memory records with unique IDs
- Contradictions are detected and old records marked superseded (status="superseded", valid_until timestamp, superseded_by pointer)
- Queries always filter for status="active" and valid_until IS NULL
- Timeline queries access full history including superseded records to show chronological updates
- Temporal layer maintains valid_from/valid_until windows for time-scoped facts

## Key Abstractions

**Memory:**
- Purpose: Core data structure representing a single fact/record
- Examples: `agent_memory/models.py`
- Pattern: Dataclass with 12 fields (id, content, tags, category, entity, temporal fields, confidence, status, metadata)
- Immutable: modifications create new Memory objects, old ones marked superseded

**RetrievalResult:**
- Purpose: Wrapper around Memory with metadata about how it was found
- Examples: `agent_memory/models.py`
- Pattern: Dataclass with (memory, score, match_source) where match_source ∈ {tag, graph, vector}
- Used: By retrieval orchestrator to track which layer found each result

**SQLiteStore:**
- Purpose: Encapsulate all SQLite operations
- Examples: `agent_memory/store/sqlite_store.py`
- Pattern: Repository pattern — public methods are insert, get, update, delete, list_memories, tag_search, execute
- Guarantees: WAL mode for concurrency, foreign keys enforced, JSON fields for extensibility

**RetrievalOrchestrator:**
- Purpose: Coordinate multi-layer retrieval and result fusion
- Examples: `agent_memory/retrieval/orchestrator.py`
- Pattern: Orchestrator pattern — receives multiple data sources (SQLite, vector store, graph), produces fused result
- Key Algorithm: Reciprocal Rank Fusion deduplicates cross-source results and combines scores

**TemporalManager:**
- Purpose: Handle time-based queries and contradiction detection
- Examples: `agent_memory/temporal/manager.py`
- Pattern: Encapsulates temporal logic separately from storage
- Algorithms: Rule-based contradiction (same entity+category+different text), linear decay temporal boost

## Entry Points

**API Entry Point:**
- Location: `agent_memory/core.py` - `AgentMemory` class
- Triggers: `mem = AgentMemory(path); mem.add(...); mem.recall(...)`
- Responsibilities: Manage initialization of all sub-layers, provide public API for write/read/search/timeline operations

**CLI Entry Point:**
- Location: `agent_memory/cli.py` - `main()` function
- Triggers: `agent-memory add|recall|search|list|timeline|stats|export|import|doctor|inspect|compact|forget`
- Responsibilities: Argument parsing, invoke AgentMemory methods, format output for terminal

**MCP Entry Point:**
- Location: `agent_memory/mcp/server.py` - `create_server(memory_path)` factory
- Triggers: Claude Code invokes tools via MCP protocol (stdio or HTTP)
- Responsibilities: Expose AgentMemory as 8+ named tools with typed schemas

**Health Check Entry Point:**
- Location: `agent_memory/core.py` - `AgentMemory.health()` method
- Triggers: `mem.health()` or MCP `memory_health` tool
- Responsibilities: Check all 8 components (Docker, containers, PostgreSQL, pgvector, Neo4j, SQLite, embeddings API, retrieval pipeline), return structured report

## Error Handling

**Strategy:** Fail safe — optional services are non-fatal, core path always works

**Patterns:**

1. **Import guards:** Try/except ImportError for optional dependencies (psycopg, neo4j, openai)
   - Location: `agent_memory/core.py` lines 48-94
   - Logs error, continues with service marked unavailable

2. **Service connection failures:** Try/except around service initialization and operations
   - Vector store failures: `core.py` lines 165-173 (caught silently during add)
   - Graph failures: `core.py` lines 176-191 (caught silently during add)
   - Example: Graph expansion in orchestrator `retrieval/orchestrator.py` lines 46-67 wrapped in try/except

3. **Validation at boundaries:** Input validation in public API
   - Memory model enforces non-empty content, tags are lists
   - Query strings can be None (handled gracefully)

4. **Rollback behavior:** None — SQLite transactions are immediate
   - add() inserts first, then marks contradictions superseded (if one fails, memory stays stored)

## Cross-Cutting Concerns

**Logging:**
- Using Python stdlib logging: `logger = logging.getLogger("agent_memory")`
- Location: Most modules import this logger
- Pattern: Log errors for service failures (vector, graph, embeddings), not debug logs in hot paths

**Validation:**
- Boundaries: Memory.add() validates tags list, content not empty
- Entity field validation: checked for contradiction detection
- Tag extraction: filters stop words and short tokens

**Authentication:**
- Vector store: PostgreSQL connection string from env PG_CONNECTION_STRING
- Graph: Neo4j password from env NEO4J_PASSWORD
- Embeddings: OpenAI API key from OPENAI_API_KEY or OpenRouter from OPENROUTER_API_KEY
- Config: `agent_memory/utils/config.py` loads from environment variables with defaults

**Configuration:**
- Centralized in `MemoryConfig` dataclass: `agent_memory/utils/config.py`
- Fields: pg_connection_string, neo4j_uri/user/password, openai_api_key, openrouter_api_key, default_token_budget, min_results
- Loaded from: .env file or environment variables, defaults to safe values
- Persisted: Saved to memory store path as config.json for reproducibility

---

*Architecture analysis: 2025-03-15*
