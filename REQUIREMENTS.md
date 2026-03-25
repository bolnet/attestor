# AgentMemory — Requirements Document

## Project Overview

**AgentMemory** is an embedded, zero-infrastructure, cross-session memory framework for AI agents. It uses SQLite + FTS5 for storage and retrieval with an optional FAISS vector index as a plugin. No vector database. No managed service. No API keys required for core functionality.

**Tagline:** "Embedded cross-session memory for AI agents. No servers. No API keys. Just works."

**Positioning:** The SQLite of AI memory. Where Mem0/Supermemory are managed services (like RDS), AgentMemory is a local embedded library (like SQLite).

---

## Core Design Principles

1. **Zero infrastructure** — `poetry add memwright` and a directory. No Docker, no server, no account signup.
2. **Data sovereignty** — Everything lives in a local SQLite file. User owns their data. `cp`, `rsync`, `scp` is the backup strategy.
3. **Graduated complexity** — Start with tags + FTS5 (zero dependencies). Optionally add FAISS for semantic search. Optionally add LLM extraction. Optionally expose via MCP.
4. **Sub-5ms retrieval** — Local disk operations only for core retrieval path. No network calls.
5. **Token efficient** — Retrieve only relevant memories (300-500 tokens) instead of full history (15,000+ tokens). Target 90%+ token reduction vs full conversation history replay.
6. **Inspectable and debuggable** — Users can `sqlite3 memory.db` and see exactly what the agent remembers. No black box.
7. **No vector database** — Use embedded vector index (FAISS file) as optional plugin, not a running service. Core retrieval is tag match + BM25 full-text search.

---

## Target Users

- Solo developers building AI agents
- Edge/embedded agents (Raspberry Pi, local-first apps, offline assistants)
- Regulated industries (healthcare, finance, legal) where data can't leave infra
- Claude Code / CLI agent builders wanting MCP memory without cloud dependency
- Agent framework authors (LangGraph, CrewAI, AutoGen) wanting a drop-in memory layer

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    AgentMemory                        │
│                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────┐ │
│  │ Memory      │  │ Retrieval   │  │ Extraction   │ │
│  │ Store       │  │ Orchestrator│  │ Pipeline     │ │
│  │             │  │             │  │              │ │
│  │ SQLite+FTS5 │  │ 3-Layer:    │  │ LLM-based    │ │
│  │ (required)  │  │ 1. Tag match│  │ (optional)   │ │
│  │             │  │ 2. BM25/FTS │  │              │ │
│  │ FAISS file  │  │ 3. Vector*  │  │ Rule-based   │ │
│  │ (optional)  │  │    (*opt)   │  │ (default)    │ │
│  └──────┬──────┘  └──────┬──────┘  └──────┬───────┘ │
│         │                │                │          │
│         └────────────────┼────────────────┘          │
│                          │                           │
│                ┌─────────▼─────────┐                 │
│                │ Context Assembler │                  │
│                │                   │                  │
│                │ Token budgeting   │                  │
│                │ Priority ranking  │                  │
│                │ Deduplication     │                  │
│                └─────────┬─────────┘                  │
│                          │                           │
│         ┌────────────────┼────────────────┐          │
│         │                │                │          │
│    ┌────▼────┐    ┌──────▼──────┐  ┌──────▼───────┐ │
│    │   CLI   │    │ Python API  │  │  MCP Server  │ │
│    │         │    │             │  │  (optional)  │ │
│    └─────────┘    └─────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────┘
```

### Storage Layer

- **Primary:** SQLite database with FTS5 virtual table for full-text search
- **Optional:** FAISS flat index file (`.faiss`) for vector similarity — installed only if user opts in via `poetry add memwright -E vectors`
- **No vector database.** FAISS index is a file on disk, not a running process.

### File Layout on Disk

```
~/.agent-memory/my-agent/        # Or any user-specified path
├── memory.db                    # SQLite database (memories + FTS5 index)
├── memory.faiss                 # Optional FAISS index file
└── config.json                  # Agent-level config
```

---

## Data Model

### Memory (core entity)

```python
@dataclass
class Memory:
    id: str                          # uuid hex[:12]
    content: str                     # Atomic fact, e.g. "User works at SoFi as Staff SWE AI"

    # Classification
    tags: List[str]                  # ["career", "sofi", "compensation"]
    category: str                    # "career", "project", "preference", "general"
    entity: Optional[str]            # "SoFi", "TD Bank", etc.

    # Temporal
    created_at: str                  # ISO datetime — when stored
    event_date: Optional[str]        # ISO datetime — when event actually happened
    valid_from: str                  # ISO datetime — when this fact became true
    valid_until: Optional[str]       # ISO datetime — when superseded (None = current)
    superseded_by: Optional[str]     # ID of memory that replaced this one

    # Vector (optional)
    embedding: Optional[List[float]] # Pre-computed at write time, only if FAISS enabled

    # Provenance
    source_session: Optional[str]    # Session ID this came from
    confidence: float                # 0.0 - 1.0 extraction confidence

    # Status
    status: str                      # "active", "superseded", "expired", "archived"

    # Extensible
    metadata: Dict[str, Any]         # User-defined key-value pairs
```

### SQLite Schema

```sql
CREATE TABLE memories (
    id TEXT PRIMARY KEY,
    content TEXT NOT NULL,
    tags TEXT NOT NULL,              -- JSON array stored as text
    category TEXT NOT NULL DEFAULT 'general',
    entity TEXT,
    created_at TEXT NOT NULL,
    event_date TEXT,
    valid_from TEXT NOT NULL,
    valid_until TEXT,
    superseded_by TEXT,
    source_session TEXT,
    confidence REAL DEFAULT 1.0,
    status TEXT DEFAULT 'active',
    metadata TEXT DEFAULT '{}',       -- JSON object
    FOREIGN KEY (superseded_by) REFERENCES memories(id)
);

-- FTS5 virtual table for full-text search
CREATE VIRTUAL TABLE memories_fts USING fts5(
    content,
    tags,
    category,
    entity,
    content='memories',
    content_rowid='rowid'
);

-- Indexes
CREATE INDEX idx_memories_status ON memories(status);
CREATE INDEX idx_memories_category ON memories(category);
CREATE INDEX idx_memories_entity ON memories(entity);
CREATE INDEX idx_memories_valid ON memories(valid_until);
CREATE INDEX idx_memories_created ON memories(created_at);
CREATE INDEX idx_memories_session ON memories(source_session);

-- Triggers to keep FTS5 in sync
CREATE TRIGGER memories_ai AFTER INSERT ON memories BEGIN
    INSERT INTO memories_fts(rowid, content, tags, category, entity)
    VALUES (new.rowid, new.content, new.tags, new.category, new.entity);
END;

CREATE TRIGGER memories_ad AFTER DELETE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, category, entity)
    VALUES ('delete', old.rowid, old.content, old.tags, old.category, old.entity);
END;

CREATE TRIGGER memories_au AFTER UPDATE ON memories BEGIN
    INSERT INTO memories_fts(memories_fts, rowid, content, tags, category, entity)
    VALUES ('delete', old.rowid, old.content, old.tags, old.category, old.entity);
    INSERT INTO memories_fts(rowid, content, tags, category, entity)
    VALUES (new.rowid, new.content, new.tags, new.category, new.entity);
END;

-- Sessions table
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    summary TEXT,
    metadata TEXT DEFAULT '{}'
);
```

---

## Retrieval Strategy — 3-Layer Cascade

Retrieval uses a layered cascade. Each layer is tried in order. Results are merged, deduplicated, scored, and assembled within a token budget.

### Layer 1: Tag Match (0ms, highest precision)

- Extract tags/categories from query (keyword extraction, no LLM needed)
- SQL: `SELECT * FROM memories WHERE status='active' AND valid_until IS NULL AND (tags LIKE '%career%' OR category='career')`
- Score weight: 1.0

### Layer 2: BM25 Full-Text Search via FTS5 (1-2ms, high precision)

- Use SQLite FTS5 `MATCH` with BM25 ranking
- SQL: `SELECT *, bm25(memories_fts) as rank FROM memories_fts WHERE memories_fts MATCH 'python stack' ORDER BY rank`
- Join back to memories table, filter by `status='active'` and `valid_until IS NULL`
- Score weight: 0.8

### Layer 3: Vector Similarity via FAISS (optional, 5-10ms, semantic catch-all)

- Only runs if FAISS is installed AND Layer 1+2 returned fewer results than threshold (default: 3)
- Compute embedding for query using local model or API
- Search FAISS flat index, return top-k (default: 10)
- Score weight: 0.6
- **This layer is entirely optional.** Framework works fully without it.

### Scoring and Assembly

```python
def retrieve(query, token_budget=2000):
    results = []
    results += tag_match(query, weight=1.0)
    results += fts_search(query, weight=0.8)
    if len(results) < min_results and faiss_available:
        results += vector_search(query, weight=0.6)

    ranked = deduplicate(results)
    ranked = temporal_boost(ranked)    # Recent memories score higher
    ranked = entity_boost(ranked, query)  # Mentioned entities score higher
    return fit_to_budget(ranked, token_budget)
```

### Token Budget Assembly

- Estimate tokens per memory: `len(content.split()) * 1.3`
- Fill from highest-scored down until budget exhausted
- Return assembled context string ready for prompt injection

---

## Memory Write Pipeline

### Manual Add

```python
mem.add("User prefers Python over Java", tags=["preference", "coding"], category="preference")
```

### Conversation Extraction (optional, requires LLM)

```python
# Extract memories from a conversation
memories = mem.extract(conversation_messages)
# Returns list of Memory objects with auto-generated tags, categories, entities
```

### Contradiction Resolution

On every `add()`:

1. Search existing active memories with overlapping tags/category/entity
2. For each candidate, check if new memory contradicts it:
   - **Rule-based (default):** Same entity + same category + different content → flag as potential contradiction
   - **LLM-based (optional):** Ask LLM "Do these two facts contradict each other?"
3. If contradiction detected: supersede old memory (set `valid_until`, `superseded_by`, `status='superseded'`)
4. Store new memory as active

### Session Lifecycle Hooks

```python
@mem.on_session_start
def load_context(session_id, first_message):
    return mem.recall(first_message, budget=3000)

@mem.on_session_end
def persist(session_id, conversation_history):
    new_memories = mem.extract(conversation_history)
    for m in new_memories:
        mem.add(m)
```

---

## Python API

### Core Interface

```python
from agent_memory import AgentMemory

# Initialize — just a path
mem = AgentMemory("./my-agent")
# Or with config
mem = AgentMemory("./my-agent", config={
    "default_token_budget": 2000,
    "min_results": 3,
    "enable_vectors": False,      # True requires faiss-cpu
})

# ── Write ──
mem.add(
    content="User accepted Staff SWE AI role at SoFi, ~$257K base",
    tags=["career", "sofi", "compensation"],
    category="career",
    entity="SoFi",
    event_date="2025-01-15T00:00:00Z"
)

# ── Read ──
results = mem.recall("what's the user's current role?", budget=1000)
# Returns: List[RetrievalResult] with .content, .score, .match_source

# ── Context string for prompt injection ──
context = mem.recall_as_context("what's the user's current role?", budget=1000)
# Returns: formatted string ready to inject into system prompt

# ── Search with filters ──
results = mem.search(
    query="SoFi",
    category="career",
    status="active",
    after="2025-01-01",
    limit=10
)

# ── Timeline ──
history = mem.timeline(entity="SoFi")
# Returns all memories about SoFi ordered by event_date, including superseded ones

# ── Temporal query ──
current = mem.current_facts(category="career")
# Returns only active, non-superseded memories in category

# ── Session management ──
session = mem.start_session()
# ... agent runs ...
mem.end_session(session.id, conversation_history=messages)

# ── Extraction (optional, needs LLM) ──
memories = mem.extract(conversation_messages, model="claude-haiku")

# ── Maintenance ──
mem.forget(memory_id)           # Archive a specific memory
mem.forget_before("2024-01-01") # Archive old memories
mem.compact()                   # Remove archived memories permanently
stats = mem.stats()             # Count by status, category, storage size

# ── Export / Import ──
mem.export_json("backup.json")
mem.import_json("backup.json")

# ── Direct SQL escape hatch ──
rows = mem.execute("SELECT * FROM memories WHERE entity = ?", ["SoFi"])
```

---

## CLI

```bash
# Initialize a new memory store
agent-memory init ./my-agent

# Add a memory
agent-memory add ./my-agent "User prefers Python" --tags preference,coding --category preference

# Recall
agent-memory recall ./my-agent "what language does the user prefer?"

# Search
agent-memory search ./my-agent "Python" --category preference --status active

# List all active memories
agent-memory list ./my-agent --status active

# Timeline for an entity
agent-memory timeline ./my-agent --entity SoFi

# Stats
agent-memory stats ./my-agent

# Export / Import
agent-memory export ./my-agent --format json > backup.json
agent-memory import ./my-agent < backup.json

# Inspect raw DB
agent-memory inspect ./my-agent

# Start MCP server (optional)
agent-memory serve ./my-agent --mcp --port 8080

# Compact (remove archived memories)
agent-memory compact ./my-agent
```

---

## MCP Server (Optional Feature)

Expose the memory store as an MCP server for Claude Code / Claude Desktop integration.

### MCP Tools to Expose

```
memory_add       — Store a new memory
memory_recall    — Retrieve relevant memories for a query
memory_search    — Search with filters
memory_forget    — Archive a memory
memory_timeline  — Get entity timeline
memory_stats     — Get store statistics
```

### Config for Claude Code

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "agent-memory",
      "args": ["serve", "./my-agent", "--mcp"]
    }
  }
}
```

---

## Package Structure

```
agent-memory/
├── pyproject.toml
├── README.md
├── LICENSE                         # Apache 2.0
├── REQUIREMENTS.md                 # This file
├── agent_memory/
│   ├── __init__.py                 # Public API exports
│   ├── core.py                     # AgentMemory main class, Memory dataclass, config
│   ├── store/
│   │   ├── __init__.py
│   │   ├── sqlite_store.py         # SQLite + FTS5 implementation
│   │   └── schema.sql              # DDL for tables, indexes, triggers
│   ├── retrieval/
│   │   ├── __init__.py
│   │   ├── orchestrator.py         # 3-layer cascade logic
│   │   ├── tag_matcher.py          # Layer 1: tag extraction + matching
│   │   ├── fts_searcher.py         # Layer 2: FTS5 BM25 search
│   │   ├── vector_searcher.py      # Layer 3: optional FAISS search
│   │   └── scorer.py               # Scoring, boosting, dedup, token budget
│   ├── extraction/
│   │   ├── __init__.py
│   │   ├── extractor.py            # Memory extraction from conversations
│   │   ├── rule_based.py           # Default rule-based extraction (no LLM)
│   │   ├── llm_extractor.py        # Optional LLM-based extraction
│   │   └── contradiction.py        # Contradiction detection + resolution
│   ├── temporal/
│   │   ├── __init__.py
│   │   └── manager.py              # Timeline queries, supersession logic, expiry
│   ├── mcp/
│   │   ├── __init__.py
│   │   └── server.py               # MCP server implementation
│   ├── cli.py                      # CLI entry point (click or argparse)
│   └── utils/
│       ├── __init__.py
│       ├── tokens.py               # Token estimation
│       └── config.py               # Config loading/defaults
├── tests/
│   ├── test_core.py
│   ├── test_store.py
│   ├── test_retrieval.py
│   ├── test_extraction.py
│   ├── test_temporal.py
│   ├── test_cli.py
│   └── test_mcp.py
└── examples/
    ├── basic_usage.py
    ├── chatbot_with_memory.py
    ├── claude_code_integration.py
    └── langraph_integration.py
```

---

## Dependencies

### Core (zero external dependencies)

- `sqlite3` — Built into Python standard library
- `json` — Built into Python standard library
- `uuid`, `datetime`, `dataclasses` — Built into Python standard library

### Optional Extras

```toml
[project.optional-dependencies]
vectors = ["faiss-cpu>=1.7.0"]
extraction = ["anthropic>=0.30.0"]    # Or openai, or any LLM SDK
mcp = ["mcp>=1.0.0"]
all = ["faiss-cpu>=1.7.0", "anthropic>=0.30.0", "mcp>=1.0.0"]
```

Install variants:

```bash
poetry add memwright                  # Core only — SQLite + ChromaDB + NetworkX
poetry add memwright -E extraction    # + LLM-based memory extraction
poetry add memwright -E all           # Everything
```

---

## Performance Targets

| Operation | Target | Method |
|-----------|--------|--------|
| Tag match retrieval | < 1ms | SQL index lookup |
| FTS5 BM25 search | < 3ms | SQLite FTS5 |
| Vector search (if enabled) | < 10ms | FAISS flat index |
| Full recall pipeline | < 5ms (no vectors), < 15ms (with vectors) | Cascade |
| Memory add | < 5ms (no vectors), < 50ms (with embedding) | Insert + index update |
| Token budget per recall | 300-500 tokens typical | Scoring + assembly |
| Memory store capacity | 100K+ memories per agent | SQLite handles this |

---

## Token Savings Model

```
Approach                    Tokens/call   Cost/call*   Annual (1K/day)*
────────────────────────────────────────────────────────────────────────
Full conversation history   15,000        $0.045       $16,425
Static CLAUDE.md            2,000         $0.006       $2,190
Managed memory (Mem0 etc)   800           $0.0024      $1,470**
AgentMemory (this)          300-500       $0.0015      $547

* Based on Claude Sonnet at $3/M input tokens
** Includes platform fee
```

---

## Build Order

Build and test in this order. Each phase should be independently usable.

### Phase 1 — Core Store + Manual CRUD

- `Memory` dataclass
- `SQLiteStore` with schema creation, insert, update, delete, get
- FTS5 table + sync triggers
- Basic `AgentMemory` class wrapping the store
- `mem.add()`, `mem.get()`, `mem.forget()`, `mem.search()` with SQL filters
- Export/import JSON
- Unit tests for all CRUD operations

### Phase 2 — Retrieval Orchestrator

- Tag extraction from queries (keyword-based, no LLM)
- Layer 1: Tag matcher
- Layer 2: FTS5 BM25 searcher
- Scoring, deduplication, temporal boost, entity boost
- Token budget assembly
- `mem.recall()` and `mem.recall_as_context()`
- Unit tests with various query types

### Phase 3 — Temporal Logic

- Contradiction detection (rule-based: same entity + category)
- Supersession on add (auto-close old memories)
- `mem.timeline()` and `mem.current_facts()`
- Expiry logic (memories with explicit valid_until)
- Unit tests for contradiction scenarios

### Phase 4 — CLI

- All commands listed in CLI section above
- Uses `click` or `argparse`
- Entry point in pyproject.toml: `agent-memory = "agent_memory.cli:main"`

### Phase 5 — Optional Vector Layer (FAISS plugin)

- Graceful import: if `faiss` not installed, Layer 3 is skipped silently
- Embedding computation: support local sentence-transformers or API-based
- FAISS flat index: create, add, search, save/load from file
- Integration into retrieval orchestrator as Layer 3
- Tests with and without FAISS installed

### Phase 6 — Optional LLM Extraction

- `mem.extract(conversation_messages)` using LLM
- Prompt engineering for atomic fact extraction with tags, category, entity, event_date
- Rule-based fallback extractor (regex patterns for names, dates, preferences)
- Contradiction detection via LLM (optional upgrade over rule-based)
- Session hooks: `on_session_start`, `on_session_end`

### Phase 7 — MCP Server

- Implement MCP protocol for listed tools
- `agent-memory serve` command
- Config generation for Claude Code / Claude Desktop
- Integration test with MCP client

### Phase 8 — README, Examples, Packaging

- README with badges, install instructions, quick start, comparison table, cost savings
- Example scripts for common patterns
- pyproject.toml with all metadata, entry points, optional deps
- GitHub Actions CI (test on Python 3.9+)
- PyPI publish workflow

---

## Testing Strategy

- **Unit tests** for every module (store, retrieval layers, scoring, temporal, extraction)
- **Integration tests** for full `recall()` pipeline end-to-end
- **Performance benchmarks** for retrieval latency targets
- **With/without optional deps** — ensure core works without faiss, anthropic, mcp
- **Edge cases:** empty store, single memory, 10K+ memories, unicode content, long content, overlapping tags, rapid supersession chains
- Use `pytest` with fixtures for pre-populated stores

---

## Out of Scope for v0.1

- Multi-user / multi-tenant support
- Web dashboard / UI
- Distributed / networked memory
- Automatic memory consolidation / summarization
- Graph-based relationship traversal
- Streaming / real-time memory updates
- Authentication / access control on MCP server

These can be added in future versions without breaking the core API.
