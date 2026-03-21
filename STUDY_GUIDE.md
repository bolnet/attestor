# Memwright — Detailed Study Guide

> Embedded memory for AI agents. SQLite + ChromaDB + NetworkX. Zero-config.

**PyPI**: `memwright` | **Import**: `agent_memory` | **Version**: 2.0.0 | **License**: Apache-2.0 | **Python**: >=3.9

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Data Models](#2-data-models)
3. [Storage Layer — SQLite](#3-storage-layer--sqlite)
4. [Vector Store — ChromaDB](#4-vector-store--chromadb)
5. [Entity Graph — NetworkX](#5-entity-graph--networkx)
6. [Entity & Relation Extraction](#6-entity--relation-extraction)
7. [Temporal Logic](#7-temporal-logic)
8. [Retrieval Pipeline](#8-retrieval-pipeline)
9. [Scoring & Ranking](#9-scoring--ranking)
10. [Core API — AgentMemory](#10-core-api--agentmemory)
11. [Memory Extraction](#11-memory-extraction)
12. [MCP Server](#12-mcp-server)
13. [Lifecycle Hooks](#13-lifecycle-hooks)
14. [CLI Reference](#14-cli-reference)
15. [Configuration](#15-configuration)
16. [Data Flow Diagrams](#16-data-flow-diagrams)
17. [Design Patterns & Principles](#17-design-patterns--principles)
18. [Benchmarks](#18-benchmarks)
19. [File Inventory](#19-file-inventory)

---

## 1. Architecture Overview

Memwright is a **fully embedded** memory system — no external services, no Docker, no API keys. Three backends work together:

```
┌─────────────────────────────────────────────────┐
│                  AgentMemory API                │
│                   (core.py)                     │
├──────────┬──────────────┬───────────────────────┤
│  SQLite  │   ChromaDB   │      NetworkX         │
│  Store   │  Vector Store│    Entity Graph        │
│ (always) │  (non-fatal) │    (non-fatal)         │
├──────────┴──────────────┴───────────────────────┤
│          Retrieval Orchestrator                  │
│   Tag Match → Graph Expansion → Vector Search   │
│              + RRF Fusion + Boosts              │
├─────────────────────────────────────────────────┤
│          Temporal Manager                       │
│   Contradiction Detection + Supersession        │
└─────────────────────────────────────────────────┘
```

**Key principle**: SQLite is the only required backend. ChromaDB and NetworkX failures are caught silently — the system degrades gracefully.

**Initialization**: `AgentMemory("./path")` auto-provisions everything:
- `./path/memory.db` — SQLite database
- `./path/chroma/` — ChromaDB persistent store
- `./path/graph.json` — NetworkX graph serialization
- `./path/config.json` — Runtime configuration

---

## 2. Data Models

> `agent_memory/models.py`

### Memory

The atomic unit of stored knowledge.

```python
@dataclass
class Memory:
    id: str                          # uuid4().hex[:12], auto-generated
    content: str                     # The fact text
    tags: List[str]                  # Search labels
    category: str                    # "career","project","preference","personal","technical","general"
    entity: Optional[str]            # Primary subject (person, tool, company)
    created_at: str                  # UTC ISO timestamp, auto-set
    event_date: Optional[str]        # When the fact occurred (timeline ordering)
    valid_from: str                  # When this fact became active
    valid_until: Optional[str]       # Set when superseded
    superseded_by: Optional[str]     # FK → another Memory.id
    embedding: Optional[List[float]] # Transient, not stored in SQLite
    confidence: float                # 0.0–1.0 (default 1.0)
    status: str                      # "active" | "superseded" | "archived"
    metadata: Dict[str, Any]         # Open extensible bag
```

**Key methods**:
- `tags_json()` / `metadata_json()` — JSON serialization for SQLite storage
- `Memory.from_row(row)` — Reconstruct from SQLite row dict
- `to_dict()` — Plain dict for JSON export

**Status lifecycle**: `active` → `superseded` (via contradiction detection) or `archived` (via `forget()`)

### RetrievalResult

A scored wrapper around Memory, returned by the retrieval pipeline.

```python
@dataclass
class RetrievalResult:
    memory: Memory
    score: float        # Combined score (RRF + boosts)
    match_source: str   # "tag" | "graph" | "vector"
```

---

## 3. Storage Layer — SQLite

> `agent_memory/store/sqlite_store.py` + `agent_memory/store/schema.sql`

### Schema

Single table `memories` with 13 columns:

| Column | Type | Purpose |
|--------|------|---------|
| `id` | TEXT PK | 12-char hex UUID |
| `content` | TEXT | The fact |
| `tags` | TEXT | JSON array string |
| `category` | TEXT | Classification |
| `entity` | TEXT | Primary subject |
| `created_at` | TEXT | UTC ISO timestamp |
| `event_date` | TEXT | When fact occurred |
| `valid_from` | TEXT | Validity start |
| `valid_until` | TEXT | Set on supersession |
| `superseded_by` | TEXT | FK to replacement memory |
| `confidence` | REAL | 0.0–1.0 |
| `status` | TEXT | active/superseded/archived |
| `metadata` | TEXT | JSON object |

**Indexes**: `status`, `category`, `entity`, `valid_until`, `created_at`

**Pragmas**: WAL mode enabled, foreign keys on.

### SQLiteStore API

```python
class SQLiteStore:
    def __init__(self, db_path)           # Creates schema if needed
    def insert(self, memory) -> Memory    # INSERT + immediate commit
    def get(self, memory_id) -> Memory    # By primary key
    def update(self, memory) -> Memory    # Full row UPDATE + commit
    def delete(self, memory_id) -> bool   # Hard DELETE
    def list_memories(self, status=None, category=None, entity=None,
                      after=None, before=None, limit=100) -> List[Memory]
    def tag_search(self, tags, category=None, limit=20) -> List[Memory]
    def stats(self) -> Dict               # Counts + db_size_bytes
    def archive_before(self, date) -> int # Bulk soft-delete
    def compact(self) -> int              # Hard-delete archived + VACUUM
    def execute(self, sql, params)        # Raw SQL escape hatch
```

**Design notes**:
- `tag_search` uses `LIKE %tag%` on the JSON string — simple substring matching
- Every write commits immediately (no batching)
- `sqlite3.Row` used as row_factory for named column access

---

## 4. Vector Store — ChromaDB

> `agent_memory/store/chroma_store.py`

### How It Works

ChromaDB provides **semantic similarity search** using vector embeddings. Content is embedded into 384-dimensional vectors using the `all-MiniLM-L6-v2` sentence-transformer model (runs locally, no API key).

```python
class ChromaStore:
    def __init__(self, store_path, embedding_function=None)
    def add(self, memory_id, content)          # Upsert into collection
    def search(self, query_text, limit=20)     # Returns [{memory_id, content, distance}]
    def delete(self, memory_id)
    def count(self) -> int
```

### Key Details

- **Persistence**: `chromadb.PersistentClient` at `<store_path>/chroma/`
- **Similarity metric**: Cosine distance (`hnsw:space: cosine`)
- **Default model**: `all-MiniLM-L6-v2` (384D, sentence-transformers)
- **Custom embeddings**: Injectable via constructor (used for benchmarks with OpenAI)
- **Safety**: Clamps `n_results` to `collection.count()` to avoid ChromaDB errors on small stores
- **Distance → score**: `score = max(0.0, 1.0 - distance)` (done in orchestrator)

---

## 5. Entity Graph — NetworkX

> `agent_memory/graph/networkx_graph.py`

### How It Works

An in-process directed multigraph tracks **entities** (nodes) and **relationships** (edges). Enables multi-hop reasoning: "Who does Alice work with?" → Alice → CompanyX → Bob.

```python
class NetworkXGraph:
    def __init__(self, store_path)    # Loads graph.json if exists

    # Write
    def add_entity(self, name, entity_type="general", attributes=None)
    def add_relation(self, from_entity, to_entity, relation_type="related_to", metadata=None)

    # Read
    def get_related(self, entity, depth=2) -> List[str]    # BFS traversal
    def get_subgraph(self, entity, depth=2) -> Dict         # {entity, nodes, edges}
    def get_entities(self, entity_type=None) -> List[Dict]
    def get_edges(self, entity) -> List[Dict]               # SPO triples

    # Ops
    def stats(self) -> Dict     # {nodes, edges, types}
    def save(self) / load()     # JSON persistence via nx.node_link_data
    def close()                 # Saves before close
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| `nx.MultiDiGraph` | Directed + multiple edges between same pair |
| Keys lowercased, `display_name` preserved | Case-insensitive lookup, pretty display |
| Idempotent `add_entity` | Merges attributes, only promotes type (never downgrades) |
| `add_relation` auto-creates nodes | No ordering dependency |
| Relation types → `UPPER_SNAKE_CASE` | Consistent edge labeling |
| BFS both directions | Predecessors + successors for full connectivity |
| Eager save after every write | No data loss on crash |
| `nx.node_link_data` serialization | Standard NetworkX JSON format |

---

## 6. Entity & Relation Extraction

> `agent_memory/graph/extractor.py`

### Rule-Based Graph Extraction

```python
def extract_entities_and_relations(content, tags, entity, category)
    -> (List[Dict], List[Dict])  # (nodes, edges)
```

Three extraction sources:

1. **Primary entity field** — mapped to type via category (e.g., `"career"` → `"organization"`)
2. **Tags** — proper nouns (uppercase first char) or known tools become nodes; edges link to primary entity
3. **Content patterns** — regex patterns:
   - `"works at X"` → `WORKS_AT` relation
   - `"uses X"` → `USES` relation
   - `"lives in X"` → `LIVES_IN` relation
   - etc.

**Known tools set**: ~50 developer tools (python, react, docker, postgresql...) used for automatic `"tool"` type detection.

---

## 7. Temporal Logic

> `agent_memory/temporal/manager.py`

### Contradiction Detection & Supersession

The temporal manager ensures facts stay current by detecting when a new memory contradicts an existing one.

```python
class TemporalManager:
    def timeline(self, entity) -> List[Memory]
        # Chronological: ORDER BY COALESCE(event_date, created_at) ASC

    def current_facts(self, category=None, entity=None) -> List[Memory]
        # WHERE status='active' AND valid_until IS NULL

    def check_contradictions(self, new_memory) -> List[Memory]
        # Same entity + same category + different content + active + no valid_until

    def supersede(self, old_memory, new_memory_id) -> Memory
        # old.status='superseded', old.valid_until=now, old.superseded_by=new_id
```

### Rules

- Contradiction detection only fires when `new_memory.entity` is set
- Memories without entities never supersede each other
- Supersession is one-directional: old → new via `superseded_by` FK
- `valid_until` is set on supersession, serves as independent filter
- Content is **never mutated** — new memory written first, then old marked superseded

---

## 8. Retrieval Pipeline

> `agent_memory/retrieval/orchestrator.py`

### 3-Layer Cascade with RRF Fusion

The retrieval pipeline runs multiple search strategies and fuses results using Reciprocal Rank Fusion.

```python
class RetrievalOrchestrator:
    def recall(self, query, token_budget=2000) -> List[RetrievalResult]
    def recall_as_context(self, query, token_budget=2000) -> str
```

### Layer-by-Layer Breakdown

#### Layer 0 — Graph Expansion (Pre-processing)

```
query → extract_tags() → [keywords]
for each keyword:
    graph.get_related(keyword, depth=2) → related entity names
    graph.get_edges(keyword) → SPO triples
→ expanded_queries (entity names), graph_context_triples (SPO strings)
```

#### Layer 1 — Tag Match (Score: 1.0)

```
extract_tags(query) → [keywords]
sqlite_store.tag_search(keywords) → Memory list
→ RetrievalResult(score=1.0, source="tag")
```

#### Layer 2 — Graph Entity Search (Score: 0.5)

```
for entity in expanded_queries[:8]:
    sqlite_store.list_memories(entity=entity)
→ RetrievalResult(score=0.5, source="graph")
```

Only adds memories not already found in Layer 1.

#### Layer 3 — Vector Similarity (Score: 1.0 - distance)

```
chroma_store.search(query, limit=20)
→ for each new ID: sqlite_store.get(id)
→ RetrievalResult(score=1.0-distance, source="vector")
```

Only adds memories not already found in Layers 1-2.

#### Layer 4 — Triple Injection (Score: 0.6)

```
graph_context_triples → synthetic Memory objects
→ RetrievalResult(score=0.6, source="graph", category="graph_relation")
```

Injects structured relationship data (up to 20 triples) for the LLM to reason over.

### Post-Processing Pipeline

```
results → RRF fusion → temporal boost → entity boost → budget fitting
```

---

## 9. Scoring & Ranking

> `agent_memory/retrieval/scorer.py`

### Reciprocal Rank Fusion (RRF)

When results come from multiple sources:

```
score(memory) = Σ_sources  1 / (k + rank_in_source)
```

Where `k = 60` (standard RRF constant). Memories appearing in multiple sources get additive boosts.

If only a single source returns results, RRF just deduplicates.

### Temporal Boost

Linear time decay:

```
boost = max(0, 1 - age_days / 90) × 0.2
```

- Max boost: +0.2 (for brand-new memories)
- Decays to 0 over 90 days
- Can be disabled for benchmarks spanning long time periods

### Entity Boost

Query-entity matching:

```
+0.30  if memory.entity matches a proper noun in the query (exact match)
+0.15  if an entity name appears in the memory content (min 3 chars)
```

### Budget Fitting

```python
def fit_to_budget(results, token_budget):
    # Sort by score DESC
    # Greedily add results while within budget
    # Always includes at least 1 result even if over budget
```

Token estimation: `tokens ≈ word_count × 1.3`

---

## 10. Core API — AgentMemory

> `agent_memory/core.py`

### Constructor

```python
class AgentMemory:
    def __init__(self, path: str | Path, config: Optional[Dict] = None)
```

**Initialization sequence**:
1. `mkdir -p path`
2. Load/save `config.json`
3. Open `SQLiteStore(path/memory.db)` — always
4. Try `ChromaStore(path)` — non-fatal
5. Try `NetworkXGraph(path)` — non-fatal
6. Create `TemporalManager` and `RetrievalOrchestrator`

Supports context manager: `with AgentMemory(path) as mem:`

### Write Operations

```python
def add(self, content, tags=None, category="general", entity=None,
        event_date=None, confidence=1.0, metadata=None) -> Memory
def forget(self, memory_id) -> bool         # Sets status="archived"
def forget_before(self, date) -> int        # Bulk archive by date
def compact(self) -> int                    # Hard-delete archived + VACUUM
```

### Read Operations

```python
def get(self, memory_id) -> Optional[Memory]
def recall(self, query, budget=None) -> List[RetrievalResult]       # Full pipeline
def recall_as_context(self, query, budget=None) -> str              # Formatted string
def search(self, query=None, category=None, entity=None,
           status="active", after=None, before=None, limit=10) -> List[Memory]
```

**`search` vs `recall`**:
- `search()` — filter-oriented, returns raw `Memory` objects
- `recall()` — relevance-oriented, runs full 3-layer cascade, returns scored `RetrievalResult`

### Temporal Operations

```python
def timeline(self, entity) -> List[Memory]
def current_facts(self, category=None, entity=None) -> List[Memory]
```

### Extraction

```python
def extract(self, messages, model="claude-haiku", use_llm=False) -> List[Memory]
```

### Import/Export & Ops

```python
def export_json(self, filepath) -> None
def import_json(self, filepath) -> int
def batch_embed(self, batch_size=100) -> int   # Re-index into ChromaDB
def stats(self) -> Dict
def health(self) -> Dict                        # Checks all 4 components
def execute(self, sql, params=None) -> List     # Raw SQL
```

### Health Check Output

```python
{
    "healthy": True,
    "checks": [
        {"name": "SQLite", "status": "ok", "memory_count": 42, "latency_ms": 1.2},
        {"name": "ChromaDB", "status": "ok", "vector_count": 42, "embedding_provider": "local"},
        {"name": "NetworkX Graph", "status": "ok", "nodes": 15, "edges": 23},
        {"name": "Retrieval Pipeline", "status": "ok", "active_layers": 3, "max_layers": 3}
    ]
}
```

---

## 11. Memory Extraction

> `agent_memory/extraction/`

### Dispatcher (`extractor.py`)

```python
def extract_memories(messages, use_llm=False, model="openai/gpt-4.1-mini") -> List[Memory]
def extract_from_session(turns, speaker_a, speaker_b, session_date, model, api_key)
    -> (List[Memory], List[Dict])
```

Routes to either rule-based or LLM extraction.

### Rule-Based Extraction (`rule_based.py`)

```python
def extract_from_text(text) -> List[Dict]   # [{content, tags, category, entity}]
```

Uses regex patterns:
- **Preference patterns**: `prefer`, `like`, `love`, `hate` → category `"preference"`
- **Fact patterns**: `works at`, `lives in`, `uses` → category inferred from keywords

### LLM Extraction (`llm_extractor.py`)

```python
def llm_extract(messages, model, api_key=None) -> List[Memory]
def llm_extract_session(turns, speaker_a, speaker_b, session_date, model, api_key)
    -> (List[Memory], List[Dict])
```

- Requires `openai` package + `OPENROUTER_API_KEY`
- Default model: `openai/gpt-4.1-mini`
- Session extraction returns both facts and relation triples
- Constrained predicate vocabulary: `knows`, `works_at`, `lives_in`, `married_to`, etc.

---

## 12. MCP Server

> `agent_memory/mcp/server.py`

### Tools (8)

| Tool | Args | Description |
|------|------|-------------|
| `memory_add` | content (req), tags, category, entity, event_date, confidence | Store a new fact |
| `memory_get` | memory_id (req) | Fetch by ID |
| `memory_recall` | query (req), budget | Smart 3-layer retrieval |
| `memory_search` | query, category, entity, status, after, before, limit | Filter-based search |
| `memory_forget` | memory_id (req) | Archive a memory |
| `memory_timeline` | entity (req) | Chronological entity history |
| `memory_stats` | — | Store statistics |
| `memory_health` | — | Health check (call first!) |

### Resources

URI scheme: `memwright://`

| URI Pattern | Returns |
|-------------|---------|
| `memwright://entity/{key}` | Entity node + related entities |
| `memwright://memory/{id}` | Single memory object |

### Prompts

| Name | Arg | Description |
|------|-----|-------------|
| `recall` | `query` | Formatted recall results |
| `timeline` | `entity` | Formatted chronological history |

### Transport

stdio via `mcp.server.stdio.stdio_server`. Start with: `memwright mcp` or `agent-memory mcp`

---

## 13. Lifecycle Hooks

> `agent_memory/hooks/`

All hooks: read JSON from stdin → process → write JSON to stdout. Never raise exceptions.

### SessionStart Hook

- **Trigger**: Session begins
- **Store**: `<cwd>/.memwright`
- **Action**: `recall_as_context("session context project overview recent decisions", budget=20000)`
- **Output**: `{"additionalContext": "<memories>"}`
- **Budget**: 20K tokens (aggressive, user-configured)

### PostToolUse Hook

- **Trigger**: After every tool use
- **Store**: `<cwd>/.memwright`
- **Captures**:
  - `Write` tool → `"Created/wrote file {path}"` (tags: file-change, write)
  - `Edit` tool → `"Edited file {path}"` (tags: file-change, edit)
  - `Bash` tool → `"Ran command: {cmd}\nOutput: {first 200 chars}"` (tag: command)
- **Ignores**: `Read` tool, read-only bash (cat, ls, head, etc.), unknown tools

### Stop Hook

- **Trigger**: Session ends
- **Store**: `<cwd>/.memwright`
- **Action**: Queries memories from last hour, builds summary, stores as `category="session"` memory
- **Summary**: `"Session summary: N file changes, M commands run. Categories: cmd: N, project: M."`

---

## 14. CLI Reference

> `agent_memory/cli.py`

### Commands

| Command | Description |
|---------|-------------|
| `init <path> [--tool]` | Create store + print MCP config |
| `add <path> <content>` | Add a memory |
| `recall <path> <query>` | Smart retrieval with scores |
| `search <path> [query]` | Filter-based search |
| `list <path>` | List memories |
| `timeline <path> --entity E` | Entity timeline |
| `stats <path>` | Count/size statistics |
| `export <path> [-o file]` | JSON export |
| `import <path> <file>` | JSON import |
| `inspect <path>` | Raw DB dump (last 50) |
| `compact <path>` | Remove archived records |
| `forget <path> <id>` | Archive one memory |
| `serve <path>` | Start MCP server (explicit path) |
| `mcp [--path]` | Zero-config MCP server (`$MEMWRIGHT_PATH` or `~/.memwright`) |
| `setup-claude-code <path>` | Print Claude Code integration JSON |
| `doctor [path]` | Health check with pretty output |
| `locomo` | Run LOCOMO benchmark |
| `mab` | Run MemoryAgentBench |
| `hook {session-start\|post-tool-use\|stop}` | Dispatch lifecycle hook |

### Entry Points

Both `agent-memory` and `memwright` CLI commands map to `agent_memory.cli:main`.

---

## 15. Configuration

> `agent_memory/utils/config.py`

```python
@dataclass
class MemoryConfig:
    default_token_budget: int = 2000   # Default recall budget
    min_results: int = 3               # Minimum results to return
```

- Stored at `<store_path>/config.json`
- `from_dict()` filters unknown keys (forward-compatible)
- Loaded once during `AgentMemory.__init__`

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `MEMWRIGHT_PATH` | Default store path for MCP server |
| `OPENROUTER_API_KEY` | LLM extraction + benchmarks |
| `OPENAI_API_KEY` | Optional OpenAI embeddings for benchmarks |

---

## 16. Data Flow Diagrams

### Write Path: `mem.add()`

```
add(content, tags, category, entity, ...)
│
├── 1. Construct Memory object (uuid[:12], timestamps)
│
├── 2. TemporalManager.check_contradictions()
│      └── SQLite: SELECT WHERE entity=? AND category=? AND status='active'
│      └── Returns: conflicting Memory objects
│
├── 3. SQLiteStore.insert(memory)  [commit]
│
├── 4. For each conflict: TemporalManager.supersede(old, new.id)
│      └── old.status='superseded', old.valid_until=now
│      └── SQLiteStore.update(old)  [commit]
│
├── 5. ChromaStore.add(id, content)  [non-fatal]
│
└── 6. Extract entities/relations → graph updates  [non-fatal]
       └── NetworkXGraph.add_entity() + add_relation()
       └── Saves graph.json after each
```

### Read Path: `mem.recall()`

```
recall(query, budget=2000)
│
├── Layer 0: Graph Expansion
│   └── extract_tags(query) → keywords
│   └── For each keyword: graph.get_related(depth=2) → expanded entities
│   └── graph.get_edges() → SPO triples
│
├── Layer 1: Tag Match (score=1.0)
│   └── sqlite.tag_search(tags) → Memory list
│
├── Layer 2: Graph Entity Search (score=0.5)
│   └── For each expanded entity: sqlite.list_memories(entity=)
│   └── Only new (non-duplicate) memories added
│
├── Layer 3: Vector Search (score=1-distance)
│   └── chroma.search(query, limit=20)
│   └── For each new ID: sqlite.get(id) → Memory
│
├── Layer 4: Triple Injection (score=0.6)
│   └── SPO strings → synthetic Memory objects
│
└── Post-Processing
    ├── Reciprocal Rank Fusion (k=60)
    ├── Temporal Boost (max +0.2, decays over 90 days)
    ├── Entity Boost (+0.15 to +0.30)
    └── Budget Fitting (greedy token selection)
```

---

## 17. Design Patterns & Principles

### Zero-Config

Everything auto-provisions on first use. `AgentMemory("./path")` creates directory, schema, collections, and config. No setup step.

### Non-Fatal Failure

ChromaDB and NetworkX are wrapped in try/except. If they fail to initialize or during operations, the system continues with SQLite only. Use `health()` to detect degradation.

### Immutable Supersession

Memory content is **never mutated**. When facts change:
1. New memory is written first
2. Old memory gets `status='superseded'`, `valid_until=now`, `superseded_by=new.id`
3. Full history preserved via linked list

### Separation of Concerns

- `search()` — **filter-oriented**: direct field matching, returns `Memory`
- `recall()` — **relevance-oriented**: multi-layer cascade, returns scored `RetrievalResult`

### Hook Store Convention

- Hooks use `<cwd>/.memwright` — project-local
- MCP server uses `$MEMWRIGHT_PATH` or `~/.memwright` — user-global

### Eager Persistence

- SQLite commits after every write
- NetworkX saves graph.json after every entity/relation add
- No batching = no data loss on crash

---

## 18. Benchmarks

### LOCOMO

Tests conversational memory over multi-session dialogues.

- **v2 Score**: 81.2% accuracy
- **Competition**: MemMachine 84.9%, Zep ~75%, Letta 74%, Mem0 66.9%, OpenAI 52.9%

### MemoryAgentBench (MAB)

Tests multi-hop reasoning with complex queries.

| Configuration | Score | Notes |
|---------------|-------|-------|
| v2 (local embeddings) | 6.4% | all-MiniLM-L6-v2 too weak for multi-hop |
| v2 (OpenAI embeddings) | 18.8% | 3x improvement, CR still bottlenecked |

**Root cause**: 384D local embeddings insufficient for MAB's multi-hop reasoning. OpenAI text-embedding-3-small (1536D) helps but retrieval pipeline still limits cross-reference (CR) performance.

---

## 19. File Inventory

| File | Lines | Role |
|------|-------|------|
| `core.py` | ~428 | Main AgentMemory API |
| `models.py` | ~105 | Memory, RetrievalResult dataclasses |
| `cli.py` | ~615 | CLI entry point |
| `store/sqlite_store.py` | ~217 | SQLite storage |
| `store/chroma_store.py` | ~95 | ChromaDB vector store |
| `store/schema.sql` | ~22 | SQLite table definition |
| `graph/networkx_graph.py` | ~238 | Entity graph |
| `graph/extractor.py` | ~125 | Rule-based entity extraction |
| `retrieval/orchestrator.py` | ~205 | 3-layer retrieval cascade |
| `retrieval/tag_matcher.py` | ~36 | Keyword extraction |
| `retrieval/scorer.py` | ~89 | Scoring, boosts, budget fitting |
| `temporal/manager.py` | ~75 | Contradiction detection |
| `extraction/extractor.py` | ~101 | Extraction dispatcher |
| `extraction/rule_based.py` | ~79 | Regex-based extraction |
| `extraction/llm_extractor.py` | ~243 | LLM extraction via OpenRouter |
| `mcp/server.py` | ~500 | MCP server (8 tools) |
| `hooks/session_start.py` | ~57 | SessionStart hook |
| `hooks/post_tool_use.py` | ~97 | PostToolUse hook |
| `hooks/stop.py` | ~108 | Stop hook |
| `utils/config.py` | ~49 | Configuration |
| `utils/tokens.py` | ~6 | Token estimation |

### Dependencies

**Core** (always required):
- `chromadb>=0.4.0` — vector store
- `networkx>=3.0` — entity graph

**Optional extras**:
- `mcp>=1.0.0` — MCP server protocol
- `openai>=1.0.0` — LLM extraction

**Built-in** (no install):
- `sqlite3` — core storage
- `sentence-transformers` — pulled in by ChromaDB

---

## Quick Reference Card

```
# Initialize
mem = AgentMemory("./my-store")

# Write
m = mem.add("Alice works at Acme Corp", tags=["career"], category="career", entity="Alice")

# Smart Recall (full pipeline)
results = mem.recall("Where does Alice work?", budget=2000)
for r in results:
    print(f"[{r.match_source}:{r.score:.2f}] {r.content}")

# Filter Search
mems = mem.search(category="career", entity="Alice")

# Timeline
history = mem.timeline("Alice")

# Health Check
h = mem.health()
print("Healthy" if h["healthy"] else "Degraded")

# Cleanup
mem.close()
```
