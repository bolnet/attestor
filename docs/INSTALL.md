# Memwright Install Guide

A step-by-step guide to installing and verifying Memwright across different topologies. Each chapter is a self-contained setup you can follow from scratch.

**Chapters**

| # | Topology | Backend | Network required |
|---|----------|---------|------------------|
| [01](#chapter-01--local-with-chromadb) | Local embedded | SQLite + ChromaDB + NetworkX | No (after first model download) |
| [02](#chapter-02--sidecar-rest-api) | Sidecar REST API | Same as local | Localhost only |
| [03](#chapter-03--cloud-managed) | Cloud managed | Postgres / ArangoDB / AWS / Azure / GCP | Yes |

---

## Chapter 01 — Local with ChromaDB

**The zero-config path.** SQLite for documents, ChromaDB for vectors, NetworkX for the entity graph. Everything runs in-process. No Docker. No API keys. No external services.

### Prerequisites

- Python 3.10 or later
- `pip`, `pipx`, or `poetry`

That's it. All three backends are embedded and auto-provision on first use.

### Step 1 — Install the package

Choose one:

```bash
# Via pip (into current environment)
pip install memwright

# Via pipx (isolated CLI tool)
pipx install memwright

# Via poetry (project dependency)
poetry add memwright
```

Verify the CLI is available:

```bash
memwright --help
```

### Step 2 — Create a store

```python
from agent_memory import AgentMemory

mem = AgentMemory("~/.memwright")
```

On first call, Memwright:
1. Creates the directory if it doesn't exist
2. Provisions SQLite with WAL mode, 17 columns, 6 indexes
3. Initializes ChromaDB with local `all-MiniLM-L6-v2` embeddings (384 dimensions, ~90 MB download on first use)
4. Creates the NetworkX graph with JSON persistence

No configuration file is needed. The defaults work.

### Step 3 — Write your first memory

```python
mem.add(
    "The order service uses event sourcing with a 30-day retention policy",
    entity="order-service",
    tags=["architecture", "decision"],
)
```

This writes to all three stores in parallel:
- **Document store** (SQLite) — content, tags, entity, timestamp, confidence
- **Vector store** (ChromaDB) — 384-dimensional embedding of the content
- **Graph store** (NetworkX) — entity node `order-service` + typed edges

### Step 4 — Recall

```python
results = mem.recall("how is the order service structured?", budget=2000)
for r in results:
    print(f"[{r.score:.2f}] {r.memory.content}")
```

The 5-layer retrieval pipeline runs:
1. **Tag match** — SQLite FTS finds memories tagged with relevant terms
2. **Graph expansion** — BFS depth=2 finds related entities
3. **Vector search** — cosine similarity on the query embedding
4. **Fusion + Rank** — RRF (k=60) merges all candidates, applies PageRank and confidence decay
5. **Diversity + Fit** — MMR (lambda=0.7) removes near-duplicates, greedy packs under the token budget

### Step 5 — Run the doctor

The doctor verifies all four components are healthy:

```bash
memwright doctor ~/.memwright
```

Expected output:

```
Memwright Doctor
==================================================

Overall: ALL HEALTHY

  [OK] SQLiteStore (0.4ms, 1 memories, 40,960 bytes)
  [OK] ChromaDB (1 vectors)
  [OK] NetworkXGraph (1 nodes, 0 edges)
  [OK] Retrieval Pipeline (3/3 layers)
```

All four checks must show `[OK]`. If vector or graph shows `[FAIL]`, the self-healing health check will attempt automatic recovery — run doctor again and check for the `^ recovered at health check` note.

### Step 6 — Verify from the CLI

```bash
# Add a memory
memwright add ~/.memwright "API rate limit is 1000 req/min" --tags api,limits

# Recall
memwright recall ~/.memwright "what are the rate limits?"

# View timeline
memwright timeline ~/.memwright --limit 10

# Check stats
memwright stats ~/.memwright
```

### What's on disk

After setup, `~/.memwright/` contains:

```
~/.memwright/
├── memory.db          # SQLite — source of truth (WAL mode)
├── memory.db-wal      # SQLite write-ahead log
├── chroma/            # ChromaDB persistent storage
│   └── ...            # Embedding model cached here on first use
└── graph.json         # NetworkX graph (entity nodes + edges)
```

Total disk footprint: ~90 MB (mostly the sentence-transformers model). The SQLite database itself starts at 40 KB.

### Self-healing

If ChromaDB or NetworkX fail at startup (e.g., corrupted model cache, missing graph file), Memwright degrades gracefully:

- **Vector down** — retrieval falls back to tag match + graph expansion
- **Graph down** — retrieval falls back to tag match + vector search
- **Document store** is the only hard dependency

When `health()` or `memwright doctor` is called, failed stores are automatically re-initialized. If recovery succeeds, the store is wired back into the retrieval pipeline without a restart.

### Claude Code integration

The fastest path for Claude Code users:

```
> install agent memory
```

This triggers the interactive wizard that configures:
- MCP server scope (global or project)
- Store path
- Lifecycle hooks (session-start, post-tool-use, stop)
- Namespace and token budget

Or manually add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "memwright",
      "args": ["mcp", "--store-path", "~/.memwright"]
    }
  }
}
```

### Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `memwright: command not found` | Not on PATH | Use `pipx install memwright` or check `pip show memwright` |
| ChromaDB shows `[FAIL]` | Model not downloaded | Run any `recall` — model downloads automatically on first embedding |
| `safetensors` noise on startup | Rust model loader prints to stderr | Normal on first load; suppressed after warmup |
| Doctor shows 0 vectors but N memories | Embedding failed for some memories | Re-add affected memories; check disk space for model cache |
| Graph shows 0 nodes after adding memories | Entity extraction found nothing | Add memories with explicit `entity=` parameter |

### Next steps

- **Chapter 02** — Run the REST API as a sidecar for non-Python agents
- **Chapter 03** — Switch to a cloud backend for shared multi-agent deployments

---

## Chapter 02 — Sidecar REST API

*Coming soon.* Run `memwright api --store-path ~/.memwright --port 8080` to expose the same API over HTTP. Any language can use the memory layer.

---

## Chapter 03 — Cloud Managed

*Coming soon.* PostgreSQL (pgvector + Apache AGE), ArangoDB, AWS, Azure, and GCP backends for shared multi-agent deployments.

### `mode: local` with auto-start

With `backend = "arangodb"`, `mode = "local"`, and `docker = true` in
`config.toml`, and the `[docker]` extra installed, Memwright will start an
`arangodb:3.12` container named `memwright-arangodb` on port 8529 the first
time the store is opened. Without the `[docker]` extra the open will fail
with an actionable error pointing at `pip install "memwright[docker]"`.

With `docker = false` (the default), `mode: local` assumes you manage the
container yourself — this is the recommended path for CI and shared dev
machines.
