<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/logo-dark.svg">
    <img alt="Memwright" src="docs/logo.svg" width="400">
  </picture>
</p>

<p align="center">
  <em>Embedded memory for AI agents. Open and inspectable. Works with Claude Code.</em>
</p>

<p align="center">
  <a href="https://pypi.org/project/memwright/"><img src="https://img.shields.io/pypi/v/memwright?color=C15F3C&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/memwright/"><img src="https://img.shields.io/pypi/pyversions/memwright?style=flat-square" alt="Python"></a>
  <a href="https://github.com/bolnet/agent-memory/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bolnet/agent-memory?style=flat-square" alt="License"></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.bolnet/memwright"><img src="https://img.shields.io/badge/MCP-Registry-C15F3C?style=flat-square" alt="MCP Registry"></a>
</p>

---

## Why Memwright?

AI agents forget everything between conversations. The typical fix is a managed vector database or cloud memory service. Memwright takes a different approach:

- **Open & inspectable** — Your memories live in a SQLite file. Run `sqlite3 memory.db` and see exactly what the agent knows. No black boxes.
- **3-layer retrieval** — Tag matching, Neo4j entity graph, and pgvector semantic search fused with Reciprocal Rank Fusion.
- **Token efficient** — 300-500 tokens per recall vs 15,000+ for full history replay.
- **Fully local** — Everything runs on your machine via Docker. No cloud dependency.

Works as a **Claude Code MCP server**, a **Cursor MCP server**, or a **Python library**.

## Install

```bash
pip install memwright[all]
```

> **Tip:** On macOS with Homebrew Python, use `pipx install "memwright[all]"` to install as a standalone CLI tool.

**Requirements:** [Docker Desktop](https://docker.com/products/docker-desktop) and an embedding API key (`OPENROUTER_API_KEY` or `OPENAI_API_KEY`).

## Quick Start — Claude Code

Paste this prompt into Claude Code and it handles the rest:

```
Set up memwright as this project's persistent memory.

1. Install: pip install "memwright[all]" (if pip fails on macOS, use: pipx install "memwright[all]")
2. Make sure Docker Desktop is running
3. Initialize: agent-memory init ~/.agent-memory/PROJECT_NAME
4. Add your embedding API key to ~/.agent-memory/PROJECT_NAME/.env (OPENROUTER_API_KEY or OPENAI_API_KEY)
5. Create .mcp.json in the project root:
   {
     "mcpServers": {
       "memory": {
         "command": "agent-memory",
         "args": ["serve", "~/.agent-memory/PROJECT_NAME"],
         "env": {
           "OPENROUTER_API_KEY": "your-key-here",
           "PG_CONNECTION_STRING": "postgresql://memwright:memwright@localhost:5432/memwright",
           "NEO4J_URI": "bolt://localhost:7687",
           "NEO4J_PASSWORD": "memwright"
         }
       }
     }
   }
6. Add to CLAUDE.md: Use memory_recall at conversation start, memory_add to store context.
7. Verify: agent-memory doctor ~/.agent-memory/PROJECT_NAME
```

### Manual Setup

```bash
# 1. Install
pip install "memwright[all]"

# 2. Initialize (starts Docker containers for pgvector + Neo4j)
agent-memory init ~/.agent-memory/my-project

# 3. Add your embedding API key
echo 'OPENROUTER_API_KEY=sk-or-...' >> ~/.agent-memory/my-project/.env
```

Add to your project's `.mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "agent-memory",
      "args": ["serve", "~/.agent-memory/my-project"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-...",
        "PG_CONNECTION_STRING": "postgresql://memwright:memwright@localhost:5432/memwright",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "memwright"
      }
    }
  }
}
```

> **Note:** The `env` block is required because the MCP server process doesn't auto-load the `.env` file.

Restart Claude Code. You now have 7 memory tools: `memory_add`, `memory_get`, `memory_recall`, `memory_search`, `memory_forget`, `memory_timeline`, `memory_stats`.

Add to your `CLAUDE.md`:

```markdown
## Memory
Use `memory_recall` at the start of each conversation with the user's first message.
Use `memory_add` to store preferences, decisions, and project context.
```

## Quick Start — Cursor

```bash
# 1. Install and initialize
pip install "memwright[all]"
agent-memory init ~/.agent-memory/my-project
```

Add to `.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "memory": {
      "command": "agent-memory",
      "args": ["serve", "~/.agent-memory/my-project"],
      "env": {
        "OPENROUTER_API_KEY": "sk-or-...",
        "PG_CONNECTION_STRING": "postgresql://memwright:memwright@localhost:5432/memwright",
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_PASSWORD": "memwright"
      }
    }
  }
}
```

## Quick Start — Python Library

```python
from agent_memory import AgentMemory

mem = AgentMemory("./my-agent")

# Store facts
mem.add("User prefers Python over Java",
        tags=["preference", "coding"], category="preference")
mem.add("User works at SoFi as Staff SWE",
        tags=["career"], category="career", entity="SoFi")

# Recall relevant memories
results = mem.recall("what language does the user prefer?")
for r in results:
    print(f"[{r.match_source}:{r.score:.2f}] {r.content}")

# Get formatted context string for prompt injection
context = mem.recall_as_context("user background", budget=500)

# Contradiction handling — old facts get auto-superseded
mem.add("User works at Google as Principal Eng",
        tags=["career"], category="career", entity="SoFi")
# ^ The SoFi memory is now superseded automatically
```

## Benchmarks

### LOCOMO (Long Conversation Memory)

| System | Score |
|--------|-------|
| MemMachine | 84.9% |
| Zep | ~75% |
| Letta | 74.0% |
| Mem0 (Graph) | 66.9% |
| **Memwright** | **62.5%** |
| OpenAI Memory | 52.9% |

*LOCOMO scores are [disputed across vendors](https://blog.getzep.com/lies-damn-lies-statistics-is-mem0-really-sota-in-agent-memory/). Numbers above are self-reported.*

### MemoryAgentBench (ICLR 2026)

| Category | Score |
|----------|-------|
| Accurate Retrieval | 55% |
| Conflict Resolution | 62% |
| **Overall** | **58.5%** |

**How Memwright uses LLMs:** Embeddings, entity extraction, memory extraction, and contradiction detection all use LLM calls. Retrieval combines tag matching, graph traversal, and vector search with RRF fusion — no LLM re-ranking or judge.

## How Retrieval Works

Multi-layer cascade with Reciprocal Rank Fusion:

```mermaid
graph TD
    Q[Query] --> T["Tag Match"]
    Q --> G["Graph · Neo4j"]
    Q --> V["Vector · pgvector"]
    T --> R[RRF Fusion]
    G --> R
    V --> R
    R --> O[Ranked Results]

    style Q fill:#C15F3C,stroke:#C15F3C,color:#fff
    style R fill:#C15F3C,stroke:#C15F3C,color:#fff
    style O fill:#161b22,stroke:#C15F3C,color:#F4F3EE
    style T fill:#161b22,stroke:#30363d,color:#F4F3EE
    style G fill:#161b22,stroke:#30363d,color:#F4F3EE
    style V fill:#161b22,stroke:#30363d,color:#F4F3EE
```

Entity relationships are traversed to find related memories (e.g., querying "Python" also finds memories about "FastAPI" if they're connected). Graph relationship triples are injected as synthetic context for multi-hop reasoning.

## CLI

```bash
agent-memory init ./store              # Initialize store + Docker + .env
agent-memory add ./store "text" ...    # Add a memory
agent-memory recall ./store "query"    # Multi-layer recall
agent-memory search ./store "text"     # Search memories
agent-memory list ./store              # List memories
agent-memory timeline ./store          # Entity timeline
agent-memory stats ./store             # Store statistics
agent-memory doctor ./store            # Health check
agent-memory serve ./store             # Start MCP server
agent-memory export ./store -o bak.json
agent-memory import ./store bak.json
```

## Architecture

```
AgentMemory
├── SQLite           — Core storage, always on
├── pgvector         — Semantic vector search (PostgreSQL)
├── Neo4j            — Entity graph, multi-hop traversal
├── Retrieval        — Multi-layer cascade with RRF fusion
├── Temporal         — Contradiction detection, supersession
├── Extraction       — Rule-based + optional LLM
├── MCP Server       — Claude Code / Cursor integration
└── CLI + Doctor     — Health check for all components
```

## Configuration

AgentMemory stores `config.json` in the memory store directory:

```json
{
  "default_token_budget": 2000,
  "min_results": 3,
  "pg_connection_string": "postgresql://memwright:memwright@localhost:5432/memwright",
  "neo4j_uri": "bolt://localhost:7687",
  "neo4j_password": "memwright"
}
```

Environment variables override config: `PG_CONNECTION_STRING`, `NEO4J_PASSWORD`, `OPENROUTER_API_KEY` / `OPENAI_API_KEY`.

## License

Apache 2.0

---

<sub>mcp-name: io.github.bolnet/memwright</sub>
