# Memwright

Your agent forgets everything. We fixed that.

Open source memory for AI agents.
81.2% LOCOMO | 1.4ms recall | 607 tests | $0/mo

---

## The Pain

Claude Code's built-in memory is a flat file called MEMORY.md. It works fine at first. But there is no search, no ranking. The whole file loads into context every message.

| | Month 1 | Month 3 | Month 6 |
|---|---|---|---|
| MEMORY.md | 2K tokens | 8K tokens | 15K tokens |
| Memwright | 2K tokens | 2K tokens | 2K tokens |

By month six you have 15,000 tokens of unranked noise burning your context window. And Claude starts ignoring half of it anyway because there is too much to process.

---

## What Others Do

- **Mem0** — $249/mo for graph memory. LLM call on every add.
- **Zep** — Requires Neo4j. P95 latency 632ms under load.
- **Letta** — Docker + PostgreSQL required. Replaces your entire agent stack.
- **OpenAI** — ChatGPT only. No API access. No self-hosting.

The pattern is always the same: lock-in, cost, complexity.

---

## Our Answer

```
$ poetry add memwright && claude mcp add memory -- memwright mcp
```

One package. Two commands. Done. No Docker. No API keys. No database setup. No monthly bill.

Works with Claude Code, Cursor, Windsurf, any MCP client. It is not a framework — it is a memory layer that plugs into whatever you already use.

---

## How It Works

Memory lives on disk, completely outside the context window. SQLite for storage, ChromaDB for vector search, NetworkX for the entity graph. All embedded, all auto-provisioned.

```
On Disk (never in context):          Context Window:
  SQLite    — core storage            System prompt
  ChromaDB  — vector search           User message
  NetworkX  — entity graph            memory_recall → 2K max
  10,000+ memories                    (4 best memories)
```

When the agent calls memory_recall, Memwright searches 10,000 memories and returns only the top 4 that fit your token budget. The other 9,996 never enter context.

---

## The Retrieval Pipeline

Five layers. No LLM anywhere in this pipeline. Fully deterministic.

```
Query: "deployment setup"
  |
  1. Tag Match (SQLite)         — 15 exact hits
  2. Graph Expansion (NetworkX) — 8 related via entity links
  3. Vector Search (ChromaDB)   — 20 semantically similar
  |
  4. RRF Fusion + Scoring       — 30 unique, ranked
     PageRank + Temporal Boost + Confidence Decay
  |
  5. MMR Diversity + Budget Fit  — 4 memories, 1,800 tokens
```

Tag matching finds exact hits. Graph expansion follows entity relationships — query "deployment" and it finds memories about AWS and Docker through graph edges. Vector search catches semantic matches the other two miss. Then RRF fusion combines all three, scores them, removes near-duplicates, and packs the top results into your token budget.

---

## Install

Install takes about 30 seconds. No config files to write. No environment variables to set.

```bash
$ poetry add memwright
$ claude mcp add memory -- memwright mcp
$ memwright doctor ~/.memwright
  SQLite:             ok
  ChromaDB:           ok
  NetworkX Graph:     ok
  Retrieval Pipeline: ok
```

---

## Memory in Action

Session 1 — Claude learns:
```
Claude: I'll remember that your deployment uses ECS with Fargate.
  → memory_add("Deployment uses AWS ECS with Fargate,
       3 services, us-west-2", tags=["aws","deployment"],
       category="technical", entity="AWS")
```

Session 2 — Claude recalls:
```
User: How is our app deployed?
Claude: → memory_recall("deployment setup", budget=2000)
  Found 3 memories:
  - "Deployment uses AWS ECS with Fargate, 3 services, us-west-2"
  - "CI/CD pipeline runs through GitHub Actions"
  - "Staging environment mirrors prod at half scale"
```

Completely new context window. Claude pulls back exactly what it learned before. Three relevant memories, within your token budget. No manual bookkeeping.

---

## Contradiction Handling

Automatic. Algorithmic. No LLM call.

```python
mem.add("User works at Google", category="career", entity="Google")
# status: active

mem.add("User works at Meta", category="career", entity="Meta")
# Google memory auto-superseded
# Full history preserved — nothing deleted
# Zero LLM calls
```

Old memory marked superseded, not deleted. Full timeline preserved for audit. Rule-based, not a vector similarity coin-flip. Deterministic every time.

---

## Benchmarks

**Accuracy (LOCOMO):**

| System | Score |
|---|---|
| MemMachine | 84.9% |
| **Memwright** | **81.2%** |
| Zep | ~75% |
| Letta | 74% |
| Mem0 | 66.9% |
| OpenAI | 52.9% |

**Latency (P50 recall):**

| System | P50 |
|---|---|
| **Memwright (PG)** | **1.4ms** |
| **Memwright (local)** | **9ms** |
| Mem0 | 200ms |
| Zep P95 | 632ms |

**Cost:**

| System | Monthly |
|---|---|
| Mem0 | $19-249 |
| Zep | $25+ |
| Letta | $20+ |
| **Memwright** | **$0** |

LOCOMO benchmark — the standard long-conversation memory test — Memwright scores 81.2%, second overall. Latency: 1.4 milliseconds on PostgreSQL, 9 milliseconds fully local. Mem0 is at 200ms. Zep hits 632ms at P95. And cost — every competitor has a monthly bill. Memwright is zero dollars. Forever. Apache 2.0.

---

## Runs Everywhere

Same API. Same retrieval pipeline. Same results. Different infrastructure.

- **Local:** SQLite + ChromaDB + NetworkX (zero config)
- **AWS:** App Runner + DynamoDB + OpenSearch + Neptune
- **GCP:** Cloud Run + AlloyDB + Vertex AI
- **Azure:** Container Apps + Cosmos DB DiskANN
- **PostgreSQL:** pgvector + Apache AGE (any host)
- **ArangoDB:** Native doc + vector + graph
- **Docker:** Self-hosted anywhere

```bash
./scripts/deploy.sh aws    # One command
./scripts/deploy.sh gcp
./scripts/deploy.sh azure
```

---

## Multi-Agent Support

Namespace isolation, six RBAC roles, provenance tracking, and write quotas. Native, not bolted on.

```python
from agent_memory.context import AgentContext, AgentRole

ctx = AgentContext.from_env(
    agent_id="orchestrator",
    namespace="project:acme",
    role=AgentRole.ORCHESTRATOR,
    token_budget=20000,
)

planner = ctx.as_agent("planner", role=AgentRole.PLANNER)
researcher = ctx.as_agent("researcher", read_only=True)
```

- Namespace isolation per agent, project, or user
- 6 RBAC roles: Orchestrator, Planner, Executor, Researcher, Reviewer, Monitor
- Provenance tracking, write quotas, token budgets
- Read-only mode for research agents

---

## Get Started

```bash
poetry add memwright
claude mcp add memory -- memwright mcp
```

- GitHub: github.com/bolnet/agent-memory
- PyPI: pypi.org/project/memwright
- MCP Registry: io.github.bolnet/memwright

Two commands. Install takes 30 seconds. Your agent remembers from session one.

---

Free. Open source. Apache 2.0. Python 3.10-3.14.

Your agent remembers.

Built by Surendra Singh.
