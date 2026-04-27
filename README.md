# Attestor

**The memory layer for agent teams.** Self-hosted, deterministic retrieval, zero LLM in the critical path.

[![PyPI](https://img.shields.io/pypi/v/attestor?label=PyPI&color=C15F3C&labelColor=1A1614)](https://pypi.org/project/attestor/)
[![PyPI Downloads](https://img.shields.io/pypi/dm/attestor?label=installs%2Fmo&color=C15F3C&labelColor=1A1614)](https://pypi.org/project/attestor/)
[![GitHub Stars](https://img.shields.io/github/stars/bolnet/attestor?style=flat&label=stars&color=C15F3C&labelColor=1A1614)](https://github.com/bolnet/attestor/stargazers)
[![Build](https://github.com/bolnet/attestor/actions/workflows/workflow.yml/badge.svg)](https://github.com/bolnet/attestor/actions/workflows/workflow.yml)
[![Evals](https://github.com/bolnet/attestor/actions/workflows/evals.yml/badge.svg)](https://github.com/bolnet/attestor/actions/workflows/evals.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-1A1614.svg?labelColor=C15F3C)](LICENSE)

```
pip install attestor
```

| | |
|---|---|
| **Version** | `4.0.0a1` (alpha; greenfield rebuild — no v3 migration path) |
| **PyPI** | `attestor` |
| **Import** | `attestor` |
| **Live site** | <https://attestor.dev/> |
| **Repo** | <https://github.com/bolnet/attestor> |
| **License** | MIT |

---

## What it is

Attestor is a memory store for agent teams that need a **shared, tenant-isolated memory** with **bi-temporal replay**, **deterministic retrieval**, and an **auditable supersession chain**. It runs as a Python library, a Starlette REST service, or an MCP server — same API in all three.

It is built around three claims, each grounded in code:

1. **Bi-temporal — replay any past state.** Every memory has both event time (`valid_from` / `valid_until`) and transaction time (`t_created` / `t_expired`). Nothing is deleted; everything is queryable forever (`attestor/temporal/manager.py:43-73`, `core.py:888-890`).
2. **Semantic-first retrieval, no LLM in the hot path.** A six-step deterministic pipeline. Same query → same ranking. Unit-testable (`attestor/retrieval/orchestrator.py:1-14`).
3. **Conversation ingest with auditable conflict resolution.** Two-pass speaker-locked extraction, then a four-decision (`ADD / UPDATE / INVALIDATE / NOOP`) resolver per fact. Every supersession carries an `evidence_episode_id` (`attestor/extraction/conflict_resolver.py:98`).

### Designed for

- Multi-agent products where many LLMs write to the same memory store
- Regulated chat systems that need point-in-time reconstruction (compliance, audit, FOIA-style queries)
- Self-hosted deployments — your VPC, your Postgres, your Neo4j

### *Not* designed for

- A general-purpose vector database
- A RAG framework with built-in chunking, reranking, and orchestration
- An LLM agent runtime — Attestor is the memory backend; the agent loop is yours

---

## Quick start

### 1. Install

```bash
pip install attestor                 # or: pipx install attestor
```

**Or pull the container** (introspection-grade image, single layer over `python:3.12-slim`):

```bash
docker pull ghcr.io/bolnet/attestor:latest          # GitHub Container Registry
docker pull bolnet2025/attestor:latest              # Docker Hub
```

For full production use, point the container at an external Postgres + Neo4j via env vars (or compose them with `attestor/infra/local/docker-compose.yml`).

### 2. Bring up local Postgres + Neo4j

```bash
attestor setup local                                       # writes attestor/infra/local/docker-compose.yml
docker compose -f attestor/infra/local/docker-compose.yml up -d
```

Postgres 16 ships with `pgvector` (document + vector roles). Neo4j 5 ships with GDS (graph role: PageRank, BFS, Leiden).

### 3. Pull the default embedder

```bash
ollama pull bge-m3                   # 1024-D, 8K context, local-first default
```

The provider chain in `attestor/store/embeddings.py` checks `http://localhost:11434` first; cloud providers are fallbacks. Override via `ATTESTOR_EMBEDDING_PROVIDER` / `ATTESTOR_EMBEDDING_MODEL`.

### 4. Verify (mandatory)

```bash
attestor doctor
```

All four checks must be green for the default install: **Document Store**, **Vector Store**, **Graph Store**, **Retrieval Pipeline**. Graph (Neo4j) is required — the 6-step retrieval pipeline narrows on graph neighborhoods and the conversation ingest path writes typed edges (`uses`, `authored-by`, `supersedes`). The only hard dependency that *cannot* be down is the document store (Postgres); transient vector-probe failures are surfaced in the response trace rather than swallowed (`retrieval/orchestrator.py` — `vector_error` field).

### 5. Use it

```python
from attestor import AgentMemory, AgentContext, AgentRole

mem = AgentMemory()                  # picks up env / ~/.attestor.toml automatically

ctx = AgentContext(
    agent_id="researcher-1",
    role=AgentRole.RESEARCHER,
    namespace="acme-prod",
)

mem.add(
    content="Alice is the engineering manager",
    entity="alice",
    category="role",
    context=ctx,
)

results = mem.recall(query="who runs engineering?", context=ctx)
for r in results:
    print(r.score, r.memory.content)
```

> **SOLO mode (zero-config).** In v4, `AgentMemory().add('foo')` auto-provisions a singleton `local` user, an Inbox project (`metadata.is_inbox=true`), and a daily session — so the snippet above works on a fresh database without configuring identity (`core.py:179-209`). For multi-tenant production use, pass an explicit `AgentContext` with a real `namespace`.

### 6. Run a smoke benchmark (optional)

Verify your install end-to-end against a tiny LongMemEval slice. Defaults match the canonical benchmark stack: `openai/gpt-5.2` answerer, dual judges (`openai/gpt-5.2` + `anthropic/claude-sonnet-4.6`), `openai/gpt-5.2` distiller, OpenAI `text-embedding-3-large` truncated to 1024-D.

```bash
export OPENAI_API_KEY=...
.venv/bin/python scripts/lme_smoke_local.py --n 2
```

Every model and parameter is overridable via env var or CLI flag. See `--help` for the full table.

---

## Architecture

### Bi-temporal — replay any past state

Every memory carries two time axes:

| Axis | Columns | Meaning |
|------|---------|---------|
| **Event time** | `valid_from`, `valid_until` | When the fact is true *in the world* |
| **Transaction time** | `t_created`, `t_expired` | When the row landed *in the store* |

Plus a `superseded_by` chain. Old facts are never deleted — they remain queryable forever (`attestor/temporal/manager.py:30-66`).

```python
# What did we believe on March 1?
mem.recall(query="who runs engineering?", as_of="2026-03-01T00:00:00Z", context=ctx)

# Show me everything we knew about Alice between Feb and Apr
mem.recall(query="alice", time_window=("2026-02-01", "2026-04-01"), context=ctx)
```

`as_of` and `time_window` propagate end-to-end through the orchestrator and document store. Auto-supersession on write is wired into `core.py:add()` (`core.py:762, 784-785`): on every `add`, the temporal manager finds active rows with the same `(entity, category, namespace)` and different content, marks them `superseded`, sets `valid_until=now`, and links `superseded_by=<new_id>`. Detection is rule-based string equality today.

### Tenant isolation — Postgres Row-Level Security

Every tenant table (`users`, `projects`, `sessions`, `episodes`, `memories`, `user_quotas`, `deletion_audit`) carries a `tenant_isolation_*` policy keyed off the `attestor.current_user_id` session variable. An empty / unset value fails closed — no rows visible (`attestor/store/schema.sql:311-327`).

> **Honest disclosure.** Enforcement lives in **Postgres**, not Python. The `AgentRole` enum in `attestor/context.py:49-56` is metadata that flows onto memories for provenance; it does *not* gate operations in Python. RLS is what actually controls access. This is correct architecture for a memory backend, but worth knowing if you read the Python alone.

### The retrieval pipeline — semantic-first, six steps

`attestor/retrieval/orchestrator.py` runs the same six steps for every query:

1. **Vector top-K** — pgvector cosine, k=50
2. **Graph narrow** — Neo4j BFS depth ≤ 2 from each candidate's entity to the question entities; affinity bonus per hop (0-hop=+0.30, 1-hop=+0.20, 2-hop=+0.10; unreachable=−0.05). Discrete, not "soft".
3. **Triples inject** — typed-edge facts (`uses`, `authored-by`, `supersedes`) injected as synthetic memories
4. **MMR rerank** — λ=0.7
5. **Confidence decay + temporal boost** — recency lifts; stale, low-confidence rows fall
6. **Budget fit** — greedy monotonic-by-score pack into the caller's token budget

Every call writes a JSONL trace to `logs/attestor_trace.jsonl` (disable via `ATTESTOR_TRACE=0`).

### Three storage roles

| Role | Purpose | Default | Alternatives |
|------|---------|---------|--------------|
| **Document** | Source of truth (content, tags, entity, ts, provenance, confidence) | Postgres 16 | AlloyDB, ArangoDB, DynamoDB, Cosmos DB |
| **Vector** | Dense embedding per memory | pgvector | AlloyDB ScaNN, ArangoDB, OpenSearch Serverless, Cosmos DiskANN |
| **Graph** | Entity nodes + typed edges | Neo4j 5 + GDS | Apache AGE on AlloyDB, ArangoDB, Neptune, NetworkX (Azure) |

Postgres is the source of truth. Neo4j is **derived state, rebuildable from Postgres** — but it's required for the canonical install: graph expansion is step 2 of the retrieval pipeline and conversation ingest writes typed edges. The only role that cannot be down is the document store; the orchestrator records transient vector-probe failures in the response trace (`vector_error`) instead of swallowing them.

### Optional BM25 / FTS lane

A trigger-maintained `content_tsv` tsvector + GIN index lifts queries that embeddings under-recall (acronyms, IDs, rare proper nouns). Enabled when v4 schema is detected; fuses with the vector lane via Reciprocal Rank Fusion (RRF, k=60). Graceful no-op on backends without the column (`core.py:122-130`).

---

## Conversation ingest

The heavyweight write path that turns conversation turns into auditable memories. `core.py:ingest_round(turn)` orchestrates four passes:

```
turn  →  extract_user_facts(user_turn)        ┐
        extract_agent_facts(assistant_turn)   ┘  → resolve_conflicts → apply
```

### Two-pass speaker-locked extraction

`attestor/extraction/round_extractor.py:216, 258` — separate prompts for user vs assistant turns. The user-turn extractor only emits facts attributable to the user; the assistant-turn extractor only emits facts the assistant introduced. Stops cross-attribution. The "+53.6 over Mem0" delta in our LongMemEval scores comes from this split.

### Four-decision conflict resolver

`attestor/extraction/conflict_resolver.py:40, 98` — for each newly-extracted fact, an LLM call against existing similar memories returns one of:

| Decision | Effect |
|----------|--------|
| `ADD` | New info, no existing match — write fresh memory |
| `UPDATE` | Same entity + predicate, refined value — keep existing id |
| `INVALIDATE` | Old memory contradicted — mark superseded (timeline replays) |
| `NOOP` | Already represented — skip |

Each `Decision` carries `evidence_episode_id`. Every supersession is auditable. Failsafe: parse failure on a single fact yields `ADD`-by-default — better a duplicate-ish row than a silent drop.

> **Two write paths, two contracts.** `mem.add(...)` runs the lightweight rule-based supersession (§Bi-temporal). `mem.ingest_round(turn)` runs the full four-decision pipeline. Pick `ingest_round` for conversational data; pick `add` for structured writes where you've already done the conflict reasoning.

### Sleep-time consolidation

`mem.consolidate()` (`core.py:526`) re-extracts and synthesizes facts from recent episodes with a stronger model. Currently a Python-API-only call — no CLI command. Schedule it from your application (cron, systemd timer, ECS scheduled task) when you want fresher facts than the streaming extractor produces.

### Reflection engine

`attestor/consolidation/reflection.py` runs periodic synthesis across N episodes for one user. Outputs:

- `stable_preferences` — patterns appearing in 3+ episodes
- `stable_constraints` — rules the user repeatedly invokes
- `changed_beliefs` — preferences that shifted (old → new, with explicit invalidate)
- `contradictions_for_review` — flagged for **HUMAN REVIEW**, *not* auto-resolved

The "do not auto-resolve" stance is the load-bearing piece for regulated chat systems. The prompt is explicit (`reflection.py:35-66`): *"Do NOT auto-resolve contradictions. Flag them for human review."*

### Chain-of-Note reading

```python
pack = mem.recall_as_pack(query="who runs engineering?", context=ctx)
# pack.memories : list of {id, content, validity_window, confidence, source_episode_id}
# pack.prompt   : default Chain-of-Note prompt with NOTES → SYNTHESIS → CITE → ABSTAIN → CONFLICT structure
```

The default prompt has explicit `ABSTAIN` and `CONFLICT` clauses — every frontier model defaults to confabulation otherwise.

---

## Multi-agent primitives

### Six roles

`AgentRole`: `ORCHESTRATOR`, `PLANNER`, `EXECUTOR`, `RESEARCHER`, `REVIEWER`, `MONITOR` (`attestor/context.py:49-56`). The role flows onto every memory's metadata for provenance. Access enforcement happens at the Postgres RLS layer (see §Tenant isolation).

### AgentContext — handoff, scratchpad, trail

```python
orchestrator = AgentContext.from_env(agent_id="orchestrator", namespace="project:acme")
planner      = orchestrator.as_agent("planner",  role=AgentRole.PLANNER)
executor     = planner.as_agent("executor",      role=AgentRole.EXECUTOR)

# Each child carries parent_agent_id + accumulating agent_trail.
# All three share the same scratchpad: Dict[str, Any] for typed handoff data.
```

`as_agent()` creates a child context with `parent_agent_id`, full `agent_trail`, and a shared `scratchpad`. The trail accumulates — useful for proving "this answer came from agent X who got it from agent Y."

### Per-agent token budgets

`AgentContext.token_budget` (default 20 000) is enforced — `recall()` packs results greedily until the budget is exhausted (`scorer.py:fit_to_budget`). `token_budget_used` accumulates across calls in a session.

### Optional write quotas

`mem.set_quota(user_id, daily_writes=...)` → enforced on `add` against the v4 `user_quotas` table (`core.py:592-621`). Optional; unset means unlimited.

---

## Security & Compliance

### Row-Level Security

Cross-link to §Tenant isolation. RLS policies are the access-control surface; the Python layer trusts them. Set `attestor.current_user_id` per connection.

### Provenance on every memory

Every memory carries `agent_id`, `session_id`, `source_episode_id`. The supersession chain (`superseded_by`) is preserved forever. Conversation episodes are stored verbatim, separate from the memories extracted from them — meaning you can always reconstruct *which conversation turn produced which fact*.

### Deletion audit log

Hard deletes (e.g., GDPR purges) write a row to `deletion_audit` before the cascade — what was deleted, when, why, by whom. This is the carve-out for the otherwise-immutable schema.

### GDPR — export and purge

```python
mem.export_user(external_id="user-42")     # full data export (memories + episodes + sessions + projects)
mem.purge_user(external_id="user-42",      # cascading hard delete with audit trail
               reason="GDPR right-to-erasure request 2026-04-27")
mem.deletion_audit_log(limit=100)          # forensic readback
```

`core.py:557-590`. v4 only. Returns / writes everything Subject Access requires for Art. 15 / Art. 17.

### Optional: Ed25519 provenance signing

Enable via config (`signing.enabled = true`). On every `add`, attestor signs the canonical payload `id || agent_id || t_created || content_hash` with an Ed25519 key. `mem.verify_memory(memory_id)` returns `bool` (`core.py:623-640`). Optional, off by default — turn on for adversarial-write contexts where you need cryptographic non-repudiation.

---

## Runtime topologies

Same API across all three. Only configuration changes.

| Mode | Shape | When to use |
|------|-------|-------------|
| **A — Embedded library** | `AgentMemory(config)` in-process; talks directly to Postgres + Neo4j | Single-process agents, scripts, notebooks |
| **B — Sidecar** | `attestor api` on `localhost:8080`; language-agnostic HTTP client shares the same Postgres + Neo4j | Polyglot agents on one box (Python + TS + Go) |
| **C — Shared service** | One Attestor service in front of an agent mesh (App Runner / Cloud Run / Container Apps) backed by managed Postgres + Neo4j | Production multi-agent platforms |

```bash
attestor api    --port 8080         # Mode B / C — Starlette ASGI REST (HTTP)
attestor mcp    --path ~/.attestor  # MCP stdio server (zero-config; for Claude Desktop / Cursor / Windsurf)
attestor serve  ~/.attestor         # MCP stdio server (positional-path variant; equivalent transport)
```

---

## Backends

| Backend | Document | Vector | Graph | Status |
|---------|:--------:|:------:|:-----:|--------|
| **Postgres + Neo4j** *(default)* | ✓ | pgvector | Neo4j + GDS | Production-ready |
| **ArangoDB** | ✓ | ✓ | ✓ | Production-ready (one engine, all 3 roles) |
| **AWS** | DynamoDB | OpenSearch Serverless | Neptune | Backend code + Terraform shipped |
| **Azure** | Cosmos DB | Cosmos DiskANN | NetworkX (in-process) | Backend code shipped, Terraform forthcoming |
| **GCP** | AlloyDB | AlloyDB ScaNN | AGE on AlloyDB | Backend code shipped, Terraform forthcoming |

Override the default via config:

```toml
# ~/.attestor.toml
backend = "postgres+neo4j"   # or "arangodb" | "aws" | "azure" | "gcp"
```

Reference Terraform lives under `attestor/infra/`.

---

## Embeddings

Provider auto-detect (`attestor/store/embeddings.py:get_embedding_provider`), in this order:

1. **Local Ollama `bge-m3`** — 1024-D, 8K context — used when `http://localhost:11434` is reachable
2. **Cloud-native** — Bedrock Titan / Vertex / Azure OpenAI when their SDK + creds are present
3. **OpenAI `text-embedding-3-large`** (3072-D native; pin `OPENAI_EMBEDDING_DIMENSIONS=1024` for schema compat)
4. **OpenRouter** — for federated runs

Local-first by design. Override:

```bash
export ATTESTOR_DISABLE_LOCAL_EMBED=1            # skip the Ollama probe entirely
export ATTESTOR_EMBEDDING_PROVIDER=openai
export ATTESTOR_EMBEDDING_MODEL=text-embedding-3-large
```

---

## CLI

`attestor --help` lists everything. The most useful commands:

| Command | Purpose |
|---------|---------|
| `attestor init` | Create a starter config |
| `attestor setup local` | Generate Docker Compose for Postgres + Neo4j |
| `attestor doctor` | Health-check every store + the retrieval pipeline |
| `attestor add` / `recall` / `search` / `list` | CRUD-ish memory ops |
| `attestor timeline` | Entity timeline (uses bi-temporal manager) |
| `attestor stats` | Store statistics |
| `attestor export` / `import` | JSON dump / restore |
| `attestor compact` | Remove archived memories |
| `attestor update` / `forget` | Mutate / archive a memory |
| `attestor inspect` | Inspect raw database state |
| `attestor api` | Start the Starlette REST API |
| `attestor serve <path>` | Start MCP stdio server (positional-path variant) |
| `attestor mcp [--path …]` | Start MCP stdio server (zero-config; default for Claude Desktop / Cursor / Windsurf) |
| `attestor ui` | Read-only browser UI for the store |
| `attestor hook {session-start, post-tool-use, stop}` | Run a Claude Code lifecycle hook |
| `attestor lme` / `locomo` / `mab` | Built-in benchmark runners (see §Evaluation) |

---

## MCP server

`attestor mcp` (or `attestor serve <path>`) exposes an MCP stdio server with eight tools:

| Tool | Purpose |
|------|---------|
| `memory_add` | Write a memory with provenance |
| `memory_get` | Fetch one memory by id |
| `memory_recall` | Run the full retrieval pipeline |
| `memory_search` | Filtered list (entity / category / time / namespace) |
| `memory_forget` | Archive a memory by id |
| `memory_timeline` | Chronology for an entity |
| `memory_stats` | Store statistics |
| `memory_health` | Per-role health snapshot — call this first when integrating |

Plus MCP **resources** (memory listings) and **prompts** (canned recall prompts for IDE assistants).

---

## Hooks (Claude Code)

Three lifecycle hooks ship in `attestor/hooks/`:

- **`session_start`** — injects relevant memories into the session context based on cwd / repo
- **`post_tool_use`** — auto-captures useful artifacts from `Write` / `Edit` / `Bash`
- **`stop`** — writes a session summary on exit

Wire them up via the installer (next section) or by hand in `~/.claude/settings.json`.

---

## Install for Claude Code

Single instruction users can give Claude Code:

```
install attestor
```

(Or run `/install-attestor`.) The installer interviews you on:

1. **Scope** — global (`~/.claude/.mcp.json`) vs project (`.mcp.json`)
2. **Postgres connection** — local Docker, Neon, RDS, etc.
3. **Neo4j connection** — local Docker, AuraDB, etc.
4. **Backend override** — default `postgres+neo4j`, or `arangodb` / `aws` / `azure` / `gcp`
5. **Embedding provider** — local Ollama (default), OpenAI, or cloud-native
6. **Hooks** — whether to wire `session-start` / `post-tool-use` / `stop`
7. **Namespace + default token budget**

Then it installs `attestor` via pipx, writes the MCP config, optionally writes `settings.json` hooks, and runs `attestor doctor` to verify.

---

## Evaluation

> **Boundary statement.** The dual-LLM judge stack is a **benchmarking** mechanism, *not* the runtime contract. Recall in production is single-pipeline and deterministic. Multiple judges score answers in evaluation only — never in user-facing reads.

| Runner | Source | Measures |
|--------|--------|----------|
| `attestor lme` | LongMemEval (Google's long-memory benchmark) | answer accuracy under long history, distillation, dual-judge cross-family |
| `attestor locomo` | LoCoMo | conversational long-memory consistency |
| `attestor mab` | MultiAgentBench | multi-agent coordination |
| AbstentionBench (CI gate) | internal | when *not* to answer — known unknowns |
| `scripts/lme_smoke_local.py` | dual-LLM smoke | quick install verification (see Quick Start §6) |

The smoke driver mirrors the canonical published-benchmark stack exactly. See `--help` for the full env-var / CLI-flag override matrix.

---

## Project layout

```
attestor/
  core.py                  -- AgentMemory (main public API)
  client.py                -- MemoryClient (HTTP drop-in for remote Attestor)
  context.py               -- AgentContext, AgentRole, Visibility
  models.py                -- Memory, RetrievalResult, ContextPack
  cli.py                   -- attestor CLI entry point
  api.py                   -- Starlette ASGI REST API
  longmemeval.py           -- LongMemEval benchmark runner (dual-judge)
  locomo.py                -- LoCoMo runner
  doctor_v4.py             -- v4 schema + invariant validator
  init_wizard.py           -- interactive install flow
  store/
    base.py                -- DocumentStore / VectorStore / GraphStore protocols
    registry.py            -- backend selection
    connection.py          -- config layering / env resolution
    embeddings.py          -- provider auto-detect (Ollama / OpenAI / Bedrock / Vertex / Azure)
    postgres_backend.py    -- pgvector (document + vector roles)
    neo4j_backend.py       -- Neo4j + GDS (graph role)
    arango_backend.py      -- all 3 roles in one
    aws_backend.py         -- DynamoDB + OpenSearch Serverless + Neptune
    azure_backend.py       -- Cosmos DB DiskANN + NetworkX
    gcp_backend.py         -- AlloyDB pgvector + AGE + ScaNN
    schema.sql             -- v4 Postgres schema (RLS, bi-temporal columns, content_tsv)
  conversation/
    ingest.py              -- ingest_round() pipeline
  extraction/
    round_extractor.py     -- 2-pass speaker-locked extraction
    conflict_resolver.py   -- 4-decision contract (ADD/UPDATE/INVALIDATE/NOOP)
    rule_based.py          -- deterministic fact extraction (no LLM)
    prompts.py             -- shared prompt templates
  consolidation/
    consolidator.py        -- sleep-time re-extraction
    reflection.py          -- cross-thread synthesis (stable patterns + flagged contradictions)
  graph/
    extractor.py           -- entity / relation extraction
  retrieval/
    orchestrator.py        -- 6-step semantic-first pipeline
    tag_matcher.py
    scorer.py              -- MMR, confidence decay, entity boost, fit-to-budget
    trace.py               -- JSONL trace writer
  temporal/
    manager.py             -- timelines, supersession, contradiction detection, as_of replay
  identity/
    signing.py             -- Ed25519 provenance signing (optional)
    defaults.py            -- SOLO mode auto-provisioning
  mcp/
    server.py              -- MCP server (tools, resources, prompts)
  hooks/
    session_start.py
    post_tool_use.py
    stop.py
  ui/
    app.py                 -- Starlette read-only viewer
    static/, templates/    -- Evidence Board UI
  utils/
    config.py, tokens.py
  infra/
    local/                 -- Docker Compose (Postgres + Neo4j)
    aws_arango/            -- Reference Terraform
tests/                     -- Unit tests; live cloud tests env-gated
evals/                     -- LongMemEval / LoCoMo / MultiAgentBench / AbstentionBench harnesses
docs/                      -- Architecture notes, ADRs
commands/                  -- /install-attestor, etc.
scripts/                   -- lme_smoke_local.py, etc.
```

---

## Development

```bash
poetry install
poetry run pytest tests/ -q                          # unit tests, no external services needed
ATTESTOR_LIVE_PG=1 poetry run pytest tests/live -q   # live integration (env-gated)
```

Style: `black` formatting, `isort` imports, `ruff` lint, `mypy` types. PEP 8, type-annotated signatures, dataclasses for DTOs. Many small files (200–400 lines typical, 800 max).

Conventions worth knowing:

- Postgres is the source of truth. Neo4j is derived; rebuild it from Postgres if it drifts.
- Non-fatal errors in vector / graph paths are caught and logged. The document path never silently breaks.
- Configuration layering: env vars → `~/.attestor.toml` → in-code overrides.
- Two write paths: `add()` for structured (lightweight rule-based supersession), `ingest_round()` for conversational (full 2-pass + 4-decision contract).

---

## Health check

Always call this first when integrating:

```bash
attestor doctor                  # CLI
```

```python
mem = AgentMemory()
print(mem.health())              # Python API
```

```jsonc
// MCP
{ "tool": "memory_health" }
```

It probes Document Store (Postgres), Vector Store (pgvector), Graph Store (Neo4j), and the retrieval pipeline. All four are required for the default topology — graph expansion is step 2 of the canonical pipeline, not an optional accelerator. Transient vector-probe failures surface in the `recall()` trace (`vector_error`) so callers can distinguish a degraded result from a clean one.

---

## Status & versioning

- **Version:** 4.0.0a4 (alpha) — published to [PyPI](https://pypi.org/project/attestor/) and the [MCP Registry](https://registry.modelcontextprotocol.io/v0/servers?search=attestor) as `io.github.bolnet/attestor`
- **v3 → v4:** greenfield rebuild on a v4-native Postgres schema with hard tenant isolation, bi-temporal facts, and a no-LLM retrieval critical path. **There is no automated migration.** v3 was alpha-only with no production users; drop your v3 DB and reinstall.
- See [`CHANGELOG.md`](./CHANGELOG.md) for the full track-by-track changelog.

---

## License

MIT. See [`LICENSE`](./LICENSE).

<!-- mcp-name: io.github.bolnet/attestor -->

