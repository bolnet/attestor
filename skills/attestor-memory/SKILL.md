---
name: attestor-memory
description: Enterprise memory layer for agent teams — durable, deterministic recall with bi-temporal facts, RBAC, and audit trails. Self-hosted Postgres + Pinecone + Neo4j; zero LLM in the critical path.
version: 4.0.0
capabilities:
  - memory.recall
  - memory.add
  - memory.timeline
  - memory.supersede
  - memory.consolidate
  - memory.forget
  - memory.audit
  - memory.state
license: MIT
homepage: https://attestor.dev
repository: https://github.com/bolnet/attestor
mcp_server: attestor
authors:
  - aarjay
---

# attestor-memory

## What this skill does

Attestor is a shared, tenant-isolated memory store for agent teams. It persists every fact across three storage roles (Postgres for documents, Pinecone for vectors, Neo4j for graph) and serves recall through a deterministic six-step cascade — same query, same ranking, no LLM in the hot path. Every fact carries a bi-temporal validity window so an agent can replay any past belief, and every supersession is auditable to its evidence episode.

## When to use it

Use attestor when the agent's job depends on memory that survives:

- Multi-session conversations where the user expects continuity weeks or months later (preferences, ongoing projects, durable identity facts).
- Multi-agent products where many agents — orchestrator, planner, executor, reviewer — share state and must not stomp on each other's writes.
- Regulated chat (healthcare, finance, legal) that needs point-in-time reconstruction: "what did the agent know on date X?" or "show every fact that ever superseded this one."
- Workflows where contradictions must be tracked, not silently overwritten — supersession produces an audit chain instead of mutation.
- Self-hosted deployments where the memory data must stay inside the customer's VPC.

## When NOT to use it

- **Short-lived single-session context.** If the relevant memory dies when the conversation ends, put it in the system prompt. Attestor is overkill.
- **Pure document retrieval.** If you need to chunk and rerank a corpus of PDFs, use a dedicated RAG framework or `file_search`. Attestor is for *facts*, not *passages*.
- **An LLM agent runtime.** Attestor is the memory backend; the orchestration loop is the caller's responsibility.
- **Inline tool use that doesn't need persistence.** Direct tool definitions are simpler.

## How to install

### 1. Install the Python package

```bash
pip install attestor
```

The wheel ships with this SKILL.md so any 2026 agent SDK that scans for skills will discover `attestor-memory` automatically.

### 2. Bring up the storage stack (local dev)

```bash
attestor setup local
docker compose -f attestor/infra/local/docker-compose.yml up -d
attestor doctor          # verifies Postgres + Pinecone + Neo4j are reachable
```

For production, point the same client at managed Postgres (Neon, RDS, Cloud SQL), Pinecone Cloud, and Neo4j AuraDB via `configs/attestor.yaml` or env vars — the API surface is identical.

### 3. Wire the MCP server

Attestor publishes an MCP server (`mcp_server: attestor` in the frontmatter above). Add it to the host harness:

#### Claude Code / Claude Desktop

```json
{
  "mcpServers": {
    "attestor": {
      "command": "attestor",
      "args": ["mcp"],
      "env": {
        "PINECONE_API_KEY": "${PINECONE_API_KEY}",
        "NEO4J_URI": "bolt://localhost:7687",
        "POSTGRES_DSN": "postgresql://attestor:attestor@localhost:5432/attestor"
      }
    }
  }
}
```

Or run the interactive installer once and let it write the config:

```bash
attestor doctor                # confirm storage is up
# In Claude Code, type:  install attestor
```

#### OpenAI Agents SDK / Responses API

Point the agent's MCP transport at the same `attestor mcp` stdio command, or at the Starlette HTTP sidecar (`attestor api` on `localhost:8080`) if the SDK prefers HTTP.

## API reference

The skill exposes six primitives on `attestor.AgentMemory`. Every signature below is verbatim from the codebase — no aliases, no aspirational names.

| Method | Purpose |
| --- | --- |
| `add(content, tags, category, entity, namespace, event_date, confidence, metadata, ...)` | Persist one fact. Auto-detects contradictions and supersedes the older one. Returns the stored `Memory`. |
| `recall(query, budget, namespace, user_id, as_of, time_window)` | Six-step retrieval cascade (vector + BM25 + RRF + graph + MMR + token-budget pack). Returns `list[RetrievalResult]`. |
| `timeline(entity, namespace)` | Chronological replay of every memory about an entity (active + superseded). Returns `list[Memory]`. |
| `current_facts(category, entity, namespace)` | Active, non-superseded memories only. The "what does the agent believe right now" view. |
| `forget(memory_id)` / `forget_before(date)` | Archive a single memory by id, or every memory created before a date. Returns `bool` / `int`. |
| `health()` | Structured status of all three backends + retrieval pipeline. Always call before integrating. |

Supplementary primitives an agent reaches for less often:

- `get(memory_id)` — fetch a single memory by id.
- `update(memory_id, content=..., tags=..., ...)` — edit fields in place. Re-indexes vectors when content changes.
- `search(query, category, entity, namespace, status, after, before, limit)` — filtered listing without the recall pipeline.
- `recall_as_pack(query, budget, user_id, as_of, time_window)` — `ContextPack` with citations + Chain-of-Note prompt for cite-or-abstain agents.
- `extract(messages, model, use_llm, namespace)` — pull facts out of a conversation transcript and store them.
- `consolidate(user_id, since=..., target_count=5, namespace=..., dry_run=False)` — **reflection pass**: distill a window of episodic memories into compact semantic facts, supersede the originals, stamp `_consolidated_from` provenance. Returns `ReflectionResult`.
- `consolidate(limit=20, ...)` — legacy queue-drain mode (no `user_id`): runs one batch through the per-episode `SleepTimeConsolidator`.
- `export_user(external_id)` / `purge_user(external_id)` / `deletion_audit_log()` — GDPR data portability + erasure with audit trail.
- `pagerank(alpha)` — entity importance from the Neo4j graph.
- `stats()` / `ops_log` — store counts and a ring buffer of recent operation latencies.

### State lane — typed profile facts (`memory.state`)

Retrieval is the wrong tool for personalization. Durable, type-checked facts (preferences, capability declarations, durable identity facts) belong in a state object, not the embedding index. OpenAI's January 2026 `context_personalization` cookbook makes this case directly. Attestor exposes the state object as `mem.state`:

| Method | Purpose |
| --- | --- |
| `mem.state.set(key, value, *, user_id, project_id=..., agent_id=..., scope=..., schema=...)` | Write a typed fact. Append-only — previous active row is stamped with `t_expired`. Optional `schema=` triggers JSON-Schema validation. |
| `mem.state.get(key, *, user_id, project_id=..., scope=...)` | Read the current value, or `None` if missing. |
| `mem.state.list(*, user_id, project_id=..., scope=..., prefix="")` | Return all active key/value pairs whose key starts with `prefix`. |
| `mem.state.history(key, *, user_id, ...)` | Every value this key has held, oldest first (bi-temporal). |
| `mem.state.as_of(key, *, ts, user_id, ...)` | Replay the value that was active at `ts`. |
| `mem.state.delete(key, *, user_id, ...)` | Mark the active row expired. History is preserved. |

Two reference schemas ship with the package: `user_preferences_v1` (theme, language, timezone, communication_style) and `agent_capability_v1` (capability_set, max_tokens, allowed_tools). Register your own schema directory with `attestor.state.register_schema_directory(...)`. Validation failures raise `StateValidationError`.

RBAC is identical to the memory lane: WRITE for `set`/`delete`, READ for `get`/`list`/`history`/`as_of`. `read_only=True` strips writes regardless of role. The `AgentContext` surface mirrors the repo: `ctx.state_set(...)`, `ctx.state_get(...)`, `ctx.state_list(...)`, `ctx.state_delete(...)`.

```python
mem.state.set(
    "preferences",
    {"theme": "dark", "language": "en"},
    user_id=user.id,
    schema="user_preferences_v1",
)
mem.state.get("preferences", user_id=user.id)
# {"theme": "dark", "language": "en"}
```

Manual contradiction resolution (rare — `add()` does this automatically):

```python
# Underlying surface lives at mem._temporal.supersede(old_memory, new_memory_id).
# Use it only when add()'s auto-detection missed a paraphrased contradiction.
```

## Examples

### Example 1 — Write a fact, then recall it

```python
from attestor import AgentMemory

mem = AgentMemory("./agent-store")

mem.add(
    "The user prefers Python over Go",
    tags=["preference", "language"],
    category="preference",
    entity="user",
)

results = mem.recall("what programming language does the user like?", budget=1024)
for r in results:
    print(r.score, r.memory.content)
```

### Example 2 — Track an entity's history through time

```python
from attestor import AgentMemory

mem = AgentMemory("./agent-store")

mem.add("Acme Corp uses Postgres 14",  entity="Acme Corp", category="stack",
        event_date="2024-01-10")
mem.add("Acme Corp uses Postgres 15",  entity="Acme Corp", category="stack",
        event_date="2025-03-01")
mem.add("Acme Corp uses Postgres 16",  entity="Acme Corp", category="stack",
        event_date="2026-02-14")

# All three rows, oldest first; the first two are auto-superseded.
for m in mem.timeline("Acme Corp"):
    print(m.event_date, m.status, m.content)

# Just the live belief.
for m in mem.current_facts(entity="Acme Corp"):
    print(m.content)        # → "Acme Corp uses Postgres 16"
```

### Example 3 — Replay a past belief (bi-temporal recall)

```python
from datetime import datetime, timezone
from attestor import AgentMemory

mem = AgentMemory("./agent-store")

# What did the agent believe about Acme Corp's stack on 2025-06-01?
as_of = datetime(2025, 6, 1, tzinfo=timezone.utc)
past_results = mem.recall("Acme Corp postgres version", as_of=as_of)
for r in past_results:
    print(r.memory.content)  # → "Acme Corp uses Postgres 15"
```

`as_of` resolves on event time (`valid_from` / `valid_until`), so the answer reflects what was true *then*, not what the agent learned later.

### Example 4 — Multi-agent shared state with RBAC + namespace isolation

```python
from attestor import AgentContext, AgentMemory, AgentRole

shared_store = AgentMemory("./team-store")

orchestrator = AgentContext(
    agent_id="orchestrator-01",
    namespace="project:acme",
    role=AgentRole.ORCHESTRATOR,        # READ + WRITE + FORGET
    memory=shared_store,
)

# Hand off to a researcher (READ + WRITE only — no forget).
researcher = orchestrator.as_agent("researcher-01", role=AgentRole.RESEARCHER)
researcher.add_memory("Vendor X has SOC2 Type II since 2024-09",
                       tags=["compliance"], category="vendor")

# A reviewer can only read — write attempts raise PermissionError.
reviewer = orchestrator.as_agent("reviewer-01", role=AgentRole.REVIEWER)
hits = reviewer.recall("vendor SOC2 status")
print(orchestrator.agent_trail)         # full handoff chain for audit
```

Roles enforced at the context layer (`attestor/context.py`): `ORCHESTRATOR` = full perms; `PLANNER` / `EXECUTOR` / `RESEARCHER` = read + write; `REVIEWER` / `MONITOR` = read-only. `read_only=True` is an independent kill switch that strips writes regardless of role.

### Example 5 — Periodic reflection (distill many episodic memories into a few semantic ones)

```python
from datetime import datetime, timedelta, timezone
from attestor import AgentMemory

mem = AgentMemory("./agent-store")

# Run nightly: condense the last 30 days of episodic memories for a
# user into 5 attributed semantic facts. Originals are kept in the
# supersession chain — nothing is deleted, the audit trail stays
# queryable forever via timeline() and recall(as_of=...).
since = datetime.now(timezone.utc) - timedelta(days=30)
result = mem.consolidate(
    user_id="user-1234",
    since=since,
    target_count=5,
)
print(result.distilled_memory_ids)        # 5 fresh semantic memories
print(result.source_memory_ids)            # ids of every superseded source
print(f"~${result.cost_estimate_usd:.4f}") # rough $$ for this pass

# Each distilled memory carries provenance metadata you can audit.
for did in result.distilled_memory_ids:
    m = mem.get(did)
    print(m.metadata["_consolidated_from"])  # source ids cited
    print(m.metadata["_reflection_model"])    # LLM used
```

`dry_run=True` calls the LLM (so the cost estimate is accurate) but skips the writes — useful for canary deployments.

### Example 6 — Audit + GDPR

```python
from attestor import AgentMemory

mem = AgentMemory("./agent-store")

# Show every memory the agent stored about a user, ready for export.
dump = mem.export_user("user-1234")

# Honor a delete request (CASCADEs through Postgres, returns audit row).
result = mem.purge_user("user-1234", reason="gdpr_request",
                         deleted_by="support-agent-7")

# Verify it landed in the audit trail.
recent = mem.deletion_audit_log(limit=10)
```

## Configuration

The single source of truth is `configs/attestor.yaml`. It carries:

- `stack.backends` — which backend handles which role (document / vector / graph).
- `stack.embedder` — provider + model + dimension. Default: Pinecone Inference `llama-text-embed-v2` (1024-D).
- `stack.models` — LLM ids for extraction, conflict resolution, judge.
- `stack.retrieval` — recall hot-path tunables (`vector_top_k`, `mmr_lambda`, BM25 / HyDE / multi-query lane configs).

Override per-instance via the `config` kwarg on `AgentMemory(path, config=...)`. Environment variables (`PINECONE_API_KEY`, `NEO4J_URI`, `POSTGRES_DSN`, etc.) are resolved by `attestor/store/connection.py`. Run `attestor doctor` to surface any missing or mismatched values.

## Audit + compliance

Attestor is built for regulated workloads:

- **Bi-temporal storage.** Every memory has both event time (`valid_from` / `valid_until`) and transaction time (`t_created` / `t_expired`). Nothing is deleted on contradiction — the older fact is marked `superseded` and stays queryable forever via `timeline()` and `recall(as_of=...)`.
- **Provenance signing.** Opt-in Ed25519 signature on every memory (`signing` block in config). `mem.verify_memory(memory_id)` re-checks the signature against the canonical payload.
- **Audit trail via traces.** OpenTelemetry-style spans on every ingest / recall / supersede call (`attestor/trace.py`). Toggle with `ATTESTOR_TRACE=1`.
- **Tenancy.** Postgres row-level security scoped by `user_id`; namespaces are first-class; Neo4j namespace enforcement is partial (graph entity nodes are still global as of v4.0.0 — see CLAUDE.md).
- **GDPR-compatible erasure.** `purge_user()` issues a CASCADE delete and writes an audit row; export via `export_user()` produces a JSON-portable dump.
- **No LLM in the critical path.** The recall cascade is fully deterministic — same query, same ranking — which is what regulators want when they audit a recommendation.

## Health check

Always call `health()` first when integrating. The MCP server exposes the same probe as `memory_health`.

```python
from attestor import AgentMemory

mem = AgentMemory("./agent-store")
report = mem.health()
assert report["healthy"], report
```

`report["checks"]` lists Postgres, Pinecone, Neo4j, and the retrieval pipeline status with per-store latencies. If a backend was down at startup, `health()` attempts recovery before reporting — long-running processes self-heal without a restart.

## Further reading

- `README.md` — full quickstart, benchmark numbers, deployment topologies.
- `CLAUDE.md` — architecture notes for agents working on the codebase.
- `attestor/core/agent_memory.py` — the canonical AgentMemory implementation.
- `attestor/retrieval/orchestrator.py` — the deterministic six-step cascade.
- `attestor/temporal/manager.py` — supersession + `as_of` replay.
- `attestor/context.py` — RBAC matrix + AgentContext handoff semantics.
