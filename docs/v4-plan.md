# Attestor v3 → v4 Implementation Plan

**Source documents** (reference, not in repo):
- `~/Downloads/attestor_v4_roadmap.md` — 8 tracks (A–H), 6 sprints
- `~/Downloads/attestor_v4_tenancy.md` — 4 phases of multi-tenant identity
- `~/Downloads/attestor_v4_defaults.md` — zero-config defaults (SOLO/HOSTED/SHARED)

**Operating constraint:** Existing Postgres + Neo4j data has been wiped. **No v3 backwards-compatibility, no `legacy-v3` user, no namespace backfill.** Schema is v4-native from commit one. The `attestor` package version bumps to `4.0.0a1` on the first phase 0 commit.

---

## 1. Critical path

```
P0 schema  →  P1 identity  →  P2 defaults  →  P3 conversation ingest (Track A)
                  ↓                                  ↓
              [P4 retrieval]                    [P5 temporal]    [P6 reading]
                                                    ↓                ↓
                                                [P7 sleep-time consolidation]
                                                    ↓
                                                [P8 multi-agent + auth]
                                                    ↓
                                                [P9 eval harness]
                                                    ↓
                                                [P10 optional + P11 release]
```

P0–P2 are sequential and load-bearing. P3 enables most of the rest. Tracks B/C/D run in parallel after P3 lands.

---

## 2. Phase-by-phase plan

### Phase 0 — Foundational schema (1–2 days)

**Goal:** v4-native schema lands; no behavior change yet.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/store/schema.sql` | Replace v3 schema with v4 (users, projects, sessions, episodes, memories with v4 columns) |
| `attestor/store/postgres_backend.py` | Update `_init_schema()` to create v4 tables; rip out v3 column-add logic |
| `attestor/store/postgres_backend.py` | Add `_init_rls()` to apply RLS policies |
| `attestor/models.py` | Add `MemoryScope` enum (USER/PROJECT/SESSION); `User`, `Project`, `Session` dataclasses |
| `tests/test_v4_schema.py` | New: schema integrity, indexes present, RLS policy correctness |
| `tests/test_v4_rls_isolation.py` | New: app-level bug ↔ RLS catches; one user can never see another's rows |

**Schema (single migration, no backfill):**

```sql
-- Drop the v3 memories table; we have no data
DROP TABLE IF EXISTS memories CASCADE;

-- Identity tables
CREATE TABLE users (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  external_id     VARCHAR(256) UNIQUE NOT NULL,
  email           VARCHAR(320),
  display_name    VARCHAR(256),
  status          VARCHAR(32) NOT NULL DEFAULT 'active',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  deleted_at      TIMESTAMPTZ,
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX idx_users_external_id ON users(external_id) WHERE status = 'active';

CREATE TABLE projects (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id         UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  name            VARCHAR(256) NOT NULL,
  description     TEXT,
  status          VARCHAR(32) NOT NULL DEFAULT 'active',
  created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  archived_at     TIMESTAMPTZ,
  metadata        JSONB NOT NULL DEFAULT '{}'::jsonb,
  UNIQUE (user_id, name)
);
CREATE INDEX idx_projects_user ON projects(user_id) WHERE status = 'active';

CREATE TABLE sessions (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  project_id          UUID REFERENCES projects(id) ON DELETE SET NULL,
  title               VARCHAR(512),
  status              VARCHAR(32) NOT NULL DEFAULT 'active',
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  last_active_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  ended_at            TIMESTAMPTZ,
  message_count       INT NOT NULL DEFAULT 0,
  consolidation_state VARCHAR(32) DEFAULT 'pending',
  metadata            JSONB NOT NULL DEFAULT '{}'::jsonb
);
CREATE INDEX idx_sessions_user_project_active
  ON sessions(user_id, project_id, last_active_at DESC)
  WHERE status = 'active';

CREATE TABLE episodes (
  id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id             UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  project_id          UUID REFERENCES projects(id),
  session_id          UUID NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
  thread_id           VARCHAR(128) NOT NULL,
  user_turn_text      TEXT NOT NULL,
  assistant_turn_text TEXT NOT NULL,
  user_ts             TIMESTAMPTZ NOT NULL,
  assistant_ts        TIMESTAMPTZ NOT NULL,
  agent_id            VARCHAR(128),
  metadata            JSONB NOT NULL DEFAULT '{}'::jsonb,
  created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
CREATE INDEX idx_episodes_user_session_ts ON episodes(user_id, session_id, user_ts);

-- Memories — v4 native
CREATE TABLE memories (
  id                UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id           UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
  project_id        UUID REFERENCES projects(id),
  session_id        UUID REFERENCES sessions(id),
  scope             VARCHAR(16) NOT NULL DEFAULT 'user',          -- user|project|session
  content           TEXT NOT NULL,
  tags              TEXT[] NOT NULL DEFAULT '{}'::text[],
  category          TEXT NOT NULL DEFAULT 'general',
  entity            TEXT,
  confidence        REAL NOT NULL DEFAULT 1.0,
  status            TEXT NOT NULL DEFAULT 'active',
  -- bi-temporal columns (Track A.4)
  valid_from        TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  valid_until       TIMESTAMPTZ,
  t_created         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
  t_expired         TIMESTAMPTZ,
  superseded_by     UUID REFERENCES memories(id),
  -- provenance (Track A.4 + F.2)
  source_episode_id UUID REFERENCES episodes(id),
  source_span       INT4RANGE,
  extraction_model  VARCHAR(64),
  agent_id          VARCHAR(128),
  parent_agent_id   VARCHAR(128),
  visibility        VARCHAR(32) NOT NULL DEFAULT 'team',
  signature         TEXT,
  -- existing fields
  metadata          JSONB NOT NULL DEFAULT '{}'::jsonb,
  embedding         VECTOR(<dim>)
);
CREATE INDEX idx_memories_user_scope_status
  ON memories(user_id, scope, status, valid_from DESC) WHERE status = 'active';
CREATE INDEX idx_memories_user_project
  ON memories(user_id, project_id) WHERE status = 'active' AND scope = 'project';
CREATE INDEX idx_memories_user_session
  ON memories(user_id, session_id) WHERE status = 'active' AND scope = 'session';
CREATE INDEX idx_memories_temporal
  ON memories USING GIST (tstzrange(valid_from, valid_until));
CREATE INDEX idx_memories_agent
  ON memories(user_id, agent_id, t_created);

-- Row-level security
ALTER TABLE memories  ENABLE ROW LEVEL SECURITY;
ALTER TABLE episodes  ENABLE ROW LEVEL SECURITY;
ALTER TABLE projects  ENABLE ROW LEVEL SECURITY;
ALTER TABLE sessions  ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_memories ON memories
  USING (user_id = current_setting('attestor.current_user_id', true)::uuid);
CREATE POLICY tenant_isolation_episodes ON episodes
  USING (user_id = current_setting('attestor.current_user_id', true)::uuid);
CREATE POLICY tenant_isolation_projects ON projects
  USING (user_id = current_setting('attestor.current_user_id', true)::uuid);
CREATE POLICY tenant_isolation_sessions ON sessions
  USING (user_id = current_setting('attestor.current_user_id', true)::uuid);

CREATE ROLE attestor_admin BYPASSRLS;
CREATE ROLE attestor_app   NOBYPASSRLS;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO attestor_app;
```

**Success criteria:**
- All tables created with NOT NULL constraints intact (no nullable user_id)
- 4 RLS policies active and verified by `test_v4_rls_isolation.py`
- `attestor doctor` reports schema OK
- v3 tests broken — that's expected; will be deleted in P1

---

### Phase 1 — Identity layer (3–5 days)

**Goal:** `User`, `Project`, `Session` are first-class objects in the Python API; AgentContext carries user_id/project_id/session_id; existing `add()`/`recall()` route through the new tenancy filter.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/identity/__init__.py` | New module |
| `attestor/identity/users.py` | `UserRepo`: create, get, find_by_external_id, soft_delete, purge |
| `attestor/identity/projects.py` | `ProjectRepo`: create, get, list_for_user, archive, delete (with Inbox protection) |
| `attestor/identity/sessions.py` | `SessionRepo`: create, get, list, resume, end, archive, autostart |
| `attestor/context.py` | Extend `AgentContext` with `user_id`, `project_id`, `session_id`, `scope_default`; add `for_chat()` factory |
| `attestor/core.py` | `AgentMemory.create_user/get_user/list_users/...` thin wrappers |
| `attestor/core.py` | `_set_rls_var()` helper that runs `SELECT set_config('attestor.current_user_id', ...)` on every checked-out connection |
| `attestor/store/postgres_backend.py` | Connection wrapper that sets RLS var per request |
| `tests/test_identity_repos.py` | CRUD on each repo; UNIQUE constraint enforcement; soft-delete semantics |
| `tests/test_isolation/test_user_isolation.py` | The 6-test contract from tenancy.md §4.4 |

**Success criteria:**
- All public AgentMemory methods accept `user_id`/`project_id`/`session_id`
- 6 isolation tests pass on every PR
- Test for "app forgot WHERE clause" returns 0 rows because of RLS
- v3 namespace parameter REMOVED (no legacy namespace string parsing)

---

### Phase 2 — Defaults and SOLO mode (2–3 days)

**Goal:** `mem.add("foo")` works on a fresh install with no config (zero-config singleton local user, Inbox project, daily session).

**Deliverables:**

| File | Action |
|---|---|
| `attestor/mode.py` | New: `AttestorMode` enum + `detect_mode()` |
| `attestor/identity/defaults.py` | New: `_ensure_solo_user()`, `ensure_inbox()`, `_get_or_create_daily_session()` |
| `attestor/core.py` | `_resolve()` method (single entry point per defaults.md §5) |
| `attestor/core.py` | All public methods (`add`, `recall`, `recall_as_context`, `search`, etc.) call `_resolve()` first |
| `attestor/cli.py` | Default to SOLO mode for `attestor` CLI commands |
| `tests/test_solo_defaults.py` | Singleton user idempotency; daily session rotation; Inbox immutability |
| `tests/test_resolve.py` | Resolution chain correctness across all mode/scope combos |

**Success criteria:**
- `from attestor import AgentMemory; AgentMemory().add("hi")` works in 0 lines of identity code
- `delete_project(inbox_id)` raises `ValidationError`
- HOSTED mode rejects calls without auth (401)

---

### Phase 3 — Track A: Conversation ingest pipeline (5–7 days)

**Goal:** `mem.ingest_round(user_turn, assistant_turn, ctx)` produces verbatim episode + extracted facts + ADD/UPDATE/INVALIDATE/NOOP decisions.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/conversation/__init__.py` | New module |
| `attestor/conversation/turns.py` | `ConversationTurn` dataclass |
| `attestor/conversation/ingest.py` | `ConversationIngest`, `IngestConfig`, `RoundResult` |
| `attestor/conversation/episodes.py` | `EpisodeRepo`: write verbatim user+assistant turns to `episodes` |
| `attestor/extraction/prompts.py` | `USER_FACT_EXTRACTION_PROMPT`, `AGENT_FACT_EXTRACTION_PROMPT`, `MEMORY_UPDATE_PROMPT` (verbatim from roadmap §A.2/A.3) |
| `attestor/extraction/extractor.py` | New: `extract_user_facts()`, `extract_agent_facts()` — speaker-locked, JSON output, source_span citations |
| `attestor/extraction/conflict_resolver.py` | New: `resolve_conflicts(new_facts, existing) → list[Decision]` using MEMORY_UPDATE_PROMPT |
| `attestor/conversation/apply.py` | Apply ADD/UPDATE/INVALIDATE/NOOP through existing supersession path |
| `attestor/core.py` | Public `mem.ingest_round()` method |
| `tests/test_conversation_ingest.py` | Round shape, episode write, fact extraction roundtrip, decision application |
| `tests/test_extraction_prompts.py` | Prompt content guards (anti-regression on speaker-lock IMPORTANT line, JSON schema, source_span) |

**Success criteria from roadmap §A.5:**
- `single-session-assistant` LongMemEval ≥ 80% (today: ~0% with no agent extraction path)
- `knowledge-update` ≥ 15 points above current baseline
- 100% of extracted facts have `source_episode_id` + `source_span`

---

### Phase 4 — Track B: Retrieval upgrades (3–5 days, parallel with P3)

**Goal:** Fact-augmented embedding keys + BM25 lane added to existing 4-lane retrieval cascade.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/store/postgres_backend.py` | Modify `_build_embedding_text()` to concat `[fact || raw_user_turn || raw_assistant_turn || tags]` (B.1) |
| `attestor/retrieval/bm25.py` | New: BM25 lane using Postgres `ts_rank_cd` over the existing FTS index |
| `attestor/retrieval/orchestrator.py` | Add bm25_hits to fan-out; RRF fusion stays the same (k=60), now over 4–5 lanes |
| `attestor/retrieval/orchestrator.py` | Optional cross-encoder rerank gated behind `config.cross_encoder_model` (B.2 stretch) |
| `tests/test_bm25_lane.py` | BM25 returns graded relevance for queries without explicit tags |
| `tests/test_embedding_keys.py` | `_build_embedding_text` produces expected concatenation; embeddings change vs content-only baseline |

**Success criteria from roadmap §B.3:**
- `single-session-user` ≥ 85%
- `multi-session` improves by ≥ 5 points
- Recall@10 on a held-out fact-recall test ≥ 90%

---

### Phase 5 — Track C: Temporal reasoning (3–4 days)

**Goal:** Time-aware query expansion + bi-temporal `as_of` filter on every recall path.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/retrieval/temporal_query.py` | New: `TemporalQueryExpander`, `TIME_EXTRACTION_PROMPT`, `TimeWindow` dataclass |
| `attestor/retrieval/orchestrator.py` | Add temporal_hits lane that pre-filters by `tstzrange` overlap when `TimeWindow` is set |
| `attestor/core.py` | Add `as_of: Optional[datetime]` param to `recall`, `search`, `recall_as_context` |
| `attestor/store/postgres_backend.py` | Bi-temporal SQL filter: `tstzrange(valid_from, valid_until) @> $as_of` and `t_created <= $as_of AND COALESCE(t_expired, infinity) > $as_of` |
| `tests/test_temporal_query.py` | "last Tuesday", "before I had kids", "what's my favorite color" cases |
| `tests/test_as_of_replay.py` | Snapshot-at-date returns past belief, not current; works after invalidations |

**Success criteria from roadmap §C.4:**
- `temporal-reasoning` ≥ 75%
- "What did the system know on date X" returns past snapshot, not current

---

### Phase 6 — Track D: Chain-of-Note reading (1–2 days)

**Goal:** `recall_as_context()` returns a structured `ContextPack` with citations + Chain-of-Note prompt that includes ABSTAIN clause.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/models.py` | `ContextPack` dataclass |
| `attestor/prompts/chain_of_note.py` | New: `DEFAULT_CHAIN_OF_NOTE_PROMPT` (verbatim from roadmap §D.2) |
| `attestor/core.py` | Update `recall_as_context()` to return `ContextPack` with `chain_of_note_prompt` |
| `tests/test_context_pack.py` | Citations included, abstain clause present, sorted by score |

**Success criteria from roadmap §D.3:**
- `abstention` ≥ 80%
- Aggregate LongMemEval +10 points across all categories

---

### Phase 7 — Track E + Tenancy P4: Sleep-time consolidation (5–7 days)

**Goal:** Background worker consolidates ended sessions, runs cross-thread reflection, promotes session-scoped memories to project/user.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/consolidation/__init__.py` | New module |
| `attestor/consolidation/consolidator.py` | `SleepTimeConsolidator`: dequeues episodes, re-extracts with stronger model, applies ADD/UPDATE/INVALIDATE |
| `attestor/consolidation/reflection.py` | `REFLECTION_PROMPT` for cross-thread synthesis (stable preferences, contradictions for review) |
| `attestor/consolidation/session_end.py` | `SESSION_PROMOTION_PROMPT` for KEEP_SESSION/PROMOTE_PROJECT/PROMOTE_USER/DISCARD |
| `attestor/identity/sessions.py` | `end_session()` enqueues consolidation job |
| `attestor/cli.py` | `attestor consolidate run` and `attestor consolidate worker --cadence 300` |
| `tests/test_consolidator.py` | Consolidate one episode, verify decisions; reflection produces preference; promotion via supersession |

**Success criteria:**
- 30+ episodes produces ≥ 5 derived `session_summary` and `reflection` memories
- Re-running LongMemEval after 24h consolidation: `multi-session` +5 beyond synchronous baseline
- Contradictions land in human-review queue, not auto-resolved

---

### Phase 8 — Tenancy P2-3 + Track F: Auth, multi-agent, MCP (5–7 days)

**Goal:** HOSTED mode auth middleware, quotas, GDPR delete/export, MCP primitives, provenance signing.

**Deliverables:**

| File | Action |
|---|---|
| `attestor/api.py` | Auth middleware: JWT verify → resolve external_id → user_id; ON CONFLICT user provisioning |
| `attestor/quotas.py` | New: `user_quotas` table, count triggers, enforcement on `add()` and session/project creation |
| `attestor/api.py` | `/export` endpoint (data portability), `/delete-account` endpoint (GDPR) |
| `attestor/audit.py` | New: `deletion_audit` table (RLS-exempt), purge_user() flow |
| `attestor/mcp/server.py` | Add 5 prompts: `record_decision`, `handoff_to`, `resume_thread`, `audit_decision`, `propose_invalidation` |
| `attestor/mcp/prompts/handoff.py` | `HANDOFF_PROMPT_TEMPLATE` |
| `attestor/identity/signing.py` | New: Ed25519 signature on `(id || agent_id || ts || content_hash)`, opt-in via config |
| `tests/test_auth_middleware.py` | JWT verify, audience check, expired token, missing sub claim |
| `tests/test_quotas.py` | Hit limits, 429 response, count consistency under concurrent writes |
| `tests/test_gdpr.py` | Delete user removes all data across stores; audit log persists |
| `tests/test_provenance_signing.py` | Sign + verify round-trip; spoofed signature rejected |

**Success criteria:**
- Multi-agent reference flow (planner → executor → reviewer) runs handoff cleanly
- Provenance verification passes 100% on legitimate writes; deliberate spoof rejected
- GDPR delete passes audit (every store reports 0 rows for user, audit log persists)

---

### Phase 9 — Track G: Eval harness (5–7 days)

**Goal:** Benchmarks in CI; merge blocked on regression.

**Deliverables:**

| File | Action |
|---|---|
| `evals/longmemeval/runner.py` | Full ingest → recall → CoN read → judge pipeline |
| `evals/beam/runner.py` | BEAM 1M token setting |
| `evals/abstention/runner.py` | AbstentionBench with the CoN prompt |
| `evals/regression/qa.yaml` | 50–100 hand-graded domain questions |
| `evals/regression/runner.py` | Runs the YAML cases through the full stack |
| `.github/workflows/evals.yml` | CI: run all evals on every PR; block merge if regression > 2 points |
| `docs/bench/v4-baseline.json` | Initial scores published with each release |

**Success criteria:**
- All 4 benchmarks runnable via single `attestor evals run` command
- CI baseline established and published in README
- A test that intentionally regresses one category by ≥3 points gets blocked at PR

---

### Phase 10 — Track H: Optional improvements (3–7 days, gated on data)

Only do these if benchmarks plateau or design partners ask:
- **H.1** Cross-encoder rerank with `bge-reranker-base`
- **H.2** Entity resolution split into separate prompts
- **H.3** Confidence-aware retrieval (`confidence_gate` per call)
- **H.4** A2A handoff over MCP

---

### Phase 11 — Hardening + v4.0.0 release (2–3 days)

**Deliverables:**
- README rewrite around v4 positioning ("first deterministic, auditable memory tier purpose-built for multi-agent chat systems")
- Migration guide (mostly empty — no v3 data; document API changes)
- `attestor doctor` updated to verify v4 schema, RLS policies, all extensions
- Cut `4.0.0` tag, push to PyPI
- Blog post / changelog highlighting LongMemEval scores

---

## 3. Effort summary

| Phase | Calendar | Critical-path? |
|---|---|---|
| P0 schema | 1–2 days | ✅ blocks everything |
| P1 identity | 3–5 days | ✅ blocks P2+ |
| P2 defaults | 2–3 days | ✅ blocks dev experience |
| P3 ingest (Track A) | 5–7 days | ✅ blocks P5/P7 |
| P4 retrieval (Track B) | 3–5 days | parallel with P3 |
| P5 temporal (Track C) | 3–4 days | depends on P3 |
| P6 reading (Track D) | 1–2 days | independent |
| P7 consolidation (Track E + Tenancy P4) | 5–7 days | depends on P3 |
| P8 auth + MCP (Tenancy P2-3 + Track F) | 5–7 days | depends on P1 |
| P9 evals (Track G) | 5–7 days | depends on P3-P6 |
| P10 optional (Track H) | 3–7 days | gated on data |
| P11 release | 2–3 days | last |

**Sequential total:** ~38–55 days ≈ 8–11 weeks.
**With single-developer parallelism (P4 alongside P3, P6 alongside P5/P7):** ~6–9 weeks.
**Critical path only (P0→P1→P2→P3→P5/P7→P8→P9→P11):** ~30–40 days ≈ 6–8 weeks.

---

## 4. Risk register

| Risk | Mitigation |
|---|---|
| Apache AGE not available on Neon → graph layer disabled | Use Neo4j as graph store for SaaS deployments; document the constraint |
| LLM extraction produces invalid JSON | Strict JSON schema validation + retry with stronger model; never write malformed facts |
| Sleep-time consolidator falls behind under burst load | Backpressure on `episodes_pending_consolidation`; alert when queue depth > N |
| RLS policy bug leaks one user's data | Mandatory isolation test suite (6 tests) on every PR; failure blocks merge |
| Conflict resolver UPDATE loses audit trail | Tests assert UPDATE preserves `id` and creates a new row only on INVALIDATE |
| Bi-temporal `as_of` filter misses rows due to NULL `valid_until` | Always wrap with `COALESCE(valid_until, 'infinity'::timestamptz)` |
| HOSTED mode users without first login can't recall | Auto-provision via `INSERT ... ON CONFLICT (external_id) DO UPDATE RETURNING` on first JWT |
| Cross-encoder rerank adds 80ms p95 latency | Gate behind config flag; default OFF; document as opt-in for accuracy/latency tradeoff |

---

## 5. What we are NOT doing in v4

Pulled forward from roadmap §"What NOT to do":
- No LLM in the critical retrieval path (extraction + consolidation only)
- No targeting > 93% on LongMemEval (gaming territory)
- No bolt-on graph database without multi-hop entity reasoning need
- No taking over the conversation buffer (working memory stays with the agent harness)
- No over-extraction (most rounds yield 0–2 facts)

---

## 6. Decision log

| Decision | Date | Rationale |
|---|---|---|
| Greenfield schema, no v3 backfill | 2026-04-25 | User wiped data; legacy migration complexity not justified |
| Postgres is recommended for SaaS (RLS) | 2026-04-25 | Honest constraint — RLS isn't universal |
| Inbox is a real project (not NULL) | 2026-04-25 | Avoids special-case in queries |
| Daily session pattern for SOLO | 2026-04-25 | Natural for CLI/MCP usage; no state to manage |
| Per-call `_resolve()` chain | 2026-04-25 | Single entry point makes audit / RLS simpler |
| ABSTAIN clause in CoN prompt | 2026-04-25 | AbstentionBench fix — frontier models confabulate without explicit instruction |

---

## 7. First commit (this session)

The smallest atomic shippable unit: **Phase 0 schema + RLS + tests**.

- 1 file: `attestor/store/schema.sql` (greenfield)
- 1 file: `attestor/store/postgres_backend.py` (call new schema; remove v3 init)
- 1 file: `attestor/models.py` (add `MemoryScope`, `User`, `Project`, `Session`)
- 1 file: `tests/test_v4_schema.py` (schema + RLS guards)
- 1 file: `attestor/__init__.py` (bump to `4.0.0a1`)

Roughly 300 lines of code + 200 lines of tests. Half-day. Lands a solid foundation for everything else.
