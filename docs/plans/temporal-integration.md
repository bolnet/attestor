# Temporal integration plan — durable governance jobs for Attestor

Status: proposed 2026-09-03. Owner: @bolnet.
Positioning context: Attestor is repositioned as **governed memory for multi-agent
production meshes** (RBAC, signed provenance, per-agent budgets, deterministic
read path, temporal supersession with `as_of`). The Claude Code plugin is an
on-ramp demo, not the headline. Temporal serves that positioning by making every
governance *job* durable, observable, and resumable.

## 1. Why

Attestor already hand-rolls durable-execution machinery, and several governance
jobs have no runner at all:

| Today | Problem | Temporal replacement |
|---|---|---|
| `consolidation/queue.py` (SKIP LOCKED + 600 s lease) + `consolidator.run_forever` | Hand-rolled task queue, lease reclaim, blanket catch-and-sleep retry. Only reachable via `AgentMemory.consolidate()`; no daemon ships. | Workflow + activity retry policy; worker process; UI shows every stuck/failed episode |
| `core/agent_memory.py::add()` vector + graph writes | Non-fatal, warning-only, **no retry, no reconciliation**. Derived state silently drifts. No rebuild/backfill code exists despite CLAUDE.md promising rebuildability. | `DeriveMemory` workflow (idempotent, retried) + `rebuild` command |
| `compliance/retention.py` | Expects "a long-running scheduler" that does not exist; `apply_retention()` is manual | Temporal Schedule → `RetentionSweep` workflow |
| `identity/sessions.py` | Docstring references a background sweeper for idle/ended/archived; none exists | Schedule → `SessionSweep` workflow |
| `compliance/retention.py::forget_user` | Walks doc → vector → graph → state; failures logged, not retried; partial success is terminal | `ForgetUser` saga: per-backend activities with retry, audit-first, per-backend result |
| `mab/retrieval.py`, `llm_trace.py` retry/backoff | Retry policy scattered in call sites | Retries owned by activity policies; LLM clients `max_retries=0` (cookbook rule; matches the 2026-04-30 timeout-cascade lesson) |
| `evals/runner.py` subprocess-per-cell, `evals/build_bench/workflow.js` | Hours-long runs die on 429 caps and restart from zero | Optional: workflow per cell, child per sample, resumable |

## 2. Non-goals (hard rules)

- **Recall never touches Temporal.** The 6-step read path stays in-process, deterministic, sub-100 ms.
- **Hooks never wait on Temporal.** `session_start` / `post_tool_use` / `stop` may *enqueue* via the client but must return immediately; if the server is unreachable they fall back to today's in-process behaviour.
- **Quickstart stays zero-question.** Temporal is an opt-in extra + compose profile. Default `attestor quickstart` is unchanged.
- **Postgres stays the source of truth.** Temporal holds job state only; nothing about a memory lives solely in a workflow history.
- **No bench numbers in the repo** (existing rule).

## 3. Architecture

```
attestor/durable/                # new package ("durable", not "temporal" — attestor/temporal/ is temporal *reasoning*)
  __init__.py
  config.py                      # DurableConfig from configs/attestor.yaml `durable:` block (YAML authoritative)
  client.py                      # connect(); Client cached per event loop; pydantic_data_converter; TEMPORAL_* env passthrough
  dispatch.py                    # fire-and-forget starts from SYNC callers (add()) via one private daemon loop thread
  worker.py                      # `attestor worker` entry; registers all workflows+activities on one task queue
  schedules.py                   # declarative Schedule specs + `apply()` (idempotent create-or-update)
  models.py                      # frozen dataclasses / pydantic models crossing the workflow boundary
  activities/
    consolidation.py             # consolidate_episode(ref: EpisodeRef) -> ConsolidationOutcome (ids in; re-reads the row via consolidate_claimed)
    derive.py                    # embed_and_upsert(ref), graph_extract(ref), list_memory_ids(page) — ids in; re-reads via AgentMemory.derive_*
    retention.py                 # list_due_policies(), apply_policy(policy_id, dry_run)
    forget.py                    # write_forget_audit(), forget_doc(), forget_vector(), forget_graph(), forget_state()
    sessions.py                  # sweep_idle(), sweep_ended(), sweep_archived()
  workflows/
    consolidate.py               # ConsolidateEpisode
    derive.py                    # DeriveMemory
    forget.py                    # ForgetUser (saga)
    retention.py                 # RetentionSweep
    sessions.py                  # SessionSweep
    rebuild.py                   # RebuildDerived (batch sliding window over Postgres ids)
```

Rules inside `workflows/`: import Attestor code only under
`workflow.unsafe.imports_passed_through()`; no `datetime.now()`, no I/O, no
randomness; every activity call carries an explicit `start_to_close_timeout` and
`RetryPolicy`; permanent errors (bad config, missing key, 4xx) raise
`ApplicationError(non_retryable=True)`.

Shared derived-state lanes live in `attestor/core/derive_service.py`
(`_DeriveMixin`, mixed into `AgentMemory`): `add()` and the `DeriveMemory`
activities call the same `_write_vector` / `_extract_graph` / `_write_graph`
code, and the activities re-read the row by id through `derive_vector(id)` /
`derive_graph(id)` — one implementation, no duplicated write logic.

Workflow IDs are deterministic so retries are idempotent:
`derive-{memory_id}`, `consolidate-{episode_id}`, `forget-{external_id}-{audit_id}`,
`retention-sweep-{YYYY-MM-DD}`, `session-sweep-{YYYY-MM-DDTHH:MM}`.

Task queue: `attestor-governance`. Namespace: `attestor` (dev server: `default`).
Multi-tenant note: if one tenant's consolidation can starve others, enable Task
Queue Fairness keyed on `user_id` (skill reference `core/priority-fairness.md`).

### Config (`configs/attestor.yaml`)

```yaml
durable:
  enabled: false                 # opt-in; when false every durable.* call is a no-op fallback to in-process
  address: "localhost:7233"
  namespace: "attestor"
  task_queue: "attestor-governance"
  tls: false
  schedules:
    retention_sweep: "0 3 * * *"       # daily 03:00
    session_sweep: "*/10 * * * *"      # every 10 min
```

Loader: raise loudly if `durable.enabled` is true and `temporalio` is not
installed or the server is unreachable at worker start (existing "fail loudly on
missing config" rule). Runtime callers (`add()`, hooks) degrade to in-process.

### Packaging

- `pyproject.toml` extra: `durable = ["temporalio>=1.32,<2"]` (SDK requires Python ≥ 3.10; repo is ≥ 3.10; venv is 3.13).
- CLI: `attestor worker` (foreground, honours `--task-queue`), `attestor durable schedules apply|list`, `attestor durable rebuild [--since ISO] [--namespace]`, `attestor durable status`.
- Compose profile `durable` in `attestor/infra/local/docker-compose.yml`:
  `attestor_temporal_server` (`temporalio/auto-setup`, `DB=postgres12`,
  `POSTGRES_SEEDS=postgres`, reuses the existing `attestor_postgres_document_db`;
  auto-setup creates `temporal` + `temporal_visibility` DBs) and
  `attestor_temporal_ui` (`temporalio/ui`, port 8233). `attestor quickstart --durable`
  enables the profile; plain quickstart does not.
- Dev without Docker: `temporal server start-dev --db-filename ~/.attestor/temporal.db`.

## 4. Phases

Each phase is one PR, TDD-first, ≥ 80 % coverage on the new package, reviewed by
code-reviewer + security-reviewer before merge. Estimates are working days.

### Phase 0 — Spike (1 day, throwaway branch, no merge)
- `pip install temporalio` in a scratch venv (docs verified via Context7 2026-09-03: SDK 1.32.0).
- `temporal server start-dev` + one worker + `ConsolidateEpisode` wrapping today's
  `_consolidate_one`. Drain 20 real episodes from the local stack.
- Exit criteria: works end-to-end; note any sandbox import failures from Attestor
  modules (psycopg2, neo4j driver, pinecone gRPC) — these dictate what must be
  passed through vs kept in activities.

### Phase 1 — Foundation + consolidation (3 days)
- `attestor/durable/{config,client,worker,models}.py`, extra, CLI `attestor worker`.
- `ConsolidateEpisode` workflow; activity wraps existing per-episode logic (queue
  row claim stays in Postgres for now — zero schema change).
- `SleepTimeConsolidator.run_forever` gains a `durable` branch: instead of
  running in-process it starts one workflow per claimed episode and returns.
- Tests: `WorkflowEnvironment.start_time_skipping()` with mocked activities;
  live test env-gated on `ATTESTOR_TEMPORAL_ADDRESS`. CI caches the test-server
  binary via `test_server_existing_path`.
- Acceptance: kill the worker mid-batch → restart → every episode completes exactly once; failures visible in UI.

### Phase 2 — Derived-state durability (3 days) ← closes the drift gap — IMPLEMENTED 2026-09-03
- `DeriveMemory` workflow: `embed_and_upsert` then `graph_extract`, each retried
  (initial 2 s, backoff 2.0, max 5 min, `maximum_attempts=0`), non-retryable on
  dim-mismatch / auth errors (`activities/derive.py::PERMANENT_ERROR_PATTERNS`,
  plus `DeriveMissing` for a deleted row). A permanent failure in one lane does
  not stop the other; the run then fails with the per-lane outcome in the
  error details so the UI shows which lane drifted.
- `add()`: keep the synchronous fast-path attempt; **on vector or graph failure,
  if durable is enabled, start `derive-{memory_id}` (fire-and-forget) instead of
  only logging**. Trace event `ingest.write.repair_scheduled` (`memory_id`,
  `namespace`, `lanes`, `workflow_id`, `task_queue`). One dispatch per `add()`
  even when both lanes fail; the workflow id dedups against a concurrent
  rebuild. When `durable.enabled` is false the path is unchanged and
  `attestor.durable` is never imported (`tests/test_add_repair_scheduling.py`).
  Sync → Temporal bridge: `durable/dispatch.py` (one daemon loop thread; add()
  never blocks on the RPC; failures are logged, not raised).
- `RebuildDerived` workflow + `attestor durable rebuild [--since ISO]
  [--namespace NS] [--page-size N] [--window N] [--wait]`: keyset pages over
  Postgres ids (`_postgres_document.list_memory_ids`, active rows, ordered by
  id, heartbeating), child `DeriveMemory` per id with at most `window` in
  flight, continue-as-new every `max_pages_per_run` pages carrying the cursor +
  counters (`RebuildProgress`), permanent child failures counted + ids
  reported (capped), already-running children skipped. `--namespace` is the
  MEMORY namespace; `--temporal-namespace` overrides the server namespace.
- Tenant scope (review fix 2026-09-03): the derive seam runs **as the row's
  owner**, mirroring `consolidator.py`. `DeriveRef` carries `user_id` +
  `agent_id` (ids only); `derive_vector` / `derive_graph` set the Postgres RLS
  session user (`_set_rls_user`) to it before the re-read, pass `agent_id` as
  the `visibility='private'` requester, refuse a row owned by another user
  (a BYPASSRLS worker role could otherwise read it), and reset the session to
  the row's owner before every derived write. One rebuild run = one tenant:
  `attestor durable rebuild --user <id>`; `list_memory_ids` sets RLS and
  filters by `user_id`. A multi-tenant store with no `--user` fails loudly
  (non-retryable `DerivePermanent`); SOLO installs resolve their default user
  and echo it in `MemoryIdPage.user_id` so every child carries an explicit
  owner. v3 stores (no `user_id` column, no RLS) stay unscoped. Guarded by
  `tests/test_derive_tenant_scope.py`.
- Acceptance (manual, live stack): stop Pinecone Local, add 50 memories, start
  Pinecone → all 50 vectors present within retry window with no manual action;
  `rebuild` from an empty Pinecone index reproduces recall results
  byte-for-byte. Unit + time-skipping coverage: `tests/test_durable_derive_*`,
  `tests/test_durable_rebuild_workflow.py`, `tests/test_durable_dispatch_sync.py`,
  `tests/test_postgres_list_memory_ids.py` (live, POSTGRES_URL).

### Phase 3 — Governance jobs (4 days) ← the positioning proof points — IMPLEMENTED 2026-09-03
- `ForgetUser` saga (`workflows/forget.py`): `write_forget_audit` first
  (append-only, RLS-exempt; if it fails permanently NO backend is touched),
  then `forget_doc`, then `forget_vector` / `forget_graph` / `forget_state`
  concurrently — each its own activity with unlimited retries, so a Neo4j
  outage keeps only the graph lane retrying. Lanes live in
  `attestor/compliance/forget_lanes.py` (plain SQL + backend calls, no
  Temporal): the audit row is inserted under a caller-minted `audit_id`
  with `ON CONFLICT (id) DO NOTHING`, so an activity retry never writes a
  second audit row, and every lane rolls back before re-raising. Workflow
  id `forget-{user_id}-{audit_id}` (the caller mints the audit id so the
  id is known at start). `ForgetUserOutcome` carries per-backend status /
  counts / `skipped` (backend absent or no `delete_by_user`) and terminal
  errors; a partial run FAILS with the outcome in the error details
  (regulator-visible in the UI) and exposes it via the `progress` query.
  `AgentMemory.forget_user()` starts the saga and returns
  `{"workflow_id", "audit_id", "durable": True, "follow": ...}` when
  durable is on (`core/forget_service.py`, start RPC awaited — an
  unreachable server raises, no silent in-process fallback for a GDPR
  delete); `dry_run` and `durable.enabled: false` keep the in-process path
  byte-for-byte. `attestor durable status forget-<id>` follows it.
- `RetentionSweep` (`list_due_policies` → `apply_policy` per policy, via
  `apply_retention(policy_ids=(id,))`; sequential; a permanently failing
  policy is recorded and the rest still run) and `SessionSweep`
  (`SessionRepo.sweep_idle` / `sweep_ended` / `sweep_archived` — pure SQL
  batch transitions, thresholds on `SessionSweepRequest`; the sweeping
  connection must own the table or hold `BYPASSRLS`).
- Schedules (`durable/schedules.py`): `attestor-retention-sweep` and
  `attestor-session-sweep`, cron strings from `durable.schedules`,
  overlap policy SKIP, workflow-id prefixes `retention-sweep` /
  `session-sweep` (Temporal appends the tick). `attestor durable
  schedules apply` is create-or-update (`Client.create_schedule`, then
  `handle.update` on `ScheduleAlreadyRunningError`) and converges;
  `schedules list` shows presence / paused / next run.
- Acceptance (`tests/test_durable_forget_workflow.py`,
  `tests/test_durable_retention.py`, `tests/test_session_sweep.py`,
  `tests/test_durable_schedules.py`): audit row precedes every delete;
  graph outage (4 transient failures) keeps retrying while doc + vector
  complete; permanent audit failure touches no backend; retention fires
  at 03:00 — proven by backfilling one day on the local dev server
  (`WorkflowEnvironment.start_local`, `temporal_dev_env` fixture), since
  the time-skipping test server does not implement the Schedule service
  (`CreateSchedule is unimplemented`, verified 2026-09-03). The fired
  workflow id is `retention-sweep-<day>T03:00:00Z`.
- Review hardening (2026-09-03): the worker's activity pool shares ONE
  `AgentMemory` / psycopg2 connection, so every SQL path serialises on the
  connection's re-entrant lock (`attestor/store/conn_lock.py`, keyed by
  connection object — `PostgresBackend._execute`, the forget lanes, the
  retention helpers and `SessionRepo` all hold the SAME lock), and the
  derive seam holds it across its whole RLS scope-set → read → ownership
  check → write unit (`tests/test_conn_lock.py`,
  `tests/test_derive_tenant_scope.py`). Activity failure messages are
  redacted + capped and raised `from None` before they reach workflow
  history (`durable/activities/_sanitize.py`,
  `tests/test_durable_error_sanitize.py`). Whole-user forget for agents
  goes through `AgentContext.forget_user` (RBAC `FORGET`);
  `AgentMemory.forget_user` stays the trusted admin path and must not be
  exposed on a CLI / API / MCP surface without an equivalent gate.

### Phase 4 — Retry consolidation (2 days)
- Move backoff out of `mab/retrieval.py` and `llm_trace.py` call sites that run
  inside activities; set `max_retries=0` on LLM/embedder clients used by
  activities. In-process (non-durable) paths keep their current behaviour.

### Phase 5 — Bench runner (optional, 3 days)
- `evals/durable_runner.py`: workflow per matrix cell, child workflow per sample,
  heartbeats on ingest, resumable after 429s. Braintrust `braintrust[temporal]`
  plugin for tracing. Does not replace `evals/runner.py`; it is an alternative
  `--durable` mode.

### Phase 6 — Docs + positioning (1 day, after Phase 3)
- `docs/INSTALL.md`: "Governance jobs (Temporal)" chapter; runtime topologies
  Mode B/C gain a worker box.
- README / `docs/index.html` / PyPI description: lead with governed multi-agent
  memory; proof points = RBAC matrix, signed provenance, `as_of` replay,
  auditable forget saga, deterministic read path, durable governance jobs with a
  UI. Quickstart / Claude Code plugin move below the fold as the on-ramp.
- CLAUDE.md: add `durable/` to the tree and the hot-path rule.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Another service to run | Opt-in extra + compose profile; shares the existing Postgres; dev server is one binary |
| Workflow sandbox rejects Attestor imports (drivers, pydantic v1 shims) | Phase 0 spike finds them; keep all I/O in activities; `imports_passed_through` only for pure models |
| Test-server binary download in CI | Cache + `test_server_existing_path`; live tests env-gated |
| Name confusion `attestor/temporal` vs Temporal | Package named `durable`; docs say "Temporal (durable execution)" on first use |
| Duplicate derived writes (sync path succeeded *and* repair ran) | Vector upsert and graph MERGE are idempotent by memory id; workflow id dedups; `derive_vector` re-embeds the exact `add()` payload (stored `_context_prefix` included) |
| `add()` blocks or fails on Temporal when repair is scheduled | `dispatch.schedule_derive` returns the workflow id immediately; the start RPC runs on a daemon loop thread; connect / start errors are logged as warnings and the memory is still returned |
| Rebuild history unbounded on large stores | `RebuildDerived` continue-as-new every `max_pages_per_run` pages; sliding window bounds in-flight children; list activity heartbeats |
| Scope creep into recall | Non-goal enforced by review checklist + a test asserting recall never imports `attestor.durable` |
| Conversation text leaks into Temporal event history (a second, un-governed store outside the retention sweep / forget saga) | Workflow + activity payloads carry **ids only** (`EpisodeRef`); the activity re-reads turn text from the claimed Postgres row via `ConsolidationQueue.fetch_claimed(id, user_id)`. Guarded by `tests/test_durable_models.py` (no content fields on the boundary) and `tests/test_durable_dispatch.py` (dispatch payload has no turn text). Outcomes carry memory ids + counts, never fact text. Future ForgetUser / RetentionSweep workflows keep the same rule, and the namespace retention window is the only history-side control needed |

## 6. Local analysis references

Cloned 2026-09-03 for this plan (session scratchpad, not persisted):
`temporalio/sdk-python` (1.32.0), `samples-python` (`batch_sliding_window`,
`schedules`, `polling`, `external_storage`, `litellm_activity`),
`ai-cookbook` (`mcp/hello_world_durable_mcp_server`, `foundations/http_retry_enhancement_python`),
`skill-temporal-developer` (`references/core/{patterns,ai-patterns,priority-fairness}.md`),
`docker-compose` (`docker-compose-postgres.yml`).
Consider installing `skill-temporal-developer` as a Claude Code skill for the implementation PRs.
