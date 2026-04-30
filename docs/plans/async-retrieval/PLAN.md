# Async Retrieval — Audit-Preserving Plan

**Status:** RED phase — tests written, code not changed.
**Branch:** `feat/async-retrieval-plan-and-tests` (this PR ships PLAN + failing tests).
**Goal:** Reduce recall latency by parallelizing independent steps WITHOUT weakening any of Attestor's eight audit guarantees.

---

## 1. Goal & non-goals

### Goal

Cut typical recall latency from ~1.5–3.5 s (with HyDE) to ~700 ms–1 s by running independent operations concurrently. Specifically:

- HyDE LLM call ‖ original-question vector embed
- Per-lane vector searches (HyDE / multi-query) in parallel
- Vector lane ‖ BM25 lane ‖ graph candidate fetch
- Graph BFS ‖ document fetch
- Self-consistency K-fanout (K parallel answerer calls)

### Non-goals (this plan)

- **No write-side async.** All `add()`, `update()`, supersession writes stay synchronous. Audit chain depends on serial write ordering.
- **No DB driver migration in this plan.** Keep `psycopg2` (sync) wrapped via `run_in_executor`. asyncpg is its own multi-week project.
- **No public API change for embedded users.** `AgentMemory.recall()` remains callable from sync code via a `recall_sync()` shim.

---

## 2. Audit invariants that must hold

All seven phases must preserve every one of these, verified by tests in `tests/async/test_audit_invariants_under_async.py`:

| # | Invariant | Test |
|---|---|---|
| **A1** | `recall(as_of=X)` returns the same memories regardless of concurrent writes | `test_as_of_replay_under_concurrent_writes` |
| **A2** | `t_created` ceiling per recall — nothing visible inside a recall is newer than its `recall_started_at` | `test_recall_started_at_ceiling` |
| **A3** | Provenance fields immutable per memory_id | `test_provenance_immutable_across_concurrent_reads` |
| **A4** | Supersession decisions are linearizable per memory_id (no parallel `superseded_by` writes) | `test_supersession_serial_per_memory_id` |
| **A5** | Trace events reconstructable into a per-recall tree (recall_id + seq + parent_event_id) | `test_trace_reconstructable_under_async` |
| **A6** | RLS still enforced — `attestor.current_user_id` propagates correctly through async tasks | `test_rls_propagates_through_async_tasks` |
| **A7** | HyDE/multi-query LLM determinism — same query → same enrichment → same RRF result (temperature=0) | `test_hyde_temperature_zero_determinism` |
| **A8** | Deletion audit (`deletion_audit` table) gets the row even if the deletion executes async | `test_deletion_audit_under_async` |

---

## 3. Phases (in dependency order)

| # | Phase | Files | LOC | Risk |
|---|---|---|---|---|
| **P1** | HyDE LLM ‖ original-question embed | `attestor/retrieval/hyde.py` | ~50 | Low |
| **P2** | Per-lane parallel inside HyDE/multi_query | `hyde.py`, `multi_query.py` | ~60 | Low |
| **P3** | Trace schema bump (`recall_id`, `seq`, `parent_event_id`) | `attestor/trace.py` + 3-4 callers | ~80 | Low |
| **P4** | Self-consistency K-fanout | `attestor/longmemeval_consistency.py` | ~40 | Low |
| **P5** | `recall_started_at` ceiling primitive | `core.py`, `orchestrator.py`, store backends | ~120 | Medium |
| **P6** | Vector ‖ BM25 ‖ graph parallel in orchestrator | `orchestrator.py` (~1500 LOC today) | ~250-400 | High |
| **P7** | Graph BFS ‖ doc fetch | orchestrator + backends | ~60 | Medium |

P1 and P2 are independent of every other phase. They ship first as the proving ground.

---

## 4. Test coverage matrix

### Phase 1 — HyDE LLM ‖ original embed (`tests/async/test_hyde_async.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_hyde_search_async_runs_lanes_concurrently` | The 2 awaitables (LLM + embed) run via `asyncio.gather`, total wallclock ≤ max(LLM, embed) + ε | RED |
| `test_hyde_search_async_returns_same_result_as_sync` | `await hyde_search_async(q, ...)` produces the same merged hits as `hyde_search(q, ...)` given identical inputs | RED |
| `test_hyde_search_async_handles_lane_timeout` | If HyDE LLM exceeds timeout, fall back to single-lane (original-only) without raising | RED |
| `test_hyde_search_async_handles_lane_exception` | If the HyDE LLM call raises, the original-question lane still produces hits | RED |
| `test_hyde_async_preserves_temperature_zero` | The `traced_create` call still passes `temperature=0.0` | RED |

### Phase 2 — Multi-query lane parallelism (`tests/async/test_multi_query_async.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_multi_query_async_parallelizes_n_lanes` | N+1 vector_search calls overlap; wallclock ≤ max(per-lane) + ε, NOT sum | RED |
| `test_multi_query_async_preserves_RRF_order` | Async fan-out produces the same RRF-ranked output as the sync version | RED |
| `test_multi_query_async_handles_partial_lane_failure` | If one paraphrase's vector search raises, the remaining lanes still RRF-merge | RED |

### Phase 3 — Trace schema (`tests/async/test_trace_async.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_trace_event_includes_recall_id` | Every event emitted within a recall has a `recall_id` field | TODO |
| `test_trace_event_includes_monotonic_seq` | `seq` is monotonic per `recall_id` | TODO |
| `test_trace_event_parent_id_under_async_gather` | Events spawned inside `asyncio.gather` carry the parent's `event_id` | TODO |
| `test_trace_backward_compat_pre_async_format` | Old logs without `recall_id` still parse | TODO |

### Phase 4 — Self-consistency K-fanout (`tests/async/test_self_consistency_async.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_self_consistency_async_runs_K_concurrently` | Wallclock ≈ slowest-call, NOT K × per-call | TODO |
| `test_self_consistency_consensus_order_independent` | Vote result independent of completion order | TODO |
| `test_self_consistency_handles_K_minus_one_failures` | K-1 calls fail, the surviving 1 is the consensus | TODO |

### Phase 5 — `recall_started_at` ceiling (`tests/async/test_recall_started_at_ceiling.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_recall_started_at_excludes_writes_after_ceiling` | Memory inserted at t=ceiling+ε is NOT in recall result | TODO |
| `test_recall_started_at_propagates_to_postgres` | Postgres query uses ceiling as `t_created <= ceiling` | TODO |
| `test_recall_started_at_propagates_to_pinecone` | Pinecone metadata filter applies ceiling | TODO |
| `test_recall_started_at_propagates_to_neo4j` | Cypher query bounds graph reads by ceiling | TODO |

### Phase 6/7 — Orchestrator parallelism (`tests/async/test_orchestrator_async.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_orchestrator_async_parallelizes_three_lanes` | Vector ‖ BM25 ‖ graph candidate fetch overlap | TODO |
| `test_orchestrator_async_handles_one_lane_failure` | RRF degrades to remaining lanes when one fails | TODO |
| `test_graph_bfs_and_doc_fetch_concurrent` | Graph BFS + Postgres doc fetch run in parallel after RRF | TODO |

### Audit invariants — cross-cutting (`tests/async/test_audit_invariants_under_async.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_as_of_replay_under_concurrent_writes` | Hammer recall(as_of=X) while writer coroutine adds memories; recall result stable | RED (P1 scope) |
| `test_supersession_serial_per_memory_id` | Two concurrent supersession attempts on same memory_id linearize | TODO |
| `test_trace_reconstructable_under_async` | 10 concurrent recalls; per-recall trace tree is reconstructable | TODO |
| `test_rls_propagates_through_async_tasks` | Each async task sees its parent's `attestor.current_user_id` | TODO |
| `test_hyde_temperature_zero_determinism` | Same query → same hypothetical → same RRF order across N runs | RED (P1 scope) |

### Performance regression guards (`tests/async/test_async_perf_regression.py`)

| Test | What it asserts | Status |
|---|---|---|
| `test_async_overhead_under_50ms_no_op_recall` | Bare async event-loop tax on a recall with no real work < 50 ms | RED (P1 scope) |
| `test_sync_recall_shim_works` | Calling the sync `recall_sync()` from non-async code returns identical result | TODO |

---

## 5. RED → GREEN → REFACTOR ordering

This PR ships **PLAN.md + the RED tests for P1 + P2 + the P1-scope audit invariants**. All tests fail because async paths don't exist yet.

Subsequent PRs (one per phase):
1. **PR-1:** P1 implementation (`hyde_search_async`) — turn the P1 RED tests GREEN. ~50 LOC.
2. **PR-2:** P2 implementation (per-lane gather in HyDE + multi_query) — turn P2 tests GREEN. ~60 LOC.
3. **PR-3:** P3 trace schema bump — turn P3 RED tests GREEN. Schema-only, no perf change.
4. **PR-4:** P4 self-consistency K-fanout. Independent of orchestrator.
5. **PR-5:** P5 `recall_started_at` ceiling primitive. Foundation for P6/P7.
6. **PR-6:** P6 orchestrator parallel reads. The big one.
7. **PR-7:** P7 graph BFS ‖ doc fetch. Smallest, ships last.

After every GREEN PR, run the full LME-S 30q diagnostic with HyDE+BM25 hybrid and compare wallclock to the pre-async baseline. Acceptance is < 80 % of baseline wallclock with no recall@K regression.

---

## 6. Acceptance criteria for the whole effort

- [ ] All audit invariants A1–A8 hold (tests green)
- [ ] `tests/async/` covers ≥ 80 % of the new async code paths
- [ ] LME-S 30q HyDE+BM25 wallclock < 50 % of pre-async baseline (we measured ~60 s today; target < 30 s)
- [ ] Self-consistency K=5 wallclock < 2× per-call latency (today: 5×)
- [ ] Sync `recall_sync()` shim still works for embedded callers without an event loop
- [ ] No regression in `test_as_of_replay.py` (the load-bearing audit test)
- [ ] Connection pool sizes documented + sized for expected concurrency (Postgres ≥ 20, Pinecone client default, Neo4j ≥ 10)

---

## 7. Open questions to resolve before P5/P6 ship

- **Trace storage cost.** Adding `recall_id` + `seq` + `parent_event_id` is ~30 bytes/event × 6 events/recall = ~180 bytes/recall. At 1k recalls/day × 30 days = ~5 MB/month. Acceptable; not worth a schema change.
- **Connection pool sizing.** Need a pre-flight check in `attestor doctor` that warns if pool < 2× expected concurrency.
- **`recall_started_at` clock source.** Use `datetime.now(timezone.utc)` (matches Postgres `NOW()` semantics) NOT `time.monotonic()` (no wall-clock anchor). The test in `test_recall_started_at_ceiling.py` pins this.
- **Cancellation propagation.** `asyncio.gather(..., return_exceptions=True)` is the default — we never want one slow lane to cancel the others. Document this in the lane closure utility.
