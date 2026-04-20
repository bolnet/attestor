# Plan — Close the LOCOMO SOTA gap (89.0% → 92%+)

**Status:** Draft 2026-04-20
**Baseline:** 87.67% on 3×50 (expt `locomo-phase3-scale-1e3ddb6f`)
**With `_question_entities` fix (uncommitted):** projected ~89.0% (+1.33pp, validated on 28 prior failures)
**SOTA reference:** MemMachine 91.69%, Mem0 91.6%, Zep 75.14%

Attack the three gaps from the 2026-04-20 failure audit in priority order, each gated by a targeted 3×50 rerun before moving on.

---

## Phase 1 — Extractor: per-item atomic facts
**Target:** Pattern-B list collapse (67% of all failures; ~13 of the 28)

### Problem
Extractor collapses enumerated lists ("I loved Charlotte's Web, Nothing is Impossible") into one concept row. List-type gold answers have no atomic row to retrieve.

### Changes
- `attestor/extraction/llm_extractor.py` — extractor prompt must emit one triple per list item: `(Melanie, read, Charlotte's Web)`, `(Melanie, read, Nothing is Impossible)`. Tag each with `kind=list_item` and share the `source_quote`.
- `evals/braintrust_locomo.py::_dual_path_ingest` B.1 — loosen dedup so list items with same subject+predicate but different object aren't merged.
- Keep concept/profile rows unchanged (still useful for summary Qs).

### Verification
Rerun `--only-file logs/phase3_scale_failures.json`. **Target:** ≥8 of the 13 Pattern-B failures move to ≥0.5.
**Budget:** 1–2 days.

---

## Phase 2 — Temporal graph + week-relative normalization
**Target:** "sunday before X" / weekend-relative class (~6 failures)

### Problem
We store `event_date` as string only. Questions like "the Sunday before 3 July 2023" never match because there is no week-anchor reasoning.

### Changes
- Postgres migration: add `event_week_start DATE`, `event_weekday SMALLINT`, `event_iso_year_week TEXT` (computed from `event_date`).
- Extractor: emit temporal-anchor triples on dated facts: `(event, occurred_on_weekday, sunday)`, `(event, in_week, 2023-W26)`.
- Neo4j: `:TimeAnchor {date, weekday, iso_week}` nodes + `OCCURRED_ON`, `OCCURRED_IN_WEEK` edges.
- Orchestrator: detect temporal relative phrases ("sunday before X", "the weekend of X", "two weekends before X") and expand into `iso_week` BFS around the anchor.

### Verification
Targeted rerun on temporal-class failures (6 in the fail list). **Target:** ≥4 flip to ≥0.5.
**Budget:** 3–4 days.

---

## Phase 3 — Entity resolution
**Target:** reduce compounding graph errors (2 current regressions + latent surface-form duplicates)

### Problem
"Melanie" vs "Mel" vs "mom" are 3 distinct `:Entity` nodes. Per 2026 GraphRAG reviews, ER is THE compounding failure mode ("entity disambiguation errors compound exponentially"; production systems with proper ER jumped 43% → 91%).

### Changes
- Post-ingest ER pass: for each new `:Entity`, vector-lookup top-3 entity-profile embeddings within same session; merge if cosine ≥0.85 AND shared speaker context.
- Add `canonical_key` column on `memories` + `:CanonicalEntity` supernode in Neo4j.
- Orchestrator `_question_entities` resolves to canonical before BFS.

### Verification
Full 3×50 rerun. **Target:** 2 regressed Qs recover AND net ≥+1pp over Phase 2 baseline.
**Budget:** 2–3 days.

---

## Cross-cutting

- **Commit the `_question_entities` fix now** (validated +7.2pp on 28 failures) so subsequent phases layer on a stable baseline.
- **Commit the `--only-file` eval flag** + `logs/*_delta.json` diff script as a proper subcommand on `braintrust_locomo.py`.
- **Decision gates:** if any phase fails to move its target class, stop and re-audit before moving on.

## Sequencing

| Step | Action | Time |
|---|---|---|
| 0 | Commit orchestrator fix + `--only-file` flag | 30 min |
| 1 | Phase 1 extractor → rerun → freeze baseline | 1–2 days |
| 2 | Phase 2 temporal → rerun → freeze | 3–4 days |
| 3 | Phase 3 ER → full 3×50 rerun | 2–3 days |
| 4 | Scale to 5×50 on fresh convs (generalization check) | 1 day |

**Expected landing:** 92–93% on 3×50 — closes most of the gap to Mem0/MemMachine.

## Not in scope (deferred)

- Chunk-level Louvain communities (GraphRAG-V, 2026)
- Token-budget trimming to <7K (Mem0 is at <7K @ 91.6%)
- Agentic chunking

These are second-order once Phase 1–3 land.

## References

- [T-GRAG: Dynamic GraphRAG for Temporal Knowledge](https://arxiv.org/pdf/2508.01680) — bi-level time graphs, +19.31% on temporal queries
- [TimeRAG](http://playbigdata.ruc.edu.cn/dou/publication/2025_CIKM_TimeRAG.pdf) — fine-grained timestamps + duration + confidence
- [GraphRAG Entity Disambiguation failure mode](https://www.sowmith.dev/blog/graphrag-entity-disambiguation)
- [GraphRAG-V: chunk-community multi-hop retrieval](https://link.springer.com/chapter/10.1007/978-3-032-14107-1_1)
- [MemMachine v0.2 LOCOMO results (91.69%)](https://memmachine.ai/blog/2025/12/memmachine-v0.2-delivers-top-scores-and-efficiency-on-locomo-benchmark/)
- [Mem0 State of AI Agent Memory 2026 (91.6%)](https://mem0.ai/blog/state-of-ai-agent-memory-2026)
- [Proposition Chunking 2026](https://medium.com/@dhruv-panchal/proposition-chunking-why-you-should-stop-indexing-paragraphs-60b7f8f165b7)
- Internal: `.claude/projects/-Users-aarjay-Documents-agent-memory/memory/project_retrieval_failure_audit.md`
- Internal: `logs/phase3_scale_failures.json` (28 fail list), `logs/phase3_fails_retry_delta.json` (per-Q delta)
