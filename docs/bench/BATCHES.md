# LongMemEval Batches — Attestor

Ledger of all LongMemEval runs. Each row is one `attestor longmemeval` invocation.
**Every completed row is a verifiable artifact** — see `REPRODUCE.md` for how
to check integrity and re-run.

**Status legend:** 🟢 complete · 🟡 running · 🔴 killed / failed · ⚪ queued

**Recipe shorthand:**
- `raw` — ingest turns as-is (`[YYYY-MM-DD] Role: text`), no LLM distillation
- `distill` — per-turn `gpt-5.1` distillation before storage (DISTILL_PROMPT, 10 rules)
- `CoT` — `ANSWER_PROMPT` has the `<reasoning>…</reasoning>` CoT + date-arithmetic rubric
- `verify` — second-pass verifier re-checks first answer against same facts
- Judges are listed as `answer-model-agnostic`; they only see (question, gold, generated, category)

**Verifiability**
- Every output JSON embeds `provenance` (git SHA, dataset SHA256, argv,
  UTC timestamps, model IDs) and `run_config` (every knob). See
  [`REPRODUCE.md`](./REPRODUCE.md).
- Every output JSON has a sidecar `<file>.sha256` you can diff against the
  content hash to detect tampering.
- Every completed row below carries its output hash below so a reader can
  verify without downloading extras.

**All batches use:**
- Dataset: `longmemeval_s_cleaned.json` (HuggingFace `xiaowu0162/longmemeval-cleaned`)
- Backend: Attestor (Postgres + Neo4j) via `~/.attestor/config.json`
- Embeddings: `openai/text-embedding-3-small` (1536d)
- Parallelism: `--parallel 8`
- Date: 2026-04-23

---

## Completed

| # | Category | n | Recipe | Answer | Verify | Judge A | Judge B | Judge A acc | Judge B acc | Agreement | Git SHA | Output SHA256 | Output |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | single-session-user | 50 | raw | gpt-4.1-mini | off | gpt-4.1-mini | claude-haiku-4.5 | 47/50 (94%) | 44/50 (88%) | 94% | `6975396` | `323729ed…` | `longmemeval-attestor-50sample-parallel8-dualjudge.json` |
| 2 | temporal-reasoning | 10 | raw | gpt-4.1-mini | off | gpt-4.1-mini | claude-haiku-4.5 | 6/10 (60%) | 7/10 (70%) | 90% | `859f62b` | `1dbbd7db…` | `longmemeval-attestor-10samp-temporal.json` |
| 3 | temporal-reasoning | 10 | raw + CoT + verify | gpt-5.1 | on | gpt-4.1-mini | claude-haiku-4.5 | 8/10 (80%) | 7/10 (70%) | 90% | `f3c50aa` | `3923b0ab…` | `longmemeval-attestor-10samp-temporal-cot-verify-gpt51.json` |
| 4 | temporal-reasoning | 10 | **distill** + CoT + verify | gpt-5.1 | on | gpt-4.1-mini | claude-haiku-4.5 | 8/10 (80%) | 9/10 (90%) | 90% | `d03c79a` | `b11b17f1…` | `longmemeval-attestor-10samp-temporal-distill-gpt51.json` |

Note: batches 1–4 were produced **before** the provenance metadata
landed in the runner. They are reproducible via the recipe + git SHA
above but the output JSON does not yet include the embedded
`provenance` block — we back-filled SHA256 sidecars only. Batches 5+
carry full provenance inside the JSON.

## Running

| # | Category | n | Recipe | Answer | Verify | Judge A | Judge B | Output | Log | Started |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | temporal-reasoning | 30 | distill + CoT + verify | gpt-5.1 | on | **gpt-5.1** | **claude-sonnet-4.6** | `longmemeval-attestor-30samp-temporal-gpt51-sonnet46.json` | `lme_temporal_30_sonnet.log` | 2026-04-23 17:42 |
| 6 | knowledge-update | 10 | distill + CoT + verify | gpt-5.1 | on | gpt-5.1 | claude-sonnet-4.6 | `longmemeval-attestor-10samp-knowledge-update.json` | `lme_knowledge_10.log` | 2026-04-23 ~18:00 |

## Queued

_none_

---

## Per-sample diagnoses (for reproducibility)

### Temporal samples that flipped with distillation

| sample_id | Raw baseline | CoT+verify | Distill+verify | Notes |
|---|---|---|---|---|
| `gpt4_59149c77` (MoMA↔Met days) | both WRONG (171 vs 7–8) | both CORRECT | both CORRECT | Distillation normalized visit dates → retrieval tightened |
| `0bc8ad92` (months since museum) | both WRONG (4 vs 5) | both CORRECT | both CORRECT | CoT forced explicit calendar-month math |
| `b46e15ed` (months since charity) | gpt WRONG claude CORRECT | gpt CORRECT claude WRONG | **both CORRECT** | Verify flipped abstention → concrete answer |
| `af082822` | both CORRECT | both WRONG (DB-state regression) | both CORRECT | DB truncate between runs fixes carryover |

### Persistent temporal failure

| sample_id | All runs | Why |
|---|---|---|
| `9a707b81` (baking class days ago when making cake) | WRONG across raw / CoT+verify / distill+verify | Anchor-parsing: "when I made the cake" should anchor the question, but the answerer keeps treating `question_date` as anchor. Needs a dedicated prompt rule or question-rewriting pre-pass. |

---

## Operational notes

- **DB truncation between runs** (TRUNCATE `memories` CASCADE + Neo4j DETACH DELETE) is required to avoid stale-namespace carryover. We saw a regression in batch 3 that disappeared after truncation.
- **`~/Documents/attestor/.env`** still has `PG_CONNECTION_STRING=postgresql://memwright:memwright@...` (stale). We use `/tmp/lme.env` with the corrected Postgres + OpenRouter + Neo4j env vars.
- **`~/.attestor/config.json`** must list `backends=[postgres, neo4j]` with explicit `auth.username / auth.password` — `CloudConnection.from_config` does NOT parse passwords from the `url` field.
- Batches 5 and 6 run in parallel on the same DB; per-sample namespaces (`lme_<qid>`) prevent collision.

---

## Cost summary

| Batch | Est. cost | Actual | Notes |
|---|---|---|---|
| 1 | $4 | ~$4 | single-session-user is easiest/cheapest category |
| 2 | $1 | ~$1 | baseline, gpt-4.1-mini everywhere |
| 3 | $3 | ~$3 | gpt-5.1 answer + verify adds spend |
| 4 | $5 | ~$5 | distillation dominates (500 turns × gpt-5.1) |
| 5 | $15 | in flight | n=30, sonnet-4.6 judge adds pennies |
| 6 | $5 | in flight | running in parallel |

**Running total: ~$30.** Budget cap is $200.
