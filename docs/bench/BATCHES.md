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
| 6 | knowledge-update | 10 | distill + CoT + verify | gpt-5.1 | on | gpt-5.1 | claude-sonnet-4.6 | 9/10 (90%) | 8/10 (80%) | 90% | `d03c79a`† | `3ccc6ac6…` | `longmemeval-attestor-10samp-knowledge-update.json` |
| 5 | temporal-reasoning | **30** | distill + CoT + verify | gpt-5.1 | on | gpt-5.1 | claude-sonnet-4.6 | **25/30 (83.3%)** | **26/30 (86.7%)** | **96.7%** | `d03c79a`† | `4fc0e6b5…` | `longmemeval-attestor-30samp-temporal-gpt51-sonnet46.json` |
| 7 | multi-session | 10 | distill + CoT + verify | gpt-5.1 | on | gpt-5.1 | claude-sonnet-4.6 | 6/10 (60%) | 6/10 (60%) | 100% | `d03c79a`† | `af6431c1…` | `longmemeval-attestor-10samp-multi-session.json` |
| 8 | single-session-preference | 10 | distill + CoT + verify | gpt-5.1 | on | gpt-5.1 | claude-sonnet-4.6 | **4/10 (40%)** | **5/10 (50%)** | 90% | `d03c79a`† | `c9613461…` | `longmemeval-attestor-10samp-single-session-preference.json` |
| 9 | single-session-assistant | 10 | distill + CoT + verify | gpt-5.1 | on | gpt-5.1 | claude-sonnet-4.6 | **2/10 (20%)** | **3/10 (30%)** | 90% | `d03c79a`† | `25cdecea…` | `longmemeval-attestor-10samp-single-session-assistant.json` |

### Diagnosis of batches 8 and 9

The distiller's DISTILL_PROMPT rule #7 **drops assistant-provided facts**
unless the user explicitly committed to acting on them. Rule #9 further
drops assistant turns that "echo" user facts. For single-session-assistant
questions (which ask back the assistant's stated facts, e.g. "what color
was the Plesiosaur the assistant described") this is catastrophic. For
single-session-preference questions (which often surface in assistant
responses rather than user statements) it is significantly degrading.

This is a prompt-tuning bug in the distiller, **not** a memory-layer
bug — retrieval is fine on categories where the fact survives ingest.
Fix is in-flight: relax rule #7 to preserve all factual content,
remove rule #9, explicitly instruct the distiller to retain "the
assistant told the user that X" memories.

† Batch 6 was launched from the pre-provenance code (git `d03c79a`); its
output has a sidecar SHA but no embedded `provenance` block. Reproducible
via the command in `REPRODUCE.md` under the same git SHA.

Note: batches 1–4 and batch 6 were produced **before** the provenance
metadata landed in the runner (commit for that is after `d03c79a`).
They are reproducible via the recipe + git SHA above but the output JSON
does not yet include the embedded `provenance` block — we back-filled
SHA256 sidecars only. Batch 5 onward carry full provenance inside the JSON.

## Running

_none — batches 5 and 6 both finished._

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
