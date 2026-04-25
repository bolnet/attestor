# Reproducing Attestor's LongMemEval Benchmark

Every run in [`BATCHES.md`](./BATCHES.md) is designed to be reproducible by a
third party with sufficient API credits. This file documents the exact steps.

## What an output JSON contains

Each `longmemeval-attestor-*.json` carries, in addition to the per-sample
results, a `provenance` block:

```json
{
  "provenance": {
    "git_sha":           "<commit hash of attestor at run time>",
    "git_dirty":         false,
    "attestor_version":  "3.x.x",
    "python_version":    "3.13.12",
    "platform":          "darwin",
    "argv":              ["python", "-m", "attestor.cli", "longmemeval", "..."],
    "dataset_path":      "/abs/path/to/longmemeval_s_cleaned.json",
    "dataset_sha256":    "<SHA256 of the dataset file>",
    "dataset_sample_count": 500,
    "started_at_utc":    "2026-04-23T…+00:00",
    "completed_at_utc":  "2026-04-23T…+00:00"
  },
  "run_config": {
    "answer_model":       "openai/gpt-5.1",
    "judge_models":       ["openai/gpt-5.1", "anthropic/claude-sonnet-4.6"],
    "use_distillation":   true,
    "distill_model":      "openai/gpt-5.1",
    "verify":             true,
    "parallel":           8,
    "budget":             4000,
    ...
  },
  "schema_version": "1.0"
}
```

Each output JSON also has a sidecar `.sha256` file:

```
$ shasum -a 256 longmemeval-attestor-*.json
b11b17f175501b8f…  longmemeval-attestor-10samp-temporal-distill-gpt51.json
```

Anyone can verify the file hasn't been edited by recomputing the hash and
comparing to the sidecar. If you distrust the sidecar, the BATCHES.md ledger
also records each row's output hash.

## Prerequisites

- Python 3.13+
- Docker (for local Postgres + Neo4j)
- An OpenRouter API key with sufficient credit for ~$200 (full 500-sample run)
- The LongMemEval dataset from HuggingFace:
  ```
  curl -L -o longmemeval_s_cleaned.json \
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json
  ```

Expected SHA256 of the dataset file:
```
$ shasum -a 256 longmemeval_s_cleaned.json
<recompute — our runs pin this in provenance.dataset_sha256>
```

## Local infrastructure

```bash
cd attestor/attestor/infra/local
docker compose up -d postgres neo4j
```

Defaults: Postgres `postgres:attestor@localhost:5432/attestor`; Neo4j
`neo4j:attestor@bolt://localhost:7687`.

## Configuration

Create `~/.attestor/config.json`:

```json
{
  "backends": ["postgres", "neo4j"],
  "default_token_budget": 2000,
  "min_results": 3,
  "postgres": {
    "url": "postgresql://localhost:5432",
    "database": "attestor",
    "auth": {"username": "postgres", "password": "attestor"}
  },
  "neo4j": {
    "url": "bolt://localhost:7687",
    "auth": {"username": "neo4j", "password": "attestor"}
  }
}
```

Create an env file (e.g. `/tmp/lme.env`):

```
OPENROUTER_API_KEY=sk-or-v1-...
```

## Run commands

Every row in BATCHES.md lists its exact CLI. These are the canonical forms.

### Kill-switch (temporal-reasoning, n=30)

```bash
set -a; source /tmp/lme.env; set +a
# Truncate DB state so namespaces are fresh
docker exec attestor-pg-local  psql -U postgres -d attestor -c "TRUNCATE TABLE memories CASCADE;"
docker exec attestor-neo4j-local cypher-shell -u neo4j -p attestor "MATCH (n) DETACH DELETE n;"

python -m attestor.cli longmemeval \
  --data ./longmemeval_s_cleaned.json \
  --categories temporal-reasoning \
  --max-samples 30 \
  --parallel 8 \
  --use-distillation --distill-model openai/gpt-5.1 \
  --answer-model openai/gpt-5.1 \
  --verify \
  --judge-model openai/gpt-5.1 \
  --judge-model anthropic/claude-sonnet-4.6 \
  --verbose \
  --output longmemeval-attestor-30samp-temporal-gpt51-sonnet46.json
```

### Full 500 samples (all categories)

```bash
# Same truncation step as above.
python -m attestor.cli longmemeval \
  --data ./longmemeval_s_cleaned.json \
  --parallel 8 \
  --use-distillation --distill-model openai/gpt-5.1 \
  --answer-model openai/gpt-5.1 \
  --verify \
  --judge-model openai/gpt-5.1 \
  --judge-model anthropic/claude-sonnet-4.6 \
  --verbose \
  --output longmemeval-attestor-full500.json
```

## Verifying a published output

1. Compute the SHA256 of the output JSON and compare to its `.sha256` sidecar:
   ```
   shasum -a 256 longmemeval-attestor-*.json | diff - *.sha256
   ```
2. Check `provenance.git_sha` against the attestor repo's git log:
   ```
   git -C /path/to/attestor cat-file -t <sha>
   ```
3. Check `provenance.dataset_sha256` against your downloaded LongMemEval file.
4. Inspect any per-sample result directly — `samples[i]` carries `question`,
   `gold`, `answer`, and full `judgments[judge_model].reasoning`. You can
   re-judge with your own judge model and compare.

## Non-determinism

LLM outputs are non-deterministic. Two runs with identical configs will
produce slightly different per-sample answers and verdicts. The aggregated
accuracy is stable to within a few percentage points at n≥30. For exact
reproducibility at the sample level, you would need to pin the underlying
model version + seed, which OpenRouter does not guarantee. Provenance
records the model ID so you can at least reproduce *against the same
model*.

## What we do NOT vary across runs

- The dataset (pinned by SHA256)
- The ingest prompt / distillation prompt / answer prompt / judge prompt
  (all in source; pinned by git SHA)
- The backend (Postgres + Neo4j with explicit config)
- The embedding model (`text-embedding-3-small`, 1536d)

Anything else that could affect the number is in `run_config` and
`provenance` inside each output JSON.
