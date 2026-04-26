# `docs/bench/` — published benchmark baseline

This directory carries the baseline numbers the v4 regression gate
compares every PR against.

| File | What it is |
|---|---|
| `v4-baseline.json` | The published baseline. CI loads this on every PR; if the new run regresses any benchmark by more than its threshold, the gate blocks the merge. |

## How the gate uses this file

`.github/workflows/evals.yml` runs two always-on jobs on every PR:

1. **evals-unit** — pytest on the evals + regression test files.
2. **evals-gate** — runs `python -m evals.gate` against any uploaded
   `*_summary.json` files, comparing each to the matching entry in this
   baseline file. Exit 1 = blocked.

A third job, **evals-heavy**, is `workflow_dispatch`-only. It spins up
Postgres, runs the live LongMemEval / BEAM / AbstentionBench runners,
and uploads the resulting summaries as the `bench-summaries` artifact.
The next gate job picks them up.

## Promoting a new baseline

1. Trigger the heavy benchmark run from the Actions tab:
   `Run workflow → Evals → run_heavy_benchmarks=true`.
2. Wait for the run to complete and download the `bench-summaries`
   artifact.
3. Inspect each `*_summary.json`. If the numbers look right, promote
   them to the baseline:

```bash
# Dry-run first — see exactly what would change
python -m evals.publish_baseline \
  --summaries-dir bench-summaries/ \
  --baseline docs/bench/v4-baseline.json \
  --dry-run

# When happy, drop --dry-run
python -m evals.publish_baseline \
  --summaries-dir bench-summaries/ \
  --baseline docs/bench/v4-baseline.json

# Commit the new baseline
git add docs/bench/v4-baseline.json
git commit -m "bench(v4): publish 4.0.0 baseline"
```

## Promoting a partial baseline

After fixing one specific benchmark, you may want to update only that
benchmark's number without touching the others (e.g., LME numbers
shifted because we changed the answerer model, but BEAM is still the
same code-path):

```bash
python -m evals.publish_baseline \
  --summaries-dir bench-summaries/ \
  --baseline docs/bench/v4-baseline.json \
  --only longmemeval
```

`merge_summaries()` preserves any benchmark in the existing baseline
that the incoming summaries don't mention, so a partial run never
silently deletes the rest of your baseline.

## Threshold tuning

The baseline file carries two threshold sources:

```json
{
  "default_threshold": 2.0,
  "thresholds": {
    "abstention": 1.0,
    "regression": 0.5
  },
  ...
}
```

* `default_threshold` — drift > this many points fails the gate.
  Default 2.0pt (conservative; 95% CI on most LLM-judged benchmarks).
* `thresholds[<benchmark>]` — per-benchmark overrides.
  Tighter for deterministic benchmarks (`regression: 0.5` — pure logic
  shouldn't drift at all). Tighter for sensitive metrics
  (`abstention: 1.0` — F1 changes fast under calibration shifts).

Tune by editing the file directly; `publish_baseline` preserves your
overrides through promotions.

## What lives here vs `evals/`

* `evals/` — the runners + scorers + gate. **Code.**
* `docs/bench/` — the baseline numbers + this runbook. **Data + docs.**

Reviewers care about `docs/bench/v4-baseline.json` diffs the same way
they care about migrations: every change is intentional, signed off on
by a human, and explained in the commit message.
