# Attestor evals

Braintrust-driven evaluation harness for the attestor recall pipeline.
Currently scoped to two public benchmarks:

| Script | Benchmark | Judge |
|---|---|---|
| `braintrust_locomo.py` | [LOCOMO](https://github.com/snap-research/LoCoMo) — long-conversation memory QA | Claude Opus 4.7 (CoT, 1.0/0.5/0.0) |
| `braintrust_mab.py` | MAB — multi-aspect memory QA | Claude Opus 4.6 |
| `inspect_mab.py` | Local dry-run of the MAB pipeline with per-row printing (no Braintrust upload) | Claude Opus 4.6 |

## Install

The eval scripts pin `braintrust` and `anthropic` in the `eval` optional
extra:

```bash
pip install -e '.[eval]'           # if working in a clone
pip install 'attestor[eval]'       # from PyPI
```

LOCOMO additionally needs the public dataset. The loader expects
`~/.cache/attestor/locomo10.json` by default; override with `--data`.

## Env

Every script reads keys from the environment. A `.env` file at the repo
root is fine; `set -a && source .env && set +a` before running.

| Var | Used by | Purpose |
|---|---|---|
| `BRAINTRUST_API_KEY` | all | Uploads experiment runs to Braintrust. |
| `ANTHROPIC_API_KEY` | all | Judge (Claude Opus 4.6/4.7). |
| `OPENROUTER_API_KEY` | `braintrust_locomo.py` | Answer + extraction + reflection via OpenRouter. |
| `JUDGE_MODEL` | optional | Override judge (default per script). |
| `ANSWER_MODEL` | optional | Override answer model. |

## Usage

LOCOMO smoke (1 conversation, 3 questions):

```bash
.venv/bin/python evals/braintrust_locomo.py \
  --max-conversations 1 --max-questions 3 \
  --suffix smoke --debug
```

LOCOMO full upload + run:

```bash
.venv/bin/python evals/braintrust_locomo.py --upload-dataset
.venv/bin/python evals/braintrust_locomo.py --suffix full
```

MAB smoke:

```bash
.venv/bin/python evals/braintrust_mab.py \
  --categories AR --max-examples 1 --max-questions 3
```

MAB local inspect (no Braintrust write):

```bash
.venv/bin/python evals/inspect_mab.py \
  --categories AR --max-examples 1 --max-questions 3
```

## Scoring

The LOCOMO scorer mirrors the Braintrust server-side `factuality-b2d8`
scorer so local iteration and the hosted leaderboard agree. Scores are
`1.0` (correct), `0.5` (partial), `0.0` (incorrect or "I don't know").

## Notes

- The LOCOMO pipeline uses `attestor.locomo.answer_question` end-to-end so
  scores reflect the real recall path, not a mock.
- MAB currently under-performs due to embedding-model weakness on
  multi-hop reasoning; see `project_layer0_wiring` / roadmap for the
  dual-path plan.
