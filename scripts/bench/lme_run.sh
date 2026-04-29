#!/usr/bin/env bash
# Single-slice (or full) LongMemEval run, YAML-driven.
#
# Reads bench knobs from configs/bench.yaml + stack from configs/attestor.yaml
# via scripts/_bench_config.py — never hardcodes a model, embedder, or DB URL.
# CLI flags only override what the YAML says.
#
# Usage:
#   scripts/bench/lme_run.sh                       # full LME-S, all categories
#   scripts/bench/lme_run.sh knowledge-update      # one slice
#   scripts/bench/lme_run.sh knowledge-update 50   # one slice, capped at 50q
#   scripts/bench/lme_run.sh "" "" oracle          # full Oracle variant
#
# Persists docs/bench/lme-{variant}-{category}-{date}.{report,summary}.json
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .venv/bin/python ]]; then
  echo "error: .venv/bin/python not found — run 'poetry install' first" >&2
  exit 2
fi

# Load .env so OPENROUTER_API_KEY / VOYAGE_API_KEY / NEO4J_PASSWORD are present.
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

CATEGORY="${1:-}"          # empty = all categories
LIMIT="${2:-}"             # empty = full dataset
VARIANT_OVERRIDE="${3:-}"  # empty = use bench.yaml's lme.variant

# Pull defaults from bench.yaml. Echoed back so the operator sees what
# resolved.
.venv/bin/python - <<'PY'
from scripts._bench_config import get_bench, print_bench_banner
cfg = get_bench()
print_bench_banner(cfg, run_label="lme_run")
PY

# Resolve the variant: CLI > bench.yaml.
VARIANT=$(
  .venv/bin/python -c "
from scripts._bench_config import get_bench
import sys
cfg = get_bench()
print(sys.argv[1] or cfg.lme.variant)
" "$VARIANT_OVERRIDE"
)

# Build args for lme_smoke_local.py. --output-dir always pulled from YAML.
OUTPUT_DIR=$(.venv/bin/python -c "from scripts._bench_config import get_bench; print(get_bench().lme.output_dir)")
ARGS=(--variant "$VARIANT" --output-dir "$OUTPUT_DIR" --yes)

if [[ -n "$CATEGORY" ]]; then
  ARGS+=(--category "$CATEGORY")
fi

# --n maps to --limit. When omitted, lme_smoke_local defaults to --n 2 which
# is wrong for benches; force the full slice by passing a huge --n. The
# runner truncates to dataset size.
if [[ -n "$LIMIT" ]]; then
  ARGS+=(--n "$LIMIT")
else
  ARGS+=(--n 100000)
fi

echo
echo "[lme_run] launching: .venv/bin/python scripts/lme_smoke_local.py ${ARGS[*]}"
echo

exec .venv/bin/python scripts/lme_smoke_local.py "${ARGS[@]}"
