#!/usr/bin/env bash
# Iterate every category from configs/bench.yaml's lme.categories list.
# Each slice produces its own docs/bench/lme-{variant}-{category}-{date}.*.json.
#
# Adding/removing slices is a YAML edit — never edit this script.
#
# Usage:
#   scripts/bench/lme_all.sh                 # full slices, current variant
#   scripts/bench/lme_all.sh 50              # cap each slice at 50q (smoke)
#   scripts/bench/lme_all.sh "" oracle       # full slices on Oracle variant
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LIMIT="${1:-}"
VARIANT_OVERRIDE="${2:-}"

# Pull category list from bench.yaml. Newline-separated so the bash for-loop
# preserves order even if a category name accidentally contains a space.
mapfile -t CATEGORIES < <(.venv/bin/python -c "
from scripts._bench_config import get_bench
for c in get_bench().lme.categories:
    print(c)
")

if [[ "${#CATEGORIES[@]}" -eq 0 ]]; then
  echo "error: bench.yaml has no lme.categories — nothing to run" >&2
  exit 2
fi

echo "[lme_all] running ${#CATEGORIES[@]} slices: ${CATEGORIES[*]}"
echo

FAILED=()
for cat in "${CATEGORIES[@]}"; do
  echo "════════════════════════════════════════════════════════════════════════"
  echo "[lme_all] slice: $cat"
  echo "════════════════════════════════════════════════════════════════════════"

  if ! scripts/bench/lme_run.sh "$cat" "$LIMIT" "$VARIANT_OVERRIDE"; then
    FAILED+=("$cat")
    echo "[lme_all] WARNING: $cat failed; continuing with next slice" >&2
  fi
  echo
done

if [[ "${#FAILED[@]}" -gt 0 ]]; then
  echo "[lme_all] DONE with failures: ${FAILED[*]}" >&2
  exit 1
fi

echo "[lme_all] DONE — all ${#CATEGORIES[@]} slices succeeded"
