#!/usr/bin/env bash
# Run one ablation (preference + assistant, n=10 each, parallel=8) with
# distill+CoT+verify and dual judges. Writes outputs to docs/bench/ with
# sidecar SHA256.
#
# Usage: ./scripts/run_ablation.sh <tag>
# Example: ./scripts/run_ablation.sh v3-ablate-A
#
# Prereqs:
#   - OpenRouter credits (~$10 per ablation for 20 samples)
#   - docker compose up postgres neo4j
#   - /tmp/lme.env contains OPENROUTER_API_KEY + PG_CONNECTION_STRING
#
# Output files:
#   docs/bench/longmemeval-attestor-10samp-preference-<tag>.json[.sha256]
#   docs/bench/longmemeval-attestor-10samp-assistant-<tag>.json[.sha256]
#   logs/lme_preference_<tag>.log
#   logs/lme_assistant_<tag>.log

set -euo pipefail

TAG="${1:?usage: $0 <v3-ablate-A|B|C|D>}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Sanity: tag exists
git rev-parse "$TAG" >/dev/null 2>&1 || { echo "tag not found: $TAG"; exit 1; }

DATA=/Users/aarjay/Documents/longmemeval-bench/LongMemEval/data/longmemeval_s_cleaned.json
PREF_OUT="docs/bench/longmemeval-attestor-10samp-preference-${TAG}.json"
ASST_OUT="docs/bench/longmemeval-attestor-10samp-assistant-${TAG}.json"
PREF_LOG="logs/lme_preference_${TAG}.log"
ASST_LOG="logs/lme_assistant_${TAG}.log"

echo "== Ablation run: $TAG =="
echo "  preference output: $PREF_OUT"
echo "  assistant  output: $ASST_OUT"

# 1. Check out tag (code changes only affect ablation semantics)
git checkout -q "$TAG"

# 2. Truncate DB so namespaces start clean
docker exec attestor-pg-local psql -U postgres -d attestor \
  -c "TRUNCATE TABLE memories CASCADE;" >/dev/null
docker exec attestor-neo4j-local cypher-shell -u neo4j -p attestor \
  "MATCH (n) DETACH DELETE n;" >/dev/null
echo "DB clean"

# 3. Source env
set -a
source /tmp/lme.env
set +a

COMMON_FLAGS=(
  --data "$DATA"
  --max-samples 10
  --parallel "${PARALLEL:-8}"
  --use-distillation --distill-model openai/gpt-5.1
  --answer-model openai/gpt-5.1
  --verify
  --judge-model openai/gpt-5.1
  --judge-model anthropic/claude-sonnet-4.6
  --verbose
)

# 4. Launch both batches
poetry run python -m attestor.cli longmemeval \
  --categories single-session-preference \
  "${COMMON_FLAGS[@]}" \
  --output "$PREF_OUT" \
  > "$PREF_LOG" 2>&1 &
PREF_PID=$!

poetry run python -m attestor.cli longmemeval \
  --categories single-session-assistant \
  "${COMMON_FLAGS[@]}" \
  --output "$ASST_OUT" \
  > "$ASST_LOG" 2>&1 &
ASST_PID=$!

echo "Launched: preference PID=$PREF_PID, assistant PID=$ASST_PID"
echo "Tail the logs with:  tail -F $PREF_LOG $ASST_LOG"
echo "Wait with:           wait $PREF_PID $ASST_PID"
wait $PREF_PID $ASST_PID

# 5. Sidecar SHAs
shasum -a 256 "$PREF_OUT" | tee "${PREF_OUT}.sha256"
shasum -a 256 "$ASST_OUT" | tee "${ASST_OUT}.sha256"

# 6. Summary tail
echo "== Summary =="
tail -30 "$PREF_LOG" | grep -E "judge=|Retrieval|Mode|Personalization|accuracy"
tail -30 "$ASST_LOG" | grep -E "judge=|Retrieval|Mode|Personalization|accuracy"

echo "== Done =="
echo "Next: commit outputs + update docs/bench/BATCHES.md, then run next ablation."
