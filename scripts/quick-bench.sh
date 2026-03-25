#!/usr/bin/env bash
# Quick benchmark: add 2 memories, recall them, measure latency
# Usage: bash scripts/quick-bench.sh <base_url>
set -euo pipefail

URL="${1:?Usage: bash scripts/quick-bench.sh <base_url>}"
echo "=== Quick Bench: $URL ==="

# Health check
echo ""
echo "--- Health ---"
START=$(python3 -c "import time; print(time.time())")
HEALTH=$(curl -s -w "\n%{time_total}" "$URL/health")
BODY=$(echo "$HEALTH" | head -1)
TIME=$(echo "$HEALTH" | tail -1)
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo "Latency: ${TIME}s"

# Add memory 1
echo ""
echo "--- Add Memory 1 ---"
R1=$(curl -s -w "\n%{time_total}" -X POST "$URL/add" \
    -H "Content-Type: application/json" \
    -d '{"content":"My favorite programming language is Rust because of its memory safety guarantees","tags":["preference","programming"],"category":"preference"}')
BODY=$(echo "$R1" | head -1)
TIME=$(echo "$R1" | tail -1)
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo "Latency: ${TIME}s"

# Add memory 2
echo ""
echo "--- Add Memory 2 ---"
R2=$(curl -s -w "\n%{time_total}" -X POST "$URL/add" \
    -H "Content-Type: application/json" \
    -d '{"content":"I had a meeting with Sarah about the Q2 roadmap on March 20th. We decided to prioritize the API redesign.","tags":["meeting","roadmap","sarah"],"category":"conversation"}')
BODY=$(echo "$R2" | head -1)
TIME=$(echo "$R2" | tail -1)
echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
echo "Latency: ${TIME}s"

# Recall 1: preference
echo ""
echo "--- Recall: favorite language ---"
R3=$(curl -s -w "\n%{time_total}" -X POST "$URL/recall" \
    -H "Content-Type: application/json" \
    -d '{"query":"What is my favorite programming language?","limit":3}')
BODY=$(echo "$R3" | head -1)
TIME=$(echo "$R3" | tail -1)
echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  [{r[\"source\"]}] {r[\"content\"][:80]}  (score: {r[\"score\"]:.3f})') for r in d.get('data',[])]" 2>/dev/null || echo "$BODY"
echo "Latency: ${TIME}s"

# Recall 2: meeting
echo ""
echo "--- Recall: meeting with Sarah ---"
R4=$(curl -s -w "\n%{time_total}" -X POST "$URL/recall" \
    -H "Content-Type: application/json" \
    -d '{"query":"What did I discuss with Sarah?","limit":3}')
BODY=$(echo "$R4" | head -1)
TIME=$(echo "$R4" | tail -1)
echo "$BODY" | python3 -c "import sys,json; d=json.load(sys.stdin); [print(f'  [{r[\"source\"]}] {r[\"content\"][:80]}  (score: {r[\"score\"]:.3f})') for r in d.get('data',[])]" 2>/dev/null || echo "$BODY"
echo "Latency: ${TIME}s"

# Stats
echo ""
echo "--- Stats ---"
curl -s "$URL/stats" | python3 -m json.tool 2>/dev/null

echo ""
echo "=== Done ==="
