#!/usr/bin/env bash
# Run benchmarks against deployed Memwright Lambda
# Usage:
#   bash scripts/bench-lambda.sh              # run full benchmark
#   bash scripts/bench-lambda.sh --quick      # quick smoke test (1 example)
#   bash scripts/bench-lambda.sh --stop       # stop running benchmark
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

INFRA_DIR="agent_memory/infra/lambda"
PID_FILE="$INFRA_DIR/.bench_pid"
LOG_FILE="benchmark-logs/lambda-bench-$(date +%Y%m%d-%H%M%S).log"

# Get API URL
if [[ -f "$INFRA_DIR/.api_url" ]]; then
    API_URL=$(cat "$INFRA_DIR/.api_url")
elif [[ -n "${MEMWRIGHT_LAMBDA_URL:-}" ]]; then
    API_URL="$MEMWRIGHT_LAMBDA_URL"
else
    echo "Error: No Lambda URL found. Deploy first: bash scripts/deploy-lambda.sh"
    exit 1
fi

case "${1:-run}" in
    --stop)
        if [[ -f "$PID_FILE" ]]; then
            PID=$(cat "$PID_FILE")
            if kill -0 "$PID" 2>/dev/null; then
                kill "$PID"
                echo "Benchmark stopped (PID $PID)"
            else
                echo "Benchmark already finished (PID $PID)"
            fi
            rm -f "$PID_FILE"
        else
            echo "No benchmark running"
        fi
        exit 0
        ;;
    --quick)
        BENCH_ARGS="--quick"
        ;;
    *)
        BENCH_ARGS=""
        ;;
esac

# Pre-flight: health check
echo "=== Lambda Benchmark ==="
echo "Endpoint: $API_URL"
echo ""

HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health")
if [[ "$HTTP_CODE" != "200" ]]; then
    echo "Error: Health check failed (HTTP $HTTP_CODE). Is Lambda deployed?"
    exit 1
fi
echo "Health check: OK"

# Ensure log directory exists
mkdir -p benchmark-logs

# Run benchmark in background
echo "Starting benchmark (log: $LOG_FILE)..."
echo ""

MEMWRIGHT_LAMBDA_URL="$API_URL" \
    .venv/bin/python -m pytest tests/test_lambda_live.py -v $BENCH_ARGS \
    2>&1 | tee "$LOG_FILE" &

BENCH_PID=$!
echo "$BENCH_PID" > "$PID_FILE"

echo "Benchmark running (PID $BENCH_PID)"
echo "Stop with: bash scripts/bench-lambda.sh --stop"
echo "Logs:      tail -f $LOG_FILE"

# Wait for completion
wait "$BENCH_PID" || true
rm -f "$PID_FILE"
echo ""
echo "=== Benchmark Complete ==="
echo "Results: $LOG_FILE"
