#!/usr/bin/env bash
# Teardown Memwright Lambda from AWS
# Usage: bash scripts/teardown-lambda.sh [--force]
set -euo pipefail

INFRA_DIR="agent_memory/infra/lambda"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

FORCE="${1:-}"

if [[ "$FORCE" != "--force" ]]; then
    echo "=== Memwright Lambda Teardown ==="
    echo "This will DESTROY all AWS Lambda infrastructure."
    echo "ArangoDB data on Oasis is NOT affected."
    echo ""
    read -p "Continue? (yes/no): " CONFIRM
    if [[ "$CONFIRM" != "yes" ]]; then
        echo "Aborted."
        exit 0
    fi
fi

# Load .env for Terraform vars
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")
ARANGO_URL="${ARANGO_URL:-http://localhost:8529}"
ARANGO_PASSWORD="${ARANGO_PASSWORD:-}"
ARANGO_DATABASE="${ARANGO_DATABASE:-memwright}"

cd "$INFRA_DIR"

echo "Running terraform destroy..."
terraform destroy -auto-approve \
    -var="aws_region=$AWS_REGION" \
    -var="arango_url=$ARANGO_URL" \
    -var="arango_password=$ARANGO_PASSWORD" \
    -var="arango_database=$ARANGO_DATABASE"

# Clean up local state files
rm -f .api_url

echo ""
echo "=== Teardown Complete ==="
echo "All Lambda infrastructure destroyed."
echo "ArangoDB Oasis data is untouched."
