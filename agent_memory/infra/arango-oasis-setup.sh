#!/usr/bin/env bash
# Setup ArangoDB Oasis (AWS region) for memwright
#
# Creates a free-tier ArangoDB Oasis deployment in AWS us-east-1.
# Cost: $0 (free tier: 1 deployment, limited storage)
#
# Usage:
#   bash agent_memory/infra/arango-oasis-setup.sh
#
# Requires: oasisctl (https://github.com/arangodb-managed/oasisctl)
# Auth:     oasisctl login

set -euo pipefail

DEPLOYMENT_NAME="${1:-memwright}"
REGION="aws-us-east-1"
ORGANIZATION="${OASIS_ORGANIZATION_ID:-}"

echo "=== Memwright ArangoDB Oasis Setup ==="
echo "Deployment: ${DEPLOYMENT_NAME}"
echo "Region:     ${REGION} (AWS)"
echo "Cost:       FREE (Oasis free tier)"
echo ""

if ! command -v oasisctl &>/dev/null; then
    echo "Error: oasisctl not found."
    echo "Install: https://github.com/arangodb-managed/oasisctl/releases"
    echo "  macOS:  brew install arangodb-managed/tap/oasisctl"
    echo "  Linux:  curl -L https://github.com/arangodb-managed/oasisctl/releases/latest/download/oasisctl-linux-amd64 -o /usr/local/bin/oasisctl && chmod +x /usr/local/bin/oasisctl"
    exit 1
fi

if [[ -z "${ORGANIZATION}" ]]; then
    echo "Listing organizations..."
    oasisctl list organizations
    echo ""
    read -p "Enter organization ID: " ORGANIZATION
fi

# Get the free tier model ID
echo "Finding free tier model..."
MODEL_ID=$(oasisctl list deploymentmodels \
    --organization-id "${ORGANIZATION}" \
    --output json 2>/dev/null \
    | python3 -c "
import sys, json
models = json.load(sys.stdin).get('items', [])
free = [m for m in models if 'free' in m.get('name', '').lower() or m.get('is_free_tier', False)]
if free:
    print(free[0]['id'])
else:
    # Fall back to smallest model
    print(models[0]['id'] if models else '')
" 2>/dev/null || echo "")

if [[ -z "${MODEL_ID}" ]]; then
    echo "Warning: Could not auto-detect free tier model. Using default."
fi

# Create deployment
echo "Creating Oasis deployment..."
CREATE_ARGS=(
    --name "${DEPLOYMENT_NAME}"
    --organization-id "${ORGANIZATION}"
    --region-id "${REGION}"
)
if [[ -n "${MODEL_ID}" ]]; then
    CREATE_ARGS+=(--model-id "${MODEL_ID}")
fi

OUTPUT=$(oasisctl create deployment "${CREATE_ARGS[@]}" --output json 2>/dev/null)

DEPLOYMENT_ID=$(echo "${OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('id', ''))")
echo "Deployment ID: ${DEPLOYMENT_ID}"
echo ""

# Wait for deployment to be ready
echo "Waiting for deployment to become ready (this may take 2-5 minutes)..."
for i in $(seq 1 60); do
    STATUS=$(oasisctl get deployment \
        --deployment-id "${DEPLOYMENT_ID}" \
        --organization-id "${ORGANIZATION}" \
        --output json 2>/dev/null \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',{}).get('phase',''))" 2>/dev/null || echo "")
    if [[ "${STATUS}" == "Running" ]]; then
        echo "Deployment is ready!"
        break
    fi
    printf "."
    sleep 5
done
echo ""

# Get endpoint
ENDPOINT=$(oasisctl get deployment \
    --deployment-id "${DEPLOYMENT_ID}" \
    --organization-id "${ORGANIZATION}" \
    --output json 2>/dev/null \
    | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('status',{}).get('endpoint',''))" 2>/dev/null || echo "")

# Create root password
echo "Setting root password..."
ROOT_PASSWORD=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
oasisctl update deployment \
    --deployment-id "${DEPLOYMENT_ID}" \
    --organization-id "${ORGANIZATION}" \
    --root-password "${ROOT_PASSWORD}" 2>/dev/null || true

echo ""
echo "=== Connection Details ==="
echo ""
echo "Add to .env:"
echo "  ARANGO_URL=https://${ENDPOINT}:8529"
echo "  ARANGO_PASSWORD=${ROOT_PASSWORD}"
echo "  ARANGO_DATABASE=memwright"
echo ""
echo "Test with:"
echo "  ARANGO_URL='https://${ENDPOINT}:8529' ARANGO_PASSWORD='${ROOT_PASSWORD}' \\"
echo "    .venv/bin/pytest tests/test_arango_live.py -v"
echo ""
echo "Teardown:"
echo "  bash agent_memory/infra/arango-oasis-teardown.sh ${DEPLOYMENT_ID} ${ORGANIZATION}"
