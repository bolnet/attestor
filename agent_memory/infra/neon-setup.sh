#!/usr/bin/env bash
# Setup Neon infrastructure for memwright
#
# Creates a free-tier Neon PostgreSQL project with pgvector.
# Cost: $0 (free tier: 0.5GB storage, 190 compute hours/month)
#
# Usage:
#   bash agent_memory/infra/neon-setup.sh
#
# Requires: neonctl (brew install neonctl)
# Auth:     neonctl auth

set -euo pipefail

PROJECT_NAME="${1:-memwright}"
REGION="${2:-aws-us-east-1}"

echo "=== Memwright Neon Setup ==="
echo "Project: ${PROJECT_NAME}"
echo "Region:  ${REGION}"
echo "Cost:    FREE (0.5GB storage, 190 compute hours/month)"
echo ""

# Create project
echo "Creating Neon project..."
OUTPUT=$(neonctl projects create \
    --name "${PROJECT_NAME}" \
    --region-id "${REGION}" \
    --output json 2>/dev/null)

PROJECT_ID=$(echo "${OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['project']['id'])")
CONN_URI=$(echo "${OUTPUT}" | python3 -c "import sys,json; print(json.load(sys.stdin)['connection_uris'][0]['connection_uri'])")

echo "Project ID: ${PROJECT_ID}"
echo ""
echo "Connection URL (add to .env):"
echo "  NEON_DATABASE_URL=${CONN_URI}"
echo ""
echo "Test with:"
echo "  NEON_DATABASE_URL='${CONN_URI}' .venv/bin/pytest tests/test_postgres_live.py -v"
echo ""
echo "Teardown:"
echo "  neonctl projects delete ${PROJECT_ID}"
