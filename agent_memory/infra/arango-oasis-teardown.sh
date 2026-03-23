#!/usr/bin/env bash
# Teardown ArangoDB Oasis deployment for memwright
#
# Usage:
#   bash agent_memory/infra/arango-oasis-teardown.sh <deployment-id> <organization-id>
#
# Requires: oasisctl (https://github.com/arangodb-managed/oasisctl)

set -euo pipefail

DEPLOYMENT_ID="${1:-}"
ORGANIZATION_ID="${2:-${OASIS_ORGANIZATION_ID:-}}"

if [[ -z "${DEPLOYMENT_ID}" ]]; then
    echo "Usage: $0 <deployment-id> [organization-id]"
    echo ""
    echo "List deployments:"
    echo "  oasisctl list deployments --organization-id <org-id>"
    exit 1
fi

if [[ -z "${ORGANIZATION_ID}" ]]; then
    echo "Error: organization-id required (pass as arg or set OASIS_ORGANIZATION_ID)"
    exit 1
fi

echo "=== Memwright ArangoDB Oasis Teardown ==="
echo "Deployment:   ${DEPLOYMENT_ID}"
echo "Organization: ${ORGANIZATION_ID}"
echo ""

# Confirm
read -p "Delete Oasis deployment? This is irreversible. [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo "Deleting Oasis deployment..."
oasisctl delete deployment \
    --deployment-id "${DEPLOYMENT_ID}" \
    --organization-id "${ORGANIZATION_ID}"

echo "Done. Oasis deployment deleted."
echo ""
echo "Remember to remove ARANGO_URL and ARANGO_PASSWORD from .env"
