#!/usr/bin/env bash
# Tear down Memwright Azure Container Apps deployment
# Usage: bash scripts/teardown-containerapp.sh
set -euo pipefail

INFRA_DIR="agent_memory/infra/containerapp"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Memwright Azure Container Apps Teardown ==="

if [[ ! -f "$INFRA_DIR/terraform.tfstate" ]]; then
    echo "No Terraform state found — nothing to destroy."
    exit 0
fi

cd "$INFRA_DIR"
terraform destroy -auto-approve \
    -var="arango_url=dummy" \
    -var="arango_password=dummy"

cd "$PROJECT_ROOT"
rm -f "$INFRA_DIR/.service_url"

echo ""
echo "=== Teardown Complete ==="
echo "ArangoDB Oasis data is untouched."
