#!/usr/bin/env bash
# Tear down Memwright Cloud Run deployment
# Usage: bash scripts/teardown-cloudrun.sh
set -euo pipefail

INFRA_DIR="agent_memory/infra/cloudrun"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Memwright Cloud Run Teardown ==="

GCP_PROJECT=$(gcloud config get-value project 2>/dev/null)
GCP_REGION="${GCP_REGION:-us-central1}"

if [[ ! -f "$INFRA_DIR/terraform.tfstate" ]]; then
    echo "No Terraform state found — nothing to destroy."
    exit 0
fi

cd "$INFRA_DIR"
terraform destroy -auto-approve \
    -var="gcp_project=$GCP_PROJECT" \
    -var="gcp_region=$GCP_REGION" \
    -var="arango_url=dummy" \
    -var="arango_password=dummy"

cd "$PROJECT_ROOT"
rm -f "$INFRA_DIR/.service_url"

echo ""
echo "=== Teardown Complete ==="
echo "ArangoDB Oasis data is untouched."
