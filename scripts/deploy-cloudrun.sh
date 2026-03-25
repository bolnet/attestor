#!/usr/bin/env bash
# Deploy Memwright to GCP Cloud Run
# Usage: bash scripts/deploy-cloudrun.sh
set -euo pipefail

INFRA_DIR="agent_memory/infra/cloudrun"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Memwright Cloud Run Deploy ==="

# Check prerequisites
for cmd in gcloud docker terraform; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd not found. Install it first."
        exit 1
    fi
done

# Check GCP credentials
echo "Checking GCP credentials..."
GCP_PROJECT=$(gcloud config get-value project 2>/dev/null)
GCP_REGION="${GCP_REGION:-us-central1}"
if [[ -z "$GCP_PROJECT" ]]; then
    echo "Error: No GCP project set. Run: gcloud config set project <project-id>"
    exit 1
fi
echo "Project: $GCP_PROJECT  Region: $GCP_REGION"

# Load ArangoDB config from .env
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

ARANGO_URL="${ARANGO_URL:?Set ARANGO_URL in .env}"
ARANGO_PASSWORD="${ARANGO_PASSWORD:?Set ARANGO_PASSWORD in .env}"
ARANGO_DATABASE="${ARANGO_DATABASE:-memwright}"

TF_VARS="-var=gcp_project=$GCP_PROJECT -var=gcp_region=$GCP_REGION -var=arango_url=$ARANGO_URL -var=arango_password=$ARANGO_PASSWORD -var=arango_database=$ARANGO_DATABASE"

# Phase 1: Create Artifact Registry only
echo ""
echo "=== Phase 1: Artifact Registry ==="
cd "$INFRA_DIR"
terraform init -input=false
terraform apply -auto-approve -target=google_artifact_registry_repository.app $TF_VARS

AR_URL=$(terraform output -raw artifact_registry_url)
cd "$PROJECT_ROOT"

# Phase 2: Build and push container image
echo ""
echo "=== Phase 2: Docker Build & Push ==="
gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet

IMAGE="${AR_URL}/memwright:latest"
docker build --platform linux/amd64 -t memwright-cloudrun -f "$INFRA_DIR/Dockerfile" .
docker tag memwright-cloudrun:latest "$IMAGE"
docker push "$IMAGE"

# Phase 3: Create Cloud Run service (image now exists)
echo ""
echo "=== Phase 3: Cloud Run Service ==="
cd "$INFRA_DIR"
terraform apply -auto-approve $TF_VARS

SERVICE_URL=$(terraform output -raw service_url)
cd "$PROJECT_ROOT"

# Health check
echo ""
echo "=== Health Check ==="
sleep 5
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$SERVICE_URL/health" || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    echo "Health check passed!"
    curl -s "$SERVICE_URL/health" | python3 -m json.tool
else
    echo "Health check returned HTTP $HTTP_CODE (may need cold start, try again in 15s)"
fi

echo ""
echo "=== Deployed ==="
echo "API URL:  $SERVICE_URL"
echo "Test:     curl $SERVICE_URL/health"
echo "Teardown: bash scripts/teardown-cloudrun.sh"

# Save URL for other scripts
echo "$SERVICE_URL" > "$INFRA_DIR/.service_url"
