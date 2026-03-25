#!/usr/bin/env bash
# Deploy Memwright to Azure Container Apps
# Usage: bash scripts/deploy-containerapp.sh
set -euo pipefail

INFRA_DIR="agent_memory/infra/containerapp"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Memwright Azure Container Apps Deploy ==="

# Check prerequisites
for cmd in az docker terraform; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd not found. Install it first."
        exit 1
    fi
done

# Check Azure credentials
echo "Checking Azure credentials..."
AZ_SUB=$(az account show --query id -o tsv 2>/dev/null) || {
    echo "Error: Not logged in. Run: az login"
    exit 1
}
AZURE_LOCATION="${AZURE_LOCATION:-eastus}"
echo "Subscription: $AZ_SUB  Location: $AZURE_LOCATION"

# Load ArangoDB config from .env
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

ARANGO_URL="${ARANGO_URL:?Set ARANGO_URL in .env}"
ARANGO_PASSWORD="${ARANGO_PASSWORD:?Set ARANGO_PASSWORD in .env}"
ARANGO_DATABASE="${ARANGO_DATABASE:-memwright}"

TF_VARS="-var=azure_location=$AZURE_LOCATION -var=arango_url=$ARANGO_URL -var=arango_password=$ARANGO_PASSWORD -var=arango_database=$ARANGO_DATABASE"

# Phase 1: Create RG + ACR + Log Analytics + Environment (no Container App yet)
echo ""
echo "=== Phase 1: Registry & Environment ==="
cd "$INFRA_DIR"
terraform init -input=false
terraform apply -auto-approve \
    -target=azurerm_resource_group.app \
    -target=azurerm_container_registry.app \
    -target=azurerm_log_analytics_workspace.app \
    -target=azurerm_container_app_environment.app \
    $TF_VARS

ACR_SERVER=$(terraform output -raw acr_login_server)
cd "$PROJECT_ROOT"

# Phase 2: Build and push container image
echo ""
echo "=== Phase 2: Docker Build & Push ==="
az acr login --name "${ACR_SERVER%%.*}"

IMAGE="${ACR_SERVER}/memwright:latest"
docker build --platform linux/amd64 -t memwright-containerapp -f "$INFRA_DIR/Dockerfile" .
docker tag memwright-containerapp:latest "$IMAGE"
docker push "$IMAGE"

# Phase 3: Create Container App (image now exists in ACR)
echo ""
echo "=== Phase 3: Container App ==="
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
echo "Teardown: bash scripts/teardown-containerapp.sh"

# Save URL for other scripts
echo "$SERVICE_URL" > "$INFRA_DIR/.service_url"
