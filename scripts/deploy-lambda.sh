#!/usr/bin/env bash
# Deploy Memwright Lambda to AWS
# Usage: bash scripts/deploy-lambda.sh
set -euo pipefail

INFRA_DIR="agent_memory/infra/lambda"
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "=== Memwright Lambda Deploy ==="

# Check prerequisites
for cmd in aws docker terraform; do
    if ! command -v "$cmd" &>/dev/null; then
        echo "Error: $cmd not found. Install it first."
        exit 1
    fi
done

# Check AWS credentials
echo "Checking AWS credentials..."
AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
AWS_REGION=$(aws configure get region 2>/dev/null || echo "us-east-1")
echo "Account: $AWS_ACCOUNT  Region: $AWS_REGION"

# Load ArangoDB config from .env
if [[ -f .env ]]; then
    set -a; source .env; set +a
fi

ARANGO_URL="${ARANGO_URL:?Set ARANGO_URL in .env}"
ARANGO_PASSWORD="${ARANGO_PASSWORD:?Set ARANGO_PASSWORD in .env}"
ARANGO_DATABASE="${ARANGO_DATABASE:-memwright}"

# Terraform init + apply (creates ECR first)
echo ""
echo "=== Terraform Apply ==="
cd "$INFRA_DIR"
terraform init -input=false
terraform apply -auto-approve \
    -var="aws_region=$AWS_REGION" \
    -var="arango_url=$ARANGO_URL" \
    -var="arango_password=$ARANGO_PASSWORD" \
    -var="arango_database=$ARANGO_DATABASE"

ECR_URL=$(terraform output -raw ecr_repository_url)
FUNC_NAME=$(terraform output -raw lambda_function_name)
API_URL=$(terraform output -raw api_url)
cd "$PROJECT_ROOT"

# Build and push container image
echo ""
echo "=== Docker Build & Push ==="
aws ecr get-login-password --region "$AWS_REGION" | docker login --username AWS --password-stdin "$ECR_URL"

docker build --platform linux/amd64 -t memwright-lambda -f "$INFRA_DIR/Dockerfile" .
docker tag memwright-lambda:latest "$ECR_URL:latest"
docker push "$ECR_URL:latest"

# Update Lambda to use new image
echo ""
echo "=== Update Lambda ==="
IMAGE_URI="$ECR_URL:latest"
aws lambda update-function-code \
    --function-name "$FUNC_NAME" \
    --image-uri "$IMAGE_URI" \
    --region "$AWS_REGION" \
    --no-cli-pager

echo "Waiting for Lambda to become active..."
aws lambda wait function-active-v2 --function-name "$FUNC_NAME" --region "$AWS_REGION"

# Health check
echo ""
echo "=== Health Check ==="
sleep 3
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" "$API_URL/health" || echo "000")
if [[ "$HTTP_CODE" == "200" ]]; then
    echo "Health check passed!"
    curl -s "$API_URL/health" | python3 -m json.tool
else
    echo "Health check returned HTTP $HTTP_CODE (may need cold start, try again in 15s)"
fi

echo ""
echo "=== Deployed ==="
echo "API URL:  $API_URL"
echo "Test:     curl $API_URL/health"
echo "Teardown: bash scripts/teardown-lambda.sh"
echo "Bench:    bash scripts/bench-lambda.sh"

# Save URL for other scripts
echo "$API_URL" > "$INFRA_DIR/.api_url"
