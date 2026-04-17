#!/usr/bin/env bash
# Deploy memwright + ArangoDB to AWS ECS Fargate.
#
# Prerequisites:
#   - AWS CLI configured (aws configure)
#   - Terraform >= 1.5
#   - Docker running locally
#
# Usage:
#   ./setup.sh                    # deploy with defaults
#   ./setup.sh us-west-2          # deploy to specific region
#   ./setup.sh us-east-1 prod     # deploy with environment name

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

AWS_REGION="${1:-us-east-1}"
ENVIRONMENT="${2:-dev}"
PROJECT_NAME="memwright"
NAME_PREFIX="${PROJECT_NAME}-${ENVIRONMENT}"

echo "=== Memwright AWS Deployment ==="
echo "Region:      $AWS_REGION"
echo "Environment: $ENVIRONMENT"
echo "Project:     $PROJECT_ROOT"
echo ""

# ── 1. Terraform init + apply (creates ECR, VPC, ECS, etc.) ──

cd "$SCRIPT_DIR"

terraform init -input=false

# First apply to create ECR repository (needed before docker push)
terraform apply -input=false -auto-approve \
  -var="aws_region=$AWS_REGION" \
  -var="environment=$ENVIRONMENT" \
  -target=aws_ecr_repository.memwright

# ── 2. Build and push Docker image to ECR ──

AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
ECR_URL="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"
REPO_URL="${ECR_URL}/${NAME_PREFIX}-memwright"

echo ""
echo "=== Building Docker image ==="

aws ecr get-login-password --region "$AWS_REGION" | \
  docker login --username AWS --password-stdin "$ECR_URL"

docker build --platform linux/amd64 -t "${REPO_URL}:latest" \
  -f "$SCRIPT_DIR/Dockerfile" \
  "$PROJECT_ROOT"

echo ""
echo "=== Pushing to ECR ==="

docker push "${REPO_URL}:latest"

# ── 3. Full terraform apply ──

echo ""
echo "=== Applying full infrastructure ==="

terraform apply -input=false -auto-approve \
  -var="aws_region=$AWS_REGION" \
  -var="environment=$ENVIRONMENT"

# ── 4. Output ──

echo ""
echo "=== Deployment complete ==="
terraform output

ALB_URL=$(terraform output -raw alb_url)
echo ""
echo "API endpoint: $ALB_URL"
echo "Health check: curl $ALB_URL/health"
echo ""
echo "Note: ECS task may take 1-2 minutes to start. Monitor at:"
echo "  https://${AWS_REGION}.console.aws.amazon.com/ecs/v2/clusters/${NAME_PREFIX}-cluster"
