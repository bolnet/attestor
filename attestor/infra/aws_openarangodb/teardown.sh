#!/usr/bin/env bash
# Tear down the attestor AWS deployment.
#
# Usage:
#   ./teardown.sh                    # destroy with defaults
#   ./teardown.sh us-west-2          # destroy in specific region
#   ./teardown.sh us-east-1 prod     # destroy specific environment

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

AWS_REGION="${1:-us-east-1}"
ENVIRONMENT="${2:-dev}"

echo "=== Attestor AWS Teardown ==="
echo "Region:      $AWS_REGION"
echo "Environment: $ENVIRONMENT"
echo ""

cd "$SCRIPT_DIR"

# Force-deregister ECS task definitions (Terraform doesn't always clean these)
echo "=== Deregistering ECS task definitions ==="
TASK_DEFS=$(aws ecs list-task-definitions \
  --family-prefix "attestor-${ENVIRONMENT}-task" \
  --region "$AWS_REGION" \
  --query "taskDefinitionArns[]" \
  --output text 2>/dev/null || true)

for td in $TASK_DEFS; do
  echo "  Deregistering: $td"
  aws ecs deregister-task-definition --task-definition "$td" --region "$AWS_REGION" > /dev/null 2>&1 || true
done

# Terraform destroy
echo ""
echo "=== Destroying infrastructure ==="

terraform destroy -input=false -auto-approve \
  -var="aws_region=$AWS_REGION" \
  -var="environment=$ENVIRONMENT"

echo ""
echo "=== Teardown complete ==="
echo "All AWS resources have been destroyed."
echo ""
echo "Note: CloudWatch log groups may take a few minutes to fully delete."
