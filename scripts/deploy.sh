#!/usr/bin/env bash
# Deploy Memwright to any cloud
# Usage: bash scripts/deploy.sh <aws|gcp|azure> [--teardown]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
    echo "Usage: $0 <aws|gcp|azure> [--teardown]"
    echo ""
    echo "  aws    — Lambda + API Gateway (serverless, pay-per-request)"
    echo "  gcp    — Cloud Run (auto-scale 0–3, 2 CPU / 4GB)"
    echo "  azure  — Container Apps (scale-to-zero, 2 CPU / 4GB)"
    echo ""
    echo "Options:"
    echo "  --teardown  Destroy infrastructure instead of deploying"
    echo ""
    echo "Prerequisites: docker, terraform, cloud CLI (aws/gcloud/az), .env with ArangoDB creds"
    exit 1
}

[[ $# -lt 1 ]] && usage

CLOUD="$1"
TEARDOWN=false
[[ "${2:-}" == "--teardown" ]] && TEARDOWN=true

case "$CLOUD" in
    aws)
        if $TEARDOWN; then
            exec bash "$SCRIPT_DIR/teardown-lambda.sh"
        else
            exec bash "$SCRIPT_DIR/deploy-lambda.sh"
        fi
        ;;
    gcp)
        if $TEARDOWN; then
            exec bash "$SCRIPT_DIR/teardown-cloudrun.sh"
        else
            exec bash "$SCRIPT_DIR/deploy-cloudrun.sh"
        fi
        ;;
    azure)
        if $TEARDOWN; then
            exec bash "$SCRIPT_DIR/teardown-containerapp.sh"
        else
            exec bash "$SCRIPT_DIR/deploy-containerapp.sh"
        fi
        ;;
    *)
        echo "Error: Unknown cloud '$CLOUD'"
        usage
        ;;
esac
