#!/usr/bin/env bash
# Teardown Neon infrastructure for memwright
#
# Usage:
#   bash agent_memory/infra/neon-teardown.sh
#
# Requires: neonctl (brew install neonctl)

set -euo pipefail

PROJECT_ID="soft-mud-41286413"
PROJECT_NAME="memwright"

echo "=== Memwright Neon Teardown ==="
echo "Project: ${PROJECT_NAME} (${PROJECT_ID})"
echo ""

# Confirm
read -p "Delete Neon project '${PROJECT_NAME}'? This is irreversible. [y/N] " confirm
if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
    echo "Aborted."
    exit 0
fi

echo "Deleting Neon project..."
neonctl projects delete "${PROJECT_ID}" --org-id org-holy-brook-81524285
echo "Done. Neon project '${PROJECT_NAME}' deleted."
echo ""
echo "Remember to remove NEON_DATABASE_URL from .env"
