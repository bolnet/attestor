#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== AgentMemory PyPI Publisher ==="
echo ""

# Prompt for token securely (hidden input)
read -s -p "Paste your PyPI API token: " PYPI_TOKEN
echo ""

if [ -z "$PYPI_TOKEN" ]; then
    echo "Error: No token provided."
    exit 1
fi

# Clean old builds
rm -rf dist/

# Build
echo "Building..."
poetry build --quiet

# Upload
echo "Uploading to PyPI..."
poetry publish --username __token__ --password "$PYPI_TOKEN"

echo ""
echo "Published! Install with: poetry add memwright"
