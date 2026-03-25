#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────────────────────
# Memwright Startup Script
# Starts all required services and runs health check.
# Usage: ./start.sh [memory-store-path]
# ─────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STORE_PATH="${1:-$HOME/.agent-memory/default}"
VENV="$SCRIPT_DIR/.venv"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}[memwright]${NC} $1"; }
ok()    { echo -e "${GREEN}[memwright]${NC} $1"; }
warn()  { echo -e "${YELLOW}[memwright]${NC} $1"; }
fail()  { echo -e "${RED}[memwright]${NC} $1"; }

# ── 1. Check Python venv ──
info "Checking Python environment..."
if [ ! -f "$VENV/bin/python" ]; then
    fail "No .venv found. Run: poetry install"
    exit 1
fi
ok "Python venv: $VENV"

# ── 2. Initialize memory store ──
info "Initializing memory store at $STORE_PATH..."
"$VENV/bin/python" -m agent_memory.cli init "$STORE_PATH" 2>/dev/null || true

# ── 3. Run full health check ──
echo
echo "═══════════════════════════════════════════════════"
"$VENV/bin/python" -m agent_memory.cli doctor "$STORE_PATH"

# ── 4. Print MCP config ──
echo "═══════════════════════════════════════════════════"
info "MCP config for Claude Code:"
echo
"$VENV/bin/python" -m agent_memory.cli setup-claude-code "$STORE_PATH"

echo
ok "Memwright is ready."
