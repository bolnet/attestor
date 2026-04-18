# VHS tape: Attestor install wizard demo
# Render: vhs docs/demo/install-wizard.tape
# Output: docs/demo/install-wizard.gif

Output docs/demo/install-wizard.gif

Set FontSize 14
Set Width 1200
Set Height 800
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 40ms
Set PlaybackSpeed 1.0

# Title
Type "# Attestor — 'install attestor' wizard demo"
Enter
Sleep 1s
Type "clear"
Enter
Sleep 500ms

# User invokes the skill
Type "# In Claude Code, the user just types:"
Enter
Sleep 800ms
Type "#   install attestor"
Enter
Sleep 1s
Type "clear"
Enter
Sleep 400ms

# Step 1: Detection
Type "# Step 1 — detect current state"
Enter
Sleep 600ms
Type "command -v attestor && python3 --version && ls ~/.attestor >/dev/null 2>&1 && echo STORE_EXISTS"
Enter
Sleep 2s

# Q0a: Existing MCP
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q0a · Existing MCP entry detected in ~/.claude/.mcp.json"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  > Update in place (Recommended)"
Enter
Type "    Keep as-is"
Enter
Type "    Add alongside as 'attestor'"
Enter
Type "EOF"
Enter
Sleep 3s

# Q0b: Existing hooks
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q0b · Attestor hooks already wired in settings.json"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "    Keep as-is (Recommended)"
Enter
Type "  > Replace with wizard selection"
Enter
Type "    Remove all Attestor hooks"
Enter
Type "EOF"
Enter
Sleep 3s

# Q0c: Existing store
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q0c · Store at ~/.attestor already exists"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  > Reuse (Recommended)"
Enter
Type "    Pick a new path"
Enter
Type "    Reset (destructive)"
Enter
Type "EOF"
Enter
Sleep 3s

# Q1 scope
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q1 · MCP config scope"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  > Global  (~/.claude/.mcp.json)       ← every project"
Enter
Type "    Project (./.mcp.json)               ← current dir only"
Enter
Type "EOF"
Enter
Sleep 2800ms

# Q2 store path
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q2 · Store location"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  > Default  ~/.attestor/  (Recommended)"
Enter
Type "    Custom path"
Enter
Type "EOF"
Enter
Sleep 2500ms

# Q3 backend type
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q3 · Backend type"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "    Local (Recommended)  — runs on your machine"
Enter
Type "  > Cloud-managed         — managed service"
Enter
Type "EOF"
Enter
Sleep 2800ms

# Q3b cloud platform
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q3b · Cloud platform"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  > AWS"
Enter
Type "    Azure"
Enter
Type "    GCP"
Enter
Type "EOF"
Enter
Sleep 2800ms

# Q3b.i AWS variant
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q3b.i · AWS backend variant"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  > ArangoDB Oasis on AWS (Recommended)"
Enter
Type "    DynamoDB + OpenSearch + Neptune"
Enter
Type "EOF"
Enter
Sleep 2800ms

# Credentials collection
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q-backend-creds · ArangoDB credentials"
Enter
Type " Required: ARANGO_URL, ARANGO_USER, ARANGO_PASSWORD"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "    Use existing env vars (Recommended)"
Enter
Type "  > Paste them now"
Enter
Type "    Skip"
Enter
Type "EOF"
Enter
Sleep 3s

# Where to persist secrets
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q-backend-creds · Where to persist?"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  > Gitignored .env  (~/.attestor/.env, chmod 600)"
Enter
Type "    Shell profile    (~/.zshrc)"
Enter
Type "    This session only"
Enter
Type "EOF"
Enter
Sleep 2800ms

# Q5 hooks
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q5 · Claude Code hooks (multi-select)"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  [x] session-start   (Recommended — multi-agent boot context)"
Enter
Type "  [x] post-tool-use   (Recommended — auto-capture)"
Enter
Type "  [x] stop            (session summary for handoffs)"
Enter
Type "  [ ] none"
Enter
Type "EOF"
Enter
Sleep 3s

# Q6 namespace
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q6 · Namespace"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "    Default 'user'"
Enter
Type "  > Custom  →  aarjay-personal"
Enter
Type "EOF"
Enter
Sleep 2500ms

# Q7 token budget
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Q7 · Default token budget for recall()"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "    2000"
Enter
Type "    5000"
Enter
Type "  > 10000  (Recommended for multi-agent)"
Enter
Type "    Custom"
Enter
Type "EOF"
Enter
Sleep 2800ms

# Install + doctor
Type "clear"
Enter
Type "# All answered. Running install…"
Enter
Sleep 800ms
Type "attestor doctor ~/.attestor 2>&1 | tail -8"
Enter
Sleep 4s

# Show merged MCP config
Type "clear"
Enter
Type "# Merged into ~/.claude/.mcp.json"
Enter
Sleep 600ms
Type "cat ~/.claude/.mcp.json"
Enter
Sleep 3s

# Final summary
Type "clear"
Enter
Type "cat <<'EOF'"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type " Install complete"
Enter
Type "────────────────────────────────────────────────────────────"
Enter
Type "  • attestor → /Users/aarjay/.local/bin/attestor"
Enter
Type "  • MCP config  → ~/.claude/.mcp.json   (updated)"
Enter
Type "  • Hooks       → session-start, post-tool-use, stop"
Enter
Type "  • Secrets     → ~/.attestor/.env      (chmod 600)"
Enter
Type "  • Namespace   → aarjay-personal"
Enter
Type "  • Budget      → 10000 tokens"
Enter
Type ""
Enter
Type "  ▶ Restart Claude Code to attach the MCP server."
Enter
Type "EOF"
Enter
Sleep 5s
