# Attestor setup demo — Variant 5 of 8: Local + AWS Bedrock embeddings
# Same as variant 01 but Q4 picks Bedrock (cloud-native embeddings) instead of local.
# Render: vhs docs/demo/setup-05-local-bedrock-embeddings.tape
#
# Prereq cleanup (run before vhs):
#   uv tool uninstall attestor 2>/dev/null; rm -rf ~/.attestor
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"
#
# Prereq env: AWS credentials must be configured (`aws configure` or
# AWS_PROFILE / AWS_ACCESS_KEY_ID) so the Bedrock embedding call succeeds
# during `attestor doctor`.

Output docs/demo/setup-05-local-bedrock-embeddings.webm

Set Shell "bash"
Set FontSize 16
Set Width 1440
Set Height 900
Set Padding 20
Set Theme "Catppuccin Mocha"
Set TypingSpeed 35ms

# ── Launch Claude Code in the agent-memory repo (hidden) ────────────
Hide
Type "cd /Users/aarjay/Documents/agent-memory && clear && claude"
Enter
Sleep 6s
Show
Sleep 1s

# ── Trigger the install skill ───────────────────────────────────────
Type "install attestor"
Enter

# ── Q1: scope — default Global ──────────────────────────────────────
Sleep 14s
Enter

# ── Q2: store path — default ~/.attestor ───────────────────────────
Sleep 7s
Enter

# ── Q3: backend — default Local ─────────────────────────────────────
Sleep 7s
Enter

# ── Q4: embeddings — Bedrock / Vertex / Azure (third option) ────────
Sleep 7s
Down
Sleep 500ms
Down
Sleep 1500ms
Enter

# ── Q4-sub: cloud provider picker — default Bedrock ─────────────────
Sleep 5s
Enter

# ── Q5: hooks — default all three ───────────────────────────────────
Sleep 7s
Enter

# ── Q6: namespace — default user ────────────────────────────────────
Sleep 7s
Enter

# ── Q7: token budget — default 10000 ────────────────────────────────
Sleep 7s
Enter

# ── Wait for install + doctor + MCP merge to complete ───────────────
Sleep 70s
