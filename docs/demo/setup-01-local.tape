# Attestor setup demo — Variant 1 of 6: Local zero-config
# Records a real Claude Code session running `install attestor`.
# Render: vhs docs/demo/setup-01-local.tape
#
# Prereq: attestor must NOT be installed and ~/.attestor must not exist
# so the wizard runs end-to-end. Cleanup script:
#   uv tool uninstall attestor 2>/dev/null
#   rm -rf ~/.attestor
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"

Output docs/demo/setup-01-local.webm

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

# ── Let Claude run the wizard end-to-end ────────────────────────────
# Wizard does: detect → 7 questions (scope, store, backend, embeddings,
# hooks, namespace, budget) → uv tool install attestor → attestor
# doctor → merge ~/.claude/.mcp.json → optional hooks → verify.
# Defaults are "Local everything"; arrow keys + Enter accept each.
Sleep 8s
Enter
Sleep 5s
Enter
Sleep 5s
Enter
Sleep 5s
Enter
Sleep 5s
Enter
Sleep 5s
Enter
Sleep 5s
Enter
Sleep 5s
Enter

# ── Wait for install + doctor + MCP merge to complete ───────────────
Sleep 60s
