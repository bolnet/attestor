# Attestor setup demo — Variant 6 of 8: Local + minimal hooks (session-start only)
# Same as variant 01 but Q5 deselects post-tool-use and stop, keeping only session-start.
# Render: vhs docs/demo/setup-06-local-minimal-hooks.tape
#
# Prereq cleanup (run before vhs):
#   uv tool uninstall attestor 2>/dev/null; rm -rf ~/.attestor
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"

Output docs/demo/setup-06-local-minimal-hooks.webm

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

# ── Q1–Q4: accept defaults ──────────────────────────────────────────
Sleep 14s
Enter
Sleep 7s
Enter
Sleep 7s
Enter
Sleep 7s
Enter

# ── Q5: multi-select hooks — keep only session-start ────────────────
# Default has session-start checked at position 1. Navigate past
# post-tool-use, stop, and "none" to land on Submit, then Enter.
Sleep 7s
Down
Sleep 400ms
Down
Sleep 400ms
Down
Sleep 400ms
Down
Sleep 1500ms
Enter

# ── Q6–Q7: accept defaults ──────────────────────────────────────────
Sleep 7s
Enter
Sleep 7s
Enter

# ── Wait for install + doctor + MCP merge + hook wiring ─────────────
Sleep 70s
