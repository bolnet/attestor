# Attestor setup demo — Variant 8 of 8: Local + tight token budget (2000)
# Same as variant 01 but Q7 picks 2000 tokens instead of the 10000 default.
# Render: vhs docs/demo/setup-08-local-tight-budget.tape
#
# Prereq cleanup (run before vhs):
#   uv tool uninstall attestor 2>/dev/null; rm -rf ~/.attestor
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"

Output docs/demo/setup-08-local-tight-budget.webm

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

# ── Q1–Q6: accept defaults ──────────────────────────────────────────
Sleep 14s
Enter
Sleep 7s
Enter
Sleep 7s
Enter
Sleep 7s
Enter
Sleep 7s
Enter
Sleep 7s
Enter

# ── Q7: token budget — pick 2000 (two up from 10000 Recommended) ────
Sleep 7s
Up
Sleep 400ms
Up
Sleep 1500ms
Enter

# ── Wait for install + doctor + MCP merge to complete ───────────────
Sleep 70s
