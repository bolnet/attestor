# Attestor setup demo — Variant 2 of 8: Local + Project scope
# Same as variant 01 but Q1 picks Project (./.mcp.json) instead of Global.
# Render: vhs docs/demo/setup-02-local-project-scope.tape
#
# Prereq cleanup (run before vhs):
#   uv tool uninstall attestor 2>/dev/null; rm -rf ~/.attestor
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"
#   python3 -c "import json,pathlib; p=pathlib.Path('/Users/aarjay/Documents/agent-memory/.mcp.json');\
#     d=json.loads(p.read_text()) if p.exists() else {'mcpServers':{}}; d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); (p.write_text(json.dumps(d,indent=2)) if p.exists() else None)"

Output docs/demo/setup-02-local-project-scope.webm

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

# ── Q1: pick Project (./.mcp.json) — second option ──────────────────
Sleep 14s
Down
Sleep 1500ms
Enter

# ── Q2–Q7: accept defaults ──────────────────────────────────────────
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
Sleep 7s
Enter
Sleep 7s
Enter

# ── Wait for install + doctor + project MCP write ───────────────────
Sleep 70s
