# Attestor setup demo — Variant 7 of 8: Local + Custom namespace (aarjay-personal)
# Same as variant 01 but Q6 picks Custom and types a namespace name.
# Render: vhs docs/demo/setup-07-local-custom-namespace.tape
#
# Prereq cleanup (run before vhs):
#   uv tool uninstall attestor 2>/dev/null; rm -rf ~/.attestor
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"

Output docs/demo/setup-07-local-custom-namespace.webm

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

# ── Q1–Q5: accept defaults ──────────────────────────────────────────
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

# ── Q6: pick Custom namespace, type "aarjay-personal" ───────────────
Sleep 7s
Down
Sleep 1500ms
Enter
Sleep 2s
Type "aarjay-personal"
Sleep 800ms
Enter

# ── Q7: token budget, default ───────────────────────────────────────
Sleep 7s
Enter

# ── Wait for install + doctor + MCP merge ───────────────────────────
Sleep 70s
