# Attestor setup demo — Variant 3 of 8: Local + custom store path
# Same as variant 01 but Q2 picks a custom store path (~/code/memstore)
# instead of the default ~/.attestor. Everything else is defaults.
# Render: vhs docs/demo/setup-03-local-custom-path.tape
#
# Prereq cleanup (run before vhs):
#   uv tool uninstall attestor 2>/dev/null; rm -rf ~/.attestor ~/code/memstore
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"

Output docs/demo/setup-03-local-custom-path.webm

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

# ── Q1: accept default (Global) ─────────────────────────────────────
Sleep 14s
Enter

# ── Q2: pick Custom path, then type ~/code/memstore ─────────────────
Sleep 7s
Down
Sleep 1500ms
Enter
Sleep 2s
Type "~/code/memstore"
Sleep 800ms
Enter

# ── Q3–Q7: accept defaults ──────────────────────────────────────────
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

# ── Wait for install + doctor + MCP merge to complete ───────────────
Sleep 70s
