# Attestor setup demo — Variant 4 of 8: Local + OpenAI embeddings
# Same as variant 01 but Q4 picks OpenAI instead of local sentence-transformers.
# Render: vhs docs/demo/setup-04-local-openai-embeddings.tape
#
# Prereq cleanup (run before vhs):
#   uv tool uninstall attestor 2>/dev/null; rm -rf ~/.attestor
#   python3 -c "import json,pathlib; p=pathlib.Path.home()/'.claude/.mcp.json'; d=json.loads(p.read_text()); d.get('mcpServers',{}).pop('attestor',None); d.get('mcpServers',{}).pop('memory',None); p.write_text(json.dumps(d,indent=4))"
#
# Note: the wizard will prompt for OPENAI_API_KEY after Q4. Export it in the
# shell before recording so the prompt auto-fills, or let the demo capture the
# key-entry step as part of the video.

Output docs/demo/setup-04-local-openai-embeddings.webm

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

# ── Q1: scope, default (Global) ─────────────────────────────────────
Sleep 14s
Enter

# ── Q2: store path, default (~/.attestor) ──────────────────────────
Sleep 7s
Enter

# ── Q3: backend, default (Local) ────────────────────────────────────
Sleep 7s
Enter

# ── Q4: embeddings, pick OpenAI — second option ─────────────────────
Sleep 7s
Down
Sleep 1500ms
Enter

# ── Q5: hooks, default ──────────────────────────────────────────────
Sleep 7s
Enter

# ── Q6: namespace, default (user) ───────────────────────────────────
Sleep 7s
Enter

# ── Q7: token budget, default (10000) ───────────────────────────────
Sleep 7s
Enter

# ── Wait for install + doctor + MCP merge + OpenAI key capture ──────
Sleep 70s
