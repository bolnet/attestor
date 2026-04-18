# VHS tape: Attestor uninstall demo
# Render: vhs docs/demo/uninstall.tape
# Output: docs/demo/uninstall.gif + docs/demo/uninstall.mp4

Output docs/demo/uninstall.gif
Output docs/demo/uninstall.mp4

Set FontSize 15
Set Width 1200
Set Height 800
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 45ms
Set PlaybackSpeed 1.0

# ── Open Claude Code ──
Type "claude"
Enter
Sleep 3s

# ── User types the uninstall command ──
Type "uninstall attestor"
Enter
Sleep 2s

# ── Detection ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "echo ''"
Enter
Type "echo '  ╔══════════════════════════════════════════════════════╗'"
Enter
Type "echo '  ║     ATTESTOR UNINSTALL                              ║'"
Enter
Type "echo '  ╚══════════════════════════════════════════════════════╝'"
Enter
Type "echo ''"
Enter
Type "echo '  Detecting Attestor installation...'"
Enter
Type "echo '  ✓ attestor binary found at ~/.local/bin/attestor'"
Enter
Type "echo '  ✓ MCP config found in ~/.claude/.mcp.json'"
Enter
Type "echo '  ✓ Hooks found in ~/.claude/settings.json'"
Enter
Type "echo '  ✓ Store found at ~/.attestor/ (31 memories)'"
Enter
Type "echo ''"
Enter
Sleep 3s

# ── Q1: What to remove ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  What would you like to remove?                      │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  › Everything (full uninstall)"
Enter
Type "    MCP config + hooks only (keep store + binary)"
Enter
Type "    Hooks only"
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 3s

Type "echo '  ✓ Selected: Everything'"
Enter
Sleep 1500ms

# ── Q2: Confirm store deletion ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  ⚠  Store at ~/.attestor/ contains 31 memories.     │"
Enter
Type "  │     This action is irreversible.                     │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  › Delete store (all memories lost)"
Enter
Type "    Keep store (can re-attach later)"
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 3s

Type "echo '  ✓ Selected: Keep store'"
Enter
Type "echo '    ~/.attestor/ preserved — re-attach anytime.'"
Enter
Sleep 2s

# ── Removing ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "echo ''"
Enter
Type "echo '  Removing Attestor...'"
Enter
Type "echo ''"
Enter
Sleep 1s
Type "echo '  1/4  Removing MCP entry from ~/.claude/.mcp.json'"
Enter
Type "echo '       ✓ Removed attestor server entry'"
Enter
Sleep 1500ms
Type "echo ''"
Enter
Type "echo '  2/4  Removing hooks from ~/.claude/settings.json'"
Enter
Type "echo '       ✓ Removed session-start hook'"
Enter
Type "echo '       ✓ Removed post-tool-use hook'"
Enter
Type "echo '       ✓ Removed stop hook'"
Enter
Sleep 1500ms
Type "echo ''"
Enter
Type "echo '  3/4  Uninstalling attestor binary'"
Enter
Type "echo '       $ pipx uninstall attestor'"
Enter
Sleep 1500ms
Type "echo '       ✓ Uninstalled attestor'"
Enter
Sleep 1s
Type "echo ''"
Enter
Type "echo '  4/4  Store preserved at ~/.attestor/'"
Enter
Sleep 2s

# ── Final summary ──
Type "echo ''"
Enter
Type "echo '  ╔══════════════════════════════════════════════════════╗'"
Enter
Type "echo '  ║  UNINSTALL COMPLETE                                  ║'"
Enter
Type "echo '  ╠══════════════════════════════════════════════════════╣'"
Enter
Type "echo '  ║                                                      ║'"
Enter
Type "echo '  ║  ✓ MCP config removed                                ║'"
Enter
Type "echo '  ║  ✓ Hooks removed                                     ║'"
Enter
Type "echo '  ║  ✓ Binary uninstalled                                ║'"
Enter
Type "echo '  ║  ◐ Store kept at ~/.attestor/ (31 memories)         ║'"
Enter
Type "echo '  ║                                                      ║'"
Enter
Type "echo '  ║  To re-install: just say install attestor        ║'"
Enter
Type "echo '  ║  To delete store: rm -rf ~/.attestor                ║'"
Enter
Type "echo '  ║                                                      ║'"
Enter
Type "echo '  ╚══════════════════════════════════════════════════════╝'"
Enter
Sleep 5s
