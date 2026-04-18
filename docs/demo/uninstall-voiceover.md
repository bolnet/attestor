# Uninstall Attestor — Voiceover Script

**Video**: `docs/demo/uninstall.mp4`

---

## [0:00] Opening

To uninstall, type "uninstall attestor" in Claude Code.

## [0:05] Detection

The wizard finds everything — binary at ~/.local/bin, MCP config, hooks in settings.json, and a store with 31 memories.

## [0:12] What to Remove

Three options. Full uninstall removes everything. You can also keep the store and binary and just remove the MCP config and hooks. Or remove hooks only. We'll select everything.

## [0:20] Store Confirmation

The store has 31 memories. This is irreversible. Two choices — delete the store or keep it. We'll keep it. You can always re-attach it later with a fresh install, or delete it manually with rm -rf.

## [0:30] Removing

Four steps. Remove the MCP entry from mcp.json. Remove all three hooks from settings.json. Uninstall the attestor binary via pipx. Store preserved at ~/.attestor.

## [0:42] Summary

Uninstall complete. MCP config removed. Hooks removed. Binary gone. But the store is still there with all 31 memories. To reinstall, just say "install attestor" again. To delete the store permanently, rm -rf ~/.attestor. Clean exit, no artifacts left behind.
