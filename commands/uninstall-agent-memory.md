# Uninstall Memwright from Claude Code

You are uninstalling **Memwright** (PyPI: `memwright`) from the user's Claude Code setup.

This command reverses everything `/install-agent-memory` did: removes the MCP server entry, strips hooks from `settings.json`, uninstalls the CLI, and — only with explicit user confirmation — deletes the local memory store.

## Prerequisites

- This command runs interactively. Use `AskUserQuestion` for each decision.
- Never delete files without confirmation.
- Never delete JSON keys other than those Memwright installed.

## Flow

### Step 1 — Detect installed state

Check each of the following and build a removal plan. Report what you find before asking anything:

1. **CLI binary**: `command -v memwright` and `pipx list | grep memwright`
2. **Global MCP config**: `~/.claude/.mcp.json` — look for the `mcpServers.memwright` key
3. **Project MCP config**: `./.mcp.json` in the current working directory — same key
4. **Global hooks**: `~/.claude/settings.json` — scan `hooks.SessionStart`, `hooks.PostToolUse`, `hooks.Stop` for any command containing `memwright`
5. **Project hooks**: `./.claude/settings.json` — same scan
6. **Store path(s)**: default `~/.memwright/`, plus any `--path` value pulled from the MCP config args

If nothing is installed, stop and tell the user "Memwright is not installed — nothing to remove."

### Step 2 — Confirm the uninstall scope (single AskUserQuestion)

Ask: "Remove memwright from: (a) global only, (b) project only, (c) both, (d) cancel?" — pre-select whatever scopes you actually found configured.

### Step 3 — Remove MCP server entry

For each chosen scope:

- Read the `.mcp.json` file.
- Delete only `mcpServers.memwright`. Preserve all other servers.
- If `mcpServers` becomes empty, leave it as `{}` (don't delete the key — other tools may expect it).
- Write the file back with the same 2-space indent JSON formatting it had before.

### Step 4 — Remove hooks from settings.json

For each chosen scope:

- Read the `settings.json` file.
- In `hooks.SessionStart`, `hooks.PostToolUse`, `hooks.Stop`: filter out any entry whose inner `hooks[].command` contains the substring `memwright`. If an outer matcher group becomes empty, drop the group.
- If a hook array becomes empty (`"Stop": []`), delete the key.
- If `hooks` itself becomes empty after removal, delete it.
- Preserve every other hook (including other tools that share `"matcher": "*"` slots).

### Step 5 — Uninstall the CLI

Run `pipx uninstall memwright`. If pipx isn't the install method, fall back to `pip uninstall -y memwright`. Capture and show the output.

### Step 6 — Ask about the memory store (second AskUserQuestion)

Ask: "Delete local memory store at `<path>`? This permanently removes all captured memories, vectors, and graph state."

Options:
- **Yes, delete** — run `rm -rf <path>` after echoing the full path for the record.
- **Keep** — leave the store intact (default, safer).

If multiple store paths were detected (e.g. different `--path` values across scopes), ask once per unique path.

### Step 7 — Verify

Run:
- `command -v memwright` — should report nothing
- `pipx list 2>&1 | grep -i memwright || echo "not present"`
- Re-read modified JSON files and confirm no `memwright` substring remains

Output a summary:

```
Removed:
  - MCP server entry (global)
  - 3 hooks (SessionStart, PostToolUse, Stop)
  - pipx package `memwright`
Preserved:
  - ~/.memwright/  (kept at user request)
```

## Safety rules

- Never use `jq -e` or any command that would exit non-zero and abort the script halfway through JSON edits. Read → parse → mutate in memory → write atomically.
- Never pass `--force` to `rm`. The `-rf` is already enough; a typo on the path should fail loudly.
- Never delete `settings.local.json` or other files the installer didn't create.
- If any JSON file fails to parse, stop and show the user — don't try to repair unknown damage.
- If the user says "cancel" at any confirmation, stop immediately and make no changes.
