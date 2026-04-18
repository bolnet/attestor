# Migrating from Memwright to Attestor

`memwright` is now **`attestor`**. This document is the full migration checklist for v3.0. Everything old still works through v3.1 with a `DeprecationWarning`; support ends in v3.2.

- [Library users](#library-users)
- [CLI users](#cli-users)
- [Claude Code users with existing stores](#claude-code-users-with-existing-stores)
- [Deployed services](#deployed-services)
- [MCP clients](#mcp-clients)
- [Docker / container deployments](#docker--container-deployments)
- [What breaks in v3.2](#what-breaks-in-v32)

---

## Library users

```bash
pip uninstall memwright
pip install attestor
```

Rewrite imports:

```diff
- from agent_memory import AgentMemory
+ from attestor import AgentMemory

- from memwright import AgentMemory
+ from attestor import AgentMemory
```

`from memwright import ...` still works in v3.0 – v3.1 via a shim that depends on `attestor`. Every import emits a `DeprecationWarning`.

## CLI users

All three binaries are installed and functionally identical in v3.x:

```bash
attestor doctor        # new canonical
memwright doctor       # deprecated alias
agent-memory doctor    # deprecated alias
```

Switch scripts to `attestor`. The two legacy binaries are removed in v3.2.

## Claude Code users with existing stores

Existing stores at `~/.memwright/` are auto-detected and read in v3.x, so nothing breaks on upgrade. To move the data to the new canonical path:

```bash
attestor migrate                    # copies ~/.memwright/ → ~/.attestor/
attestor doctor                     # verifies the new store
```

`migrate` is non-destructive — the old directory stays until you delete it.

If your store lives somewhere else:

```bash
export ATTESTOR_PATH=/path/to/store      # new canonical
# or keep using the old variable for one more release:
export MEMWRIGHT_PATH=/path/to/store     # read + warn in v3.x
```

## Deployed services

Add the new env var alongside the existing one so rollouts are safe:

```bash
ATTESTOR_DATA_DIR=/var/lib/attestor      # new canonical
MEMWRIGHT_DATA_DIR=/var/lib/attestor     # still read in v3.x
```

The v3.0 image reads both. Drop `MEMWRIGHT_*` before deploying v3.2.

## MCP clients

Replace `memwright://` resource URIs with `attestor://`:

```diff
- memwright://recall/some-query
+ attestor://recall/some-query
```

The server still accepts `memwright://` for reads in v3.x but emits `attestor://` on write. Update `.mcp.json` server entries if they referenced `memwright`:

```diff
- "command": "memwright",
+ "command": "attestor",
  "args": ["mcp"]
```

Both binaries resolve to the same entry point in v3.x.

## Docker / container deployments

| Before | After |
|---|---|
| Image tag `memwright:latest` | `attestor:latest` |
| Env `MEMWRIGHT_DATA_DIR` | `ATTESTOR_DATA_DIR` |
| Volume mount `/var/lib/memwright` | `/var/lib/attestor` |
| Container name `memwright-api` | `attestor-api` |

For a rolling deployment, set both env vars during the transition window and switch the volume mount when convenient — `attestor` auto-detects either path.

## What breaks in v3.2

Remove these by v3.2 or the upgrade will fail:

- `from agent_memory import …` (already broken in v3.0)
- `from memwright import …` (shim removed)
- CLI invocations of `memwright` or `agent-memory`
- Env vars `MEMWRIGHT_PATH`, `MEMWRIGHT_DATA_DIR`
- MCP URIs with `memwright://` scheme
- `~/.memwright/` auto-detection — migrate first with `attestor migrate`

---

Questions, surprises, or breakage? File an issue at [github.com/bolnet/attestor/issues](https://github.com/bolnet/attestor/issues).
