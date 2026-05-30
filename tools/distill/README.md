<!--
SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
SPDX-License-Identifier: MIT
-->

# Memory distillation tool (operational / reference)

Standalone scripts that measure and realize **recall-payload token savings** by
distilling noisy auto-captured memories into compact, attributed facts. These
are operational tools, **not** part of the `attestor` package or the
`quickstart` install — they run against a live local store (`~/.attestor`).

> Productization (folding this into `attestor distill` and shipping it via
> `quickstart`) is tracked in issue #185.

## What's here

| File | Role |
|------|------|
| `distill.py` | Per-tenant `mem.consolidate(...)`: supersede noisy source rows, write compact distilled facts, re-sync Pinecone to the active set, and append a savings/spend entry to `~/.attestor/distill_savings.jsonl`. |
| `claude_sub_shim.py` | OpenAI-compatible shim over the `claude -p` **Claude Max subscription** (no per-token API billing). `distill.py` imports it directly for the consolidation LLM calls. |

"Tokens saved" = reduction in **active** memory content tokens for a tenant
(superseded-source tokens − distilled-fact tokens) — the per-recall payload
shrink the memory layer buys. The subscription tokens spent doing the
distillation are recorded alongside, so cost sits next to savings.

## Running

```bash
set -a; . ~/.attestor/.env; set +a
export PGPASSWORD="${PGPASSWORD:-postgres}"
export ATTESTOR_SKIP_SCHEMA_INIT=1            # schema already exists → skip DDL (avoids lock pile-up)
export ATTESTOR_CLAUDE_SHIM_TIMEOUT_S=120     # per-call wedge guard
python -u tools/distill/distill.py
```

Idempotent: only tenants with ≥8 un-distilled active rows are processed, and
rows already produced by a prior pass are skipped, so re-runs only distill
genuinely new material.

## Hard-won fixes baked in (do not regress)

1. **`claude -p` must pass `--setting-sources ""`** (in `claude_sub_shim.py`),
   **not** `--settings '{"hooks":{}}'`. The latter does **not** suppress hooks —
   the user's `SessionStart` hooks (including Attestor's own `attestor hook
   session-start`, which connects to the stores) fire on every headless spawn
   and hang the call past its timeout, looping indefinitely. Empty
   setting-sources skips those hooks while keeping the default config dir, so
   the Claude Max subscription/OAuth login is preserved. `--bare` and an
   isolated `CLAUDE_CONFIG_DIR` also skip hooks but **lose the login**
   (`Not logged in`), so they are not options for subscription auth.

2. **`distill.py`'s read connection must be `autocommit=True`.** Otherwise
   psycopg2 opens an implicit transaction on the first `SELECT` and leaves it
   `idle in transaction`, holding an `AccessShareLock` on `memories` for the
   whole `consolidate()` call. Any concurrent schema-init DDL
   (`AccessExclusiveLock`) then blocks behind it, and a lock pile-up wedges
   every subsequent `AgentMemory()` construction. Autocommit closes each
   `SELECT`'s transaction immediately, so no lock is held across calls.

3. Run with `ATTESTOR_SKIP_SCHEMA_INIT=1` when the schema already exists, and a
   per-call timeout, as additional wedge guards.
