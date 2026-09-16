#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Scheduled memory maintenance: distill noisy auto-captured rows into
compact attributed facts (via the Claude Max subscription shim), then
re-sync Pinecone to the active Postgres set — and record tokens saved.

Per project tenant with enough un-distilled active rows:
  mem.consolidate(reflection) -> N sources superseded, M facts added.

"Tokens saved" = reduction in ACTIVE memory content tokens for that
tenant (superseded-source tokens − distilled-fact tokens). That is the
per-recall payload shrink the memory layer buys. We also record the
REAL subscription tokens spent to do the distillation (from the shim
ledger) so cost and savings sit side by side.

Idempotent: reflection skips rows already produced by a prior pass
(`_consolidated_from`), so re-runs only distill genuinely new material.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import psycopg2
import psycopg2.extras

from attestor._paths import resolve_store_path
from attestor.core import AgentMemory

# Reuse the shim's claude -p invocation + usage ledger for a DIRECT
# (no-HTTP) subscription client. attestor's reflection._call_llm forces
# timeout=60 on the client call; a `claude -p` spawn on a big prompt can
# exceed that. Our duck-typed client ignores the forced timeout and runs
# claude with its own larger budget — so the scheduled runner is not
# bound by attestor's 60s ceiling.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("claude_sub_shim",
                                     os.path.expanduser("~/.attestor/claude_sub_shim.py"))
_shim = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_shim)

SAVINGS_LEDGER = os.path.expanduser("~/.attestor/distill_savings.jsonl")
SHIM_USAGE = os.path.expanduser("~/.attestor/claude_sub_usage.jsonl")
MIN_SOURCES = int(os.environ.get("ATTESTOR_DISTILL_MIN_SOURCES", "8"))
TARGET_COUNT = int(os.environ.get("ATTESTOR_DISTILL_TARGET", "5"))
SOURCE_LIMIT = int(os.environ.get("ATTESTOR_DISTILL_SOURCE_LIMIT", "30"))


class _Message:
    def __init__(self, content): self.content = content


class _Choice:
    def __init__(self, content): self.message = _Message(content); self.finish_reason = "stop"


class _Usage:
    def __init__(self, u):
        self.prompt_tokens = int(u.get("input_tokens", 0) or 0)
        self.completion_tokens = int(u.get("output_tokens", 0) or 0)
        self.total_tokens = self.prompt_tokens + self.completion_tokens


class _Response:
    def __init__(self, content, usage): self.choices = [_Choice(content)]; self.usage = _Usage(usage)


class _Completions:
    def create(self, *, model="sonnet", messages=None, timeout=None, **kw):  # noqa: ARG002
        msgs = messages or []
        sys_extra = "\n\n".join(str(m.get("content", "")) for m in msgs
                                if m.get("role") == "system").strip() or None
        prompt = "\n\n".join(str(m.get("content", "")) for m in msgs
                             if m.get("role") != "system").strip()
        import time as _t
        t = _t.time()
        env = _shim._call_claude(prompt, sys_extra, model)   # 180s budget; ignores timeout
        _shim._record(env, model, (_t.time() - t) * 1000.0)
        return _Response(env.get("result", ""), env.get("usage", {}) or {})


class _Chat:
    def __init__(self): self.completions = _Completions()


class ClaudeSubClient:
    """Duck-typed OpenAI client backed directly by `claude -p`."""
    def __init__(self): self.chat = _Chat()


def _toks(s: str) -> int:
    """Rough token estimate (~4 chars/token). Labeled as estimate."""
    return max(0, len((s or "")) // 4)


def _pg():
    conn = psycopg2.connect(host="localhost", port=5432, dbname="attestor",
                            user="postgres", password=os.environ.get("PGPASSWORD", ""))
    # Autocommit: this connection is read-only (tenant list + token counts).
    # Without it, psycopg2 opens an implicit transaction on the first SELECT
    # and leaves it `idle in transaction`, holding an AccessShareLock on
    # `memories` for the entire consolidate() call. Any concurrent schema
    # DDL (AccessExclusiveLock) then blocks behind it and a lock pile-up
    # wedges every subsequent AgentMemory() construction. Autocommit closes
    # each SELECT's transaction immediately, so no lock is held across calls.
    conn.autocommit = True
    return conn


def _tenants(conn) -> list[dict]:
    """Project tenants with their UUID, namespace, and un-distilled active count."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            "SELECT m.user_id::text AS uid, "
            "       COALESCE(m.metadata->>'_namespace','default') AS ns, "
            "       u.external_id AS ext, "
            "       COUNT(*) AS n "
            "FROM memories m JOIN users u ON u.id = m.user_id "
            "WHERE m.status='active' "
            "  AND (m.metadata->>'_consolidated_from') IS NULL "
            "GROUP BY 1,2,3 HAVING COUNT(*) >= %s "
            "ORDER BY n DESC",
            (MIN_SOURCES,),
        )
        return [dict(r) for r in cur.fetchall()]


def _active_content_tokens(conn, uid: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE user_id=%s AND status='active'", (uid,))
        return sum(_toks(r[0] or "") for r in cur.fetchall())


def _shim_tokens_spent() -> int:
    """Total subscription output+input tokens recorded by the shim so far."""
    total = 0
    if os.path.exists(SHIM_USAGE):
        with open(SHIM_USAGE) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    total += int(d.get("input_tokens", 0)) + int(d.get("output_tokens", 0))
                except Exception:  # noqa: BLE001
                    continue
    return total


def main() -> int:
    mem = AgentMemory(resolve_store_path())
    conn = _pg()
    tenants = _tenants(conn)
    print(f"{len(tenants)} tenant(s) with >= {MIN_SOURCES} un-distilled active rows\n")

    spent_before = _shim_tokens_spent()
    grand_saved = 0
    rows = []
    client = ClaudeSubClient()
    for t in tenants:
        uid, ns, ext, n = t["uid"], t["ns"], t["ext"], t["n"]
        before = _active_content_tokens(conn, uid)
        try:
            res = mem.consolidate(user_id=uid, namespace=ns,
                                  target_count=TARGET_COUNT, limit=SOURCE_LIMIT,
                                  dry_run=False, llm_client=client)
        except Exception as e:  # noqa: BLE001
            print(f"  {ext[:34]:36} FAILED: {type(e).__name__}: {e}")
            continue
        after = _active_content_tokens(conn, uid)
        saved = before - after
        grand_saved += max(0, saved)
        rows.append({"tenant": ext, "sources": len(res.source_memory_ids),
                     "facts": len(res.distilled_memory_ids),
                     "tokens_before": before, "tokens_after": after,
                     "tokens_saved_est": saved, "error": res.error})
        flag = "" if not res.error else f" (err={res.error})"
        print(f"  {ext[:34]:36} {n:3} rows -> superseded {len(res.source_memory_ids):3}, "
              f"+{len(res.distilled_memory_ids)} facts | ~{saved} tok saved{flag}")
    conn.close()

    spent = _shim_tokens_spent() - spent_before

    # Re-sync Pinecone to the new active set (drops superseded vectors,
    # adds distilled-fact vectors).
    print("\nRe-syncing vectors to active set...")
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "backfill_vectors", os.path.expanduser("~/.attestor/backfill_vectors.py"))
        bf = importlib.util.module_from_spec(spec); spec.loader.exec_module(bf)
        bf.main()
    except Exception as e:  # noqa: BLE001
        print(f"  (vector resync skipped: {type(e).__name__}: {e})")

    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "tenants": rows,
        "tokens_saved_est_total": grand_saved,
        "subscription_tokens_spent": spent,
    }
    with open(SAVINGS_LEDGER, "a") as f:
        f.write(json.dumps(entry) + "\n")

    print(f"\n=== SUMMARY ===")
    print(f"active-memory tokens removed (est): ~{grand_saved}")
    print(f"subscription tokens spent distilling: {spent}")
    if spent:
        print(f"removed-per-spent ratio: {grand_saved/max(1,spent):.1f}x "
              f"(pays back after ~{max(1,round(spent/max(1,grand_saved)))} recall(s) over this material)")
    print(f"ledger: {SAVINGS_LEDGER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
