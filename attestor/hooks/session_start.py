# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""SessionStart hook -- injects relevant memories into Claude Code session context."""

from __future__ import annotations

import concurrent.futures
import os
from typing import Any

from attestor import _branding as brand
from attestor.hooks._base import run_hook
from attestor.hooks._tenant import resolve_tenant

_EMPTY_RESPONSE: dict[str, Any] = {"additionalContext": ""}

# Budget for session context injection. Env override prevents context exhaustion.
_SESSION_BUDGET = int(os.environ.get(brand.ENV_TOKEN_BUDGET, "20000"))

# Broad query to surface the most useful memories at session start.
_SESSION_QUERY = "session context project overview recent decisions"

# Hard ceiling on how long the hook can take. A stalled Postgres or
# Neo4j must not block the host Claude Code session indefinitely; if
# we can't recall in this window, the host gets an empty response and
# moves on. Override with ATTESTOR_HOOK_TIMEOUT_S for slow networks.
_HOOK_TIMEOUT_S = float(os.environ.get("ATTESTOR_HOOK_TIMEOUT_S", "5.0"))


def _do_recall(cwd: str) -> dict[str, Any]:
    """The actual recall + pagerank pass — runs inside the timeout shim.

    Scopes the recall to the project that owns ``cwd`` so a session never
    surfaces another project's memories.
    """
    from attestor._paths import resolve_store_path
    from attestor.core import AgentMemory
    from attestor.retrieval.scorer import pagerank_boost

    store_path = resolve_store_path()
    mem = AgentMemory(store_path)
    try:
        user_id, namespace = resolve_tenant(mem, cwd)
        results = mem.recall(
            _SESSION_QUERY,
            budget=_SESSION_BUDGET,
            user_id=user_id,
            namespace=namespace,
        )

        pr_scores = mem.pagerank()
        if pr_scores and results:
            results = pagerank_boost(results, pr_scores, weight=0.3)
            results.sort(key=lambda r: r.score, reverse=True)

        if not results:
            return _EMPTY_RESPONSE

        lines = ["Relevant memories:"]
        for r in results:
            prefix = f"[{r.match_source}:{r.score:.2f}]"
            lines.append(f"- {prefix} {r.memory.content}")
        return {"additionalContext": "\n".join(lines)}
    finally:
        mem.close()


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    """Process a SessionStart event and return additionalContext."""
    try:
        cwd = payload.get("cwd")
        if not cwd:
            return _EMPTY_RESPONSE

        # Run the recall pass with a wall-clock deadline. ``ThreadPoolExecutor``
        # gives us ``future.result(timeout=...)`` which abandons the worker
        # thread on timeout — the underlying psycopg2 / Neo4j call leaks
        # for now, but the host session is unblocked. Pre-fix this hook
        # could hang the user's terminal forever on a stalled DB.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_recall, cwd)
            try:
                return fut.result(timeout=_HOOK_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                return _EMPTY_RESPONSE
    except Exception as exc:  # noqa: BLE001 -- handler must never crash the host
        from attestor.hooks._base import log_hook_error

        log_hook_error("session_start", exc)
        return _EMPTY_RESPONSE


def main() -> None:
    """CLI entry point: reads JSON from stdin, writes JSON to stdout."""
    run_hook("session_start", handle, empty_response=_EMPTY_RESPONSE)


if __name__ == "__main__":
    main()
