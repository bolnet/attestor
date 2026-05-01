"""SessionStart hook -- injects relevant memories into Claude Code session context."""

from __future__ import annotations

import os
from typing import Any

from attestor import _branding as brand
from attestor.hooks._base import run_hook

_EMPTY_RESPONSE: dict[str, Any] = {"additionalContext": ""}

# Budget for session context injection. Env override prevents context exhaustion.
_SESSION_BUDGET = int(os.environ.get(brand.ENV_TOKEN_BUDGET, "20000"))

# Broad query to surface the most useful memories at session start.
_SESSION_QUERY = "session context project overview recent decisions"


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    """Process a SessionStart event and return additionalContext.

    Args:
        payload: JSON payload from Claude Code with keys: event, session_id, cwd.

    Returns:
        {"additionalContext": str} -- context string to inject, or empty string.
    """
    try:
        cwd = payload.get("cwd")
        if not cwd:
            return _EMPTY_RESPONSE

        # Imports are kept lazy: AgentMemory pulls in heavy backends and we
        # MUST NOT crash on import (a hook import error would take down the
        # host session). Construct only after we know we have work to do.
        from attestor._paths import resolve_store_path
        from attestor.core import AgentMemory
        from attestor.retrieval.scorer import pagerank_boost

        store_path = resolve_store_path()

        mem = AgentMemory(store_path)
        try:
            results = mem.recall(_SESSION_QUERY, budget=_SESSION_BUDGET)

            # Boost results by PageRank importance
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
    except Exception:  # noqa: BLE001 -- handler must never crash the host
        return _EMPTY_RESPONSE


def main() -> None:
    """CLI entry point: reads JSON from stdin, writes JSON to stdout."""
    run_hook("session_start", handle, empty_response=_EMPTY_RESPONSE)


if __name__ == "__main__":
    main()
