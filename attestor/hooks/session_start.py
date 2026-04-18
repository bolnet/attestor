"""SessionStart hook -- injects relevant memories into Claude Code session context."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_EMPTY_RESPONSE = {"additionalContext": ""}

from attestor import _branding as brand

# Budget for session context injection. Env override prevents context exhaustion.
_SESSION_BUDGET = int(os.environ.get(brand.ENV_TOKEN_BUDGET, "20000"))

# Broad query to surface the most useful memories at session start.
_SESSION_QUERY = "session context project overview recent decisions"


def handle(payload: dict) -> dict:
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

        from attestor._paths import resolve_store_path

        store_path = resolve_store_path()

        from attestor.core import AgentMemory
        from attestor.retrieval.scorer import pagerank_boost

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
    except Exception:
        return _EMPTY_RESPONSE


def main():
    """CLI entry point: reads JSON from stdin, writes JSON to stdout."""
    try:
        payload = json.loads(sys.stdin.read())
        result = handle(payload)
    except Exception:
        result = _EMPTY_RESPONSE
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()


if __name__ == "__main__":
    main()
