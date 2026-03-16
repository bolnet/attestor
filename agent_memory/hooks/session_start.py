"""SessionStart hook -- injects relevant memories into Claude Code session context."""

from __future__ import annotations

import json
import sys

_EMPTY_RESPONSE = {"additionalContext": ""}

# Budget per user decision: 20K tokens for session context injection.
_SESSION_BUDGET = 20000

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

        store_path = f"{cwd}/.memwright"

        from agent_memory.core import AgentMemory

        mem = AgentMemory(store_path)
        try:
            context = mem.recall_as_context(_SESSION_QUERY, budget=_SESSION_BUDGET)
            return {"additionalContext": context or ""}
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
