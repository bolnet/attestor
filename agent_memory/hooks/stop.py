"""Stop hook -- summarizes session and stores key decisions as memories."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_EMPTY_RESPONSE = {}


def handle(payload: dict) -> dict:
    """Process a Stop event: summarize session observations and store as memory.

    Args:
        payload: JSON payload from Claude Code with keys: event, session_id, cwd.

    Returns:
        {} always (side-effect only).
    """
    try:
        cwd = payload.get("cwd")
        if not cwd:
            return _EMPTY_RESPONSE

        store_path = os.environ.get(
            "MEMWRIGHT_PATH",
            str(Path.home() / ".memwright"),
        )

        from agent_memory.core import AgentMemory

        mem = AgentMemory(store_path)
        try:
            # Query memories from the last hour (this session's captures)
            one_hour_ago = (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat()
            recent = mem.search(after=one_hour_ago, limit=50)

            if not recent:
                return _EMPTY_RESPONSE

            # Build rule-based summary
            summary = _build_summary(recent)
            if summary:
                mem.add(summary, tags=["session-summary"], category="session")

            return _EMPTY_RESPONSE
        finally:
            mem.close()
    except Exception:
        return _EMPTY_RESPONSE


def _build_summary(memories: list) -> str:
    """Build a rule-based summary from recent memories.

    Groups by category, counts file changes and commands.
    """
    file_changes = []
    commands = []
    categories: dict[str, int] = {}

    for m in memories:
        # Count categories
        categories[m.category] = categories.get(m.category, 0) + 1

        # Detect file changes by tags
        if "file-change" in m.tags:
            # Extract file path from content like "Created/wrote file X" or "Edited file X"
            parts = m.content.split(" file ", 1)
            if len(parts) > 1:
                file_path = parts[1].strip().split("\n")[0]
                file_changes.append(file_path)

        # Detect commands by tags
        if "command" in m.tags:
            commands.append(m.content)

    parts = []
    if file_changes:
        file_list = ", ".join(file_changes[:10])
        if len(file_changes) > 10:
            file_list += f" (+{len(file_changes) - 10} more)"
        parts.append(f"{len(file_changes)} file changes ({file_list})")
    if commands:
        parts.append(f"{len(commands)} command{'s' if len(commands) != 1 else ''} run")

    if not parts:
        # There were memories but none were file changes or commands
        parts.append(f"{len(memories)} observations")

    cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(categories.items()))
    summary = f"Session summary: {', '.join(parts)}. Categories: {cat_str}."
    return summary


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
