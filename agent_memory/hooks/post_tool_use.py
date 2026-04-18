"""PostToolUse hook -- captures observations from tool usage as memories."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

_EMPTY_RESPONSE = {}

# Read-only Bash command prefixes that should not be captured.
_READ_ONLY_PREFIXES = (
    "cat", "ls", "head", "tail", "echo", "grep", "find", "rg", "pwd", "which", "type",
)


def handle(payload: dict) -> dict:
    """Process a PostToolUse event and store observations as memories.

    Args:
        payload: JSON payload from Claude Code with keys: event, session_id, cwd, tool.

    Returns:
        {} always (side-effect only).
    """
    try:
        cwd = payload.get("cwd")
        # Claude Code sends tool_name / tool_input / tool_response at top level.
        # Older/alternate shape nests them under "tool".
        tool_name = payload.get("tool_name") or (payload.get("tool") or {}).get("name", "")
        tool_input = payload.get("tool_input") or (payload.get("tool") or {}).get("input", {}) or {}
        tool_output = (
            payload.get("tool_response")
            or payload.get("tool_output")
            or (payload.get("tool") or {}).get("output", "")
            or ""
        )
        if isinstance(tool_output, dict):
            tool_output = json.dumps(tool_output)[:500]
        if not cwd or not tool_name:
            return _EMPTY_RESPONSE

        content = None
        tags = []
        category = "project"

        if tool_name == "Write":
            file_path = tool_input.get("file_path", "unknown")
            content = f"Created/wrote file {file_path}"
            tags = ["file-change", "write"]

        elif tool_name == "Edit":
            file_path = tool_input.get("file_path", "unknown")
            content = f"Edited file {file_path}"
            tags = ["file-change", "edit"]

        elif tool_name == "Bash":
            command = tool_input.get("command", "")
            if not command:
                return _EMPTY_RESPONSE

            # Skip read-only commands
            first_word = command.strip().split()[0] if command.strip() else ""
            if first_word in _READ_ONLY_PREFIXES:
                return _EMPTY_RESPONSE

            output_summary = (tool_output or "")[:200]
            content = f"Ran command: {command}"
            if output_summary:
                content += f"\nOutput: {output_summary}"
            tags = ["command"]

        else:
            # Read tool and unknown tools -- silently ignore
            return _EMPTY_RESPONSE

        if content:
            from agent_memory._paths import resolve_store_path

            store_path = resolve_store_path()

            from agent_memory.core import AgentMemory

            mem = AgentMemory(store_path)
            try:
                mem.add(content, tags=tags, category=category)
            finally:
                mem.close()

        return _EMPTY_RESPONSE
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
