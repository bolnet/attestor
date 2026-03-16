"""PostToolUse hook -- captures observations from tool usage as memories."""

from __future__ import annotations

import json
import sys

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
        tool = payload.get("tool")
        if not cwd or not tool:
            return _EMPTY_RESPONSE

        tool_name = tool.get("name", "")
        tool_input = tool.get("input", {})
        tool_output = tool.get("output", "")

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
            store_path = f"{cwd}/.memwright"

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
