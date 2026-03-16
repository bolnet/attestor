"""Tests for Claude Code lifecycle hooks."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from agent_memory.hooks.session_start import handle as session_start_handle
from agent_memory.hooks.post_tool_use import handle as post_tool_handle


# ---------------------------------------------------------------------------
# SessionStart hook
# ---------------------------------------------------------------------------

class TestSessionStartHook:
    def test_empty_store_returns_empty_context(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        memwright_path = store_path / ".memwright"
        payload = {"event": "SessionStart", "session_id": "s1", "cwd": str(store_path)}
        result = session_start_handle(payload)
        assert "additionalContext" in result
        assert result["additionalContext"] == ""

    def test_returns_context_when_memories_exist(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        memwright_path = store_path / ".memwright"

        # Pre-populate memories
        from agent_memory.core import AgentMemory
        mem = AgentMemory(str(memwright_path))
        mem.add("User prefers Python over JavaScript", tags=["preference"], category="preference")
        mem.add("Project uses FastAPI framework", tags=["tech"], category="project")
        mem.close()

        payload = {"event": "SessionStart", "session_id": "s1", "cwd": str(store_path)}
        result = session_start_handle(payload)
        assert "additionalContext" in result
        # Should contain some text from the stored memories
        assert len(result["additionalContext"]) > 0

    def test_uses_20k_token_budget(self, tmp_path, monkeypatch):
        """Verify the 20000 token budget is passed to recall_as_context."""
        store_path = tmp_path / "proj"
        store_path.mkdir()
        memwright_path = store_path / ".memwright"

        captured_budgets = []

        from agent_memory.core import AgentMemory
        original_recall = AgentMemory.recall_as_context

        def mock_recall(self, query, budget=None):
            captured_budgets.append(budget)
            return ""

        monkeypatch.setattr(AgentMemory, "recall_as_context", mock_recall)

        payload = {"event": "SessionStart", "session_id": "s1", "cwd": str(store_path)}
        session_start_handle(payload)
        assert 20000 in captured_budgets

    def test_malformed_json_returns_empty(self):
        result = session_start_handle({})
        assert result == {"additionalContext": ""}

    def test_missing_cwd_returns_empty(self):
        result = session_start_handle({"event": "SessionStart", "session_id": "s1"})
        assert result == {"additionalContext": ""}


# ---------------------------------------------------------------------------
# PostToolUse hook
# ---------------------------------------------------------------------------

class TestPostToolUseHook:
    def test_write_tool_captures_memory(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        memwright_path = store_path / ".memwright"

        payload = {
            "event": "PostToolUse",
            "session_id": "s1",
            "cwd": str(store_path),
            "tool": {
                "name": "Write",
                "input": {"file_path": "src/main.py", "content": "print('hello')"},
                "output": "File written successfully",
            },
        }
        result = post_tool_handle(payload)
        assert result == {}

        # Verify memory was stored
        from agent_memory.core import AgentMemory
        mem = AgentMemory(str(memwright_path))
        memories = mem.search(limit=10)
        mem.close()
        assert len(memories) >= 1
        assert any("src/main.py" in m.content for m in memories)

    def test_edit_tool_captures_memory(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        memwright_path = store_path / ".memwright"

        payload = {
            "event": "PostToolUse",
            "session_id": "s1",
            "cwd": str(store_path),
            "tool": {
                "name": "Edit",
                "input": {"file_path": "src/utils.py", "old_string": "foo", "new_string": "bar"},
                "output": "Edit applied",
            },
        }
        result = post_tool_handle(payload)
        assert result == {}

        from agent_memory.core import AgentMemory
        mem = AgentMemory(str(memwright_path))
        memories = mem.search(limit=10)
        mem.close()
        assert len(memories) >= 1
        assert any("src/utils.py" in m.content for m in memories)

    def test_bash_tool_captures_memory(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        memwright_path = store_path / ".memwright"

        payload = {
            "event": "PostToolUse",
            "session_id": "s1",
            "cwd": str(store_path),
            "tool": {
                "name": "Bash",
                "input": {"command": "npm install express"},
                "output": "added 50 packages",
            },
        }
        result = post_tool_handle(payload)
        assert result == {}

        from agent_memory.core import AgentMemory
        mem = AgentMemory(str(memwright_path))
        memories = mem.search(limit=10)
        mem.close()
        assert len(memories) >= 1
        assert any("npm install" in m.content for m in memories)

    def test_bash_read_only_commands_ignored(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()

        for cmd in ["cat file.txt", "ls -la", "head -5 f.py", "grep foo bar", "rg pattern"]:
            payload = {
                "event": "PostToolUse",
                "session_id": "s1",
                "cwd": str(store_path),
                "tool": {
                    "name": "Bash",
                    "input": {"command": cmd},
                    "output": "some output",
                },
            }
            result = post_tool_handle(payload)
            assert result == {}

    def test_read_tool_ignored(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()

        payload = {
            "event": "PostToolUse",
            "session_id": "s1",
            "cwd": str(store_path),
            "tool": {
                "name": "Read",
                "input": {"file_path": "src/main.py"},
                "output": "file contents here",
            },
        }
        result = post_tool_handle(payload)
        assert result == {}

    def test_unknown_tool_ignored(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()

        payload = {
            "event": "PostToolUse",
            "session_id": "s1",
            "cwd": str(store_path),
            "tool": {
                "name": "SomeUnknownTool",
                "input": {},
                "output": "whatever",
            },
        }
        result = post_tool_handle(payload)
        assert result == {}

    def test_malformed_payload_returns_empty(self):
        result = post_tool_handle({})
        assert result == {}

    def test_missing_tool_key_returns_empty(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        payload = {"event": "PostToolUse", "session_id": "s1", "cwd": str(store_path)}
        result = post_tool_handle(payload)
        assert result == {}
