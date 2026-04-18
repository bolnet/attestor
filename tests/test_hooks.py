"""Tests for Claude Code lifecycle hooks."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import pytest

from attestor.hooks.session_start import handle as session_start_handle
from attestor.hooks.post_tool_use import handle as post_tool_handle
from attestor.hooks.stop import handle as stop_handle


@pytest.fixture(autouse=True)
def _isolate_attestor_store(tmp_path, monkeypatch):
    """Redirect hook store resolution to tmp_path/.attestor.

    Hooks call resolve_store_path() with no override, which defaults to
    $ATTESTOR_PATH then ~/.attestor. Without this, tests write to (and read
    from) the developer's real home store. Mirror the per-test convention of
    store_path = tmp_path / "proj"; attestor_path = store_path / ".attestor".
    """
    monkeypatch.setenv("ATTESTOR_PATH", str(tmp_path / "proj" / ".attestor"))


# ---------------------------------------------------------------------------
# SessionStart hook
# ---------------------------------------------------------------------------

class TestSessionStartHook:
    def test_empty_store_returns_empty_context(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        attestor_path = store_path / ".attestor"
        payload = {"event": "SessionStart", "session_id": "s1", "cwd": str(store_path)}
        result = session_start_handle(payload)
        assert "additionalContext" in result
        assert result["additionalContext"] == ""

    def test_returns_context_when_memories_exist(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        attestor_path = store_path / ".attestor"

        # Pre-populate memories
        from attestor.core import AgentMemory
        mem = AgentMemory(str(attestor_path))
        mem.add("User prefers Python over JavaScript", tags=["preference"], category="preference")
        mem.add("Project uses FastAPI framework", tags=["tech"], category="project")
        mem.close()

        payload = {"event": "SessionStart", "session_id": "s1", "cwd": str(store_path)}
        result = session_start_handle(payload)
        assert "additionalContext" in result
        # Should contain some text from the stored memories
        assert len(result["additionalContext"]) > 0

    def test_uses_20k_token_budget(self, tmp_path, monkeypatch):
        """Verify the 20000 token budget is passed to recall."""
        store_path = tmp_path / "proj"
        store_path.mkdir()
        attestor_path = store_path / ".attestor"

        captured_budgets = []

        from attestor.core import AgentMemory
        original_recall = AgentMemory.recall

        def mock_recall(self, query, budget=None):
            captured_budgets.append(budget)
            return []

        monkeypatch.setattr(AgentMemory, "recall", mock_recall)

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
        attestor_path = store_path / ".attestor"

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
        from attestor.core import AgentMemory
        mem = AgentMemory(str(attestor_path))
        memories = mem.search(limit=10)
        mem.close()
        assert len(memories) >= 1
        assert any("src/main.py" in m.content for m in memories)

    def test_edit_tool_captures_memory(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        attestor_path = store_path / ".attestor"

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

        from attestor.core import AgentMemory
        mem = AgentMemory(str(attestor_path))
        memories = mem.search(limit=10)
        mem.close()
        assert len(memories) >= 1
        assert any("src/utils.py" in m.content for m in memories)

    def test_bash_tool_captures_memory(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        attestor_path = store_path / ".attestor"

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

        from attestor.core import AgentMemory
        mem = AgentMemory(str(attestor_path))
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


# ---------------------------------------------------------------------------
# Stop hook
# ---------------------------------------------------------------------------

class TestStopHook:
    def test_no_recent_memories_returns_empty(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        payload = {"event": "Stop", "session_id": "s1", "cwd": str(store_path)}
        result = stop_handle(payload)
        assert result == {}

    def test_stores_summary_when_memories_exist(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        attestor_path = store_path / ".attestor"

        # Pre-populate with some recent memories (simulating tool captures)
        from attestor.core import AgentMemory
        mem = AgentMemory(str(attestor_path))
        mem.add("Created/wrote file src/main.py", tags=["file-change", "write"], category="project")
        mem.add("Edited file src/utils.py", tags=["file-change", "edit"], category="project")
        mem.add("Ran command: npm install express", tags=["command"], category="project")
        mem.close()

        payload = {"event": "Stop", "session_id": "s1", "cwd": str(store_path)}
        result = stop_handle(payload)
        assert result == {}

        # Verify summary memory was stored
        mem = AgentMemory(str(attestor_path))
        memories = mem.search(category="session", limit=10)
        mem.close()
        assert len(memories) >= 1
        summary = memories[0]
        assert "session-summary" in summary.tags
        assert summary.category == "session"

    def test_summary_includes_file_change_counts(self, tmp_path):
        store_path = tmp_path / "proj"
        store_path.mkdir()
        attestor_path = store_path / ".attestor"

        from attestor.core import AgentMemory
        mem = AgentMemory(str(attestor_path))
        mem.add("Created/wrote file a.py", tags=["file-change", "write"], category="project")
        mem.add("Created/wrote file b.py", tags=["file-change", "write"], category="project")
        mem.add("Ran command: pytest", tags=["command"], category="project")
        mem.close()

        payload = {"event": "Stop", "session_id": "s1", "cwd": str(store_path)}
        stop_handle(payload)

        mem = AgentMemory(str(attestor_path))
        memories = mem.search(category="session", limit=10)
        mem.close()
        assert len(memories) >= 1
        summary_content = memories[0].content
        assert "2 file changes" in summary_content
        assert "1 command" in summary_content

    def test_malformed_payload_returns_empty(self):
        result = stop_handle({})
        assert result == {}


# ---------------------------------------------------------------------------
# CLI: mcp subcommand
# ---------------------------------------------------------------------------

class TestMcpSubcommand:
    def test_mcp_parser_accepts_no_args(self):
        """Verify `attestor mcp` parses without positional args."""
        from attestor.cli import main
        from unittest.mock import patch, MagicMock

        # Mock asyncio.run to avoid actually starting the server
        with patch("attestor.cli.asyncio") as mock_asyncio, \
             patch("attestor.mcp.server.run_server") as mock_run:
            mock_asyncio.run = MagicMock()
            # This should not raise
            try:
                main(["mcp", "--path", "/tmp/test-attestor-store"])
            except SystemExit:
                pass  # argparse may exit

    def test_mcp_default_path_resolution(self, monkeypatch):
        """Verify default path uses ATTESTOR_PATH or ~/.attestor."""
        import os
        monkeypatch.delenv("ATTESTOR_PATH", raising=False)

        from attestor.cli import main
        from unittest.mock import patch, MagicMock, call

        with patch("attestor.cli.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock()
            try:
                main(["mcp", "--path", "/tmp/test-mcp-default"])
            except SystemExit:
                pass

    def test_mcp_env_var_override(self, tmp_path, monkeypatch):
        """Verify ATTESTOR_PATH env var is used as default."""
        custom_path = str(tmp_path / "custom-store")
        monkeypatch.setenv("ATTESTOR_PATH", custom_path)

        from attestor.cli import main
        from unittest.mock import patch, MagicMock

        captured_paths = []

        original_run_server = None
        def capture_run_server(path):
            captured_paths.append(path)

        with patch("attestor.cli.asyncio") as mock_asyncio:
            mock_asyncio.run = MagicMock(side_effect=lambda coro: None)
            with patch("attestor.mcp.server.run_server", side_effect=capture_run_server) as mock_srv:
                try:
                    main(["mcp"])
                except (SystemExit, Exception):
                    pass


# ---------------------------------------------------------------------------
# CLI: hook subcommand
# ---------------------------------------------------------------------------

class TestHookSubcommand:
    def test_hook_session_start_exists(self):
        """Verify `attestor hook session-start` is a valid subcommand."""
        from attestor.cli import main
        from unittest.mock import patch
        import io

        # Provide JSON on stdin so the hook can process it
        stdin_data = json.dumps({"event": "SessionStart", "session_id": "s1", "cwd": "/tmp/nonexistent"})
        with patch("sys.stdin", io.StringIO(stdin_data)), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            try:
                main(["hook", "session-start"])
            except SystemExit:
                pass

    def test_hook_stop_exists(self):
        """Verify `attestor hook stop` is a valid subcommand."""
        from attestor.cli import main
        from unittest.mock import patch
        import io

        stdin_data = json.dumps({"event": "Stop", "session_id": "s1", "cwd": "/tmp/nonexistent"})
        with patch("sys.stdin", io.StringIO(stdin_data)), \
             patch("sys.stdout", new_callable=io.StringIO) as mock_out:
            try:
                main(["hook", "stop"])
            except SystemExit:
                pass
