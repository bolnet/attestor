# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Stop hook -- ingests session transcript and stores key decisions as memories."""

from __future__ import annotations

import json
import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from attestor.hooks._base import run_hook
from attestor.hooks._tenant import resolve_tenant

_EMPTY_RESPONSE: dict[str, Any] = {}

# Maximum conversation turns to store per session (caps LLM/embed cost).
_MAX_TURNS = 30

# Minimum text length to bother storing.
_MIN_TEXT_LEN = 20

# Patterns that indicate a line is system noise, not a user statement.
_NOISE_PREFIXES = ("<", "/", "[Request interrupted", "MEMORY", "AUTO-MEMORY")
_NOISE_PATTERNS = re.compile(
    r"^(<[a-z]|/[a-z]|\[Request interrupted|\[local-command|\[system-reminder)"
)


def handle(payload: dict[str, Any]) -> dict[str, Any]:
    """Process a Stop event: ingest session transcript + summarise stored memories.

    Args:
        payload: JSON payload from Claude Code — keys: event, session_id, cwd.

    Returns:
        {} always (side-effect only).
    """
    try:
        cwd = payload.get("cwd")
        session_id = payload.get("session_id")
        if not cwd:
            return _EMPTY_RESPONSE

        from attestor._paths import resolve_store_path
        from attestor.core import AgentMemory

        store_path = resolve_store_path()
        mem = AgentMemory(store_path)
        try:
            user_id, namespace = resolve_tenant(mem, cwd)

            if user_id is not None and getattr(mem._store, "_v4", False):
                mem._resolve(user_id=user_id, autostart=False)

            # --- 1. Ingest the session transcript ---
            transcript_path = _find_transcript(session_id, cwd) if session_id else None
            turns = _extract_turns(transcript_path) if transcript_path else []
            if turns:
                _store_turns(turns, mem, user_id=user_id, namespace=namespace)

            # --- 2. Summarise memories already captured by PostToolUse ---
            one_hour_ago = (
                datetime.now(timezone.utc) - timedelta(hours=1)
            ).isoformat()
            recent = mem.search(after=one_hour_ago, limit=50, namespace=namespace)
            if recent:
                summary = _build_summary(recent)
                if summary:
                    mem.add(
                        summary,
                        tags=["session-summary"],
                        category="session",
                        user_id=user_id,
                        namespace=namespace,
                        scope="project",
                    )

            return _EMPTY_RESPONSE
        finally:
            mem.close()
    except Exception:  # noqa: BLE001 -- handler must never crash the host
        return _EMPTY_RESPONSE


# ---------------------------------------------------------------------------
# Transcript helpers
# ---------------------------------------------------------------------------

def _find_transcript(session_id: str, cwd: str) -> str | None:
    """Return the Claude Code transcript path for this session, or None."""
    home = os.path.expanduser("~")
    # ~/.claude/projects/<cwd-with-slashes-as-dashes>/<session_id>.jsonl
    slug = cwd.replace("/", "-")
    path = os.path.join(home, ".claude", "projects", slug, f"{session_id}.jsonl")
    return path if os.path.exists(path) else None


def _extract_turns(transcript_path: str) -> list[dict[str, str]]:
    """Parse JSONL transcript, return filtered list of {role, text} dicts."""
    turns: list[dict[str, str]] = []
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if obj.get("type") not in ("user", "assistant"):
                    continue
                if obj.get("isMeta"):
                    continue

                msg = obj.get("message", {})
                role = msg.get("role") or obj.get("type", "")
                content = msg.get("content", "")

                if isinstance(content, list):
                    text = " ".join(
                        block.get("text", "")
                        for block in content
                        if isinstance(block, dict) and block.get("type") == "text"
                    )
                else:
                    text = str(content)

                text = text.strip()
                if not text or len(text) < _MIN_TEXT_LEN:
                    continue
                if _NOISE_PATTERNS.match(text):
                    continue

                turns.append({"role": role, "text": text[:1000]})
    except OSError:
        pass
    return turns


def _store_turns(
    turns: list[dict[str, str]],
    mem: Any,
    *,
    user_id: str | None,
    namespace: str | None,
) -> None:
    """Store conversation turns as memories for future retrieval.

    Pairs user+assistant turns. User turns are stored individually
    (fact-dense); assistant turns are appended as context. Capped at
    _MAX_TURNS to bound embedding cost per session.
    """
    stored = 0
    i = 0
    while i < len(turns) and stored < _MAX_TURNS:
        turn = turns[i]
        if turn["role"] == "user":
            content = f"[User] {turn['text']}"
            # Append the next assistant response as context if present.
            if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                content += f"\n[Assistant] {turns[i + 1]['text'][:500]}"
                i += 2
            else:
                i += 1
            mem.add(
                content,
                tags=["conversation", "transcript"],
                category="conversation",
                user_id=user_id,
                namespace=namespace,
                scope="project",
            )
            stored += 1
        else:
            i += 1


# ---------------------------------------------------------------------------
# Stored-memory summary (unchanged from original)
# ---------------------------------------------------------------------------

def _build_summary(memories: list) -> str:
    """Build a rule-based summary from recent memories."""
    file_changes: list[str] = []
    commands: list[str] = []
    categories: dict[str, int] = {}

    for m in memories:
        categories[m.category] = categories.get(m.category, 0) + 1
        if "file-change" in m.tags:
            parts = m.content.split(" file ", 1)
            if len(parts) > 1:
                file_changes.append(parts[1].strip().split("\n")[0])
        if "command" in m.tags:
            commands.append(m.content)

    parts: list[str] = []
    if file_changes:
        file_list = ", ".join(file_changes[:10])
        if len(file_changes) > 10:
            file_list += f" (+{len(file_changes) - 10} more)"
        parts.append(f"{len(file_changes)} file changes ({file_list})")
    if commands:
        parts.append(f"{len(commands)} command{'s' if len(commands) != 1 else ''} run")
    if not parts:
        parts.append(f"{len(memories)} observations")

    cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(categories.items()))
    return f"Session summary: {', '.join(parts)}. Categories: {cat_str}."


def main() -> None:
    """CLI entry point: reads JSON from stdin, writes JSON to stdout."""
    run_hook("stop", handle, empty_response=_EMPTY_RESPONSE)


if __name__ == "__main__":
    main()
