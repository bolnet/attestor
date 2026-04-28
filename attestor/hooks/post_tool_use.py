"""PostToolUse hook -- captures observations from tool usage as memories."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

_EMPTY_RESPONSE = {}

# Read-only Bash command prefixes that should not be captured.
_READ_ONLY_PREFIXES = (
    "cat", "ls", "head", "tail", "echo", "grep", "find", "rg", "pwd", "which", "type",
)

# Secret patterns to redact BEFORE we persist any captured Bash command or
# tool output. The canonical leak vector this guards against: a captured
# command like `curl -H "Authorization: Bearer ey..." https://api.foo`
# would otherwise land in the memory store with the live token.
#
# Patterns cover: OpenAI / OpenRouter / Anthropic / Voyage / GitHub /
# AWS / GCP / generic bearer tokens / generic header-style "X-API-Key:".
_SECRET_PATTERNS = (
    # Authorization / bearer headers (any value after "Bearer " up to whitespace/quote)
    (re.compile(r"(?i)\b(authorization\s*:\s*bearer\s+)\S+", re.UNICODE),  r"\1<REDACTED>"),
    (re.compile(r"(?i)\b(x-api-key\s*:\s*)\S+",            re.UNICODE),   r"\1<REDACTED>"),
    # Provider-specific key prefixes (must come BEFORE the generic prefix=value rule)
    (re.compile(r"\bsk-or-v1-[A-Za-z0-9_-]{20,}"),  "<REDACTED:openrouter>"),
    (re.compile(r"\bsk-proj-[A-Za-z0-9_-]{20,}"),   "<REDACTED:openai>"),
    (re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}"),    "<REDACTED:anthropic>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{32,}"),        "<REDACTED:openai>"),
    (re.compile(r"\bpa-[A-Za-z0-9_-]{30,}"),        "<REDACTED:voyage>"),
    (re.compile(r"\bghp_[A-Za-z0-9]{30,}"),         "<REDACTED:github>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{30,}"), "<REDACTED:github>"),
    (re.compile(r"\bAKIA[A-Z0-9]{16}\b"),           "<REDACTED:aws>"),
    (re.compile(r"\bASIA[A-Z0-9]{16}\b"),           "<REDACTED:aws>"),
    (re.compile(r"\bya29\.[A-Za-z0-9_-]{20,}"),     "<REDACTED:gcp>"),
    # Env-export style: KEY=value where KEY ends in _TOKEN / _KEY / _SECRET / _PASSWORD
    (re.compile(r"(?i)\b([A-Z][A-Z0-9_]*(?:_TOKEN|_KEY|_SECRET|_PASSWORD)\s*=\s*)\S+"),
                                                    r"\1<REDACTED>"),
)


def _redact_secrets(text: str) -> str:
    """Apply all secret-pattern redactions to ``text``.

    Order matters: provider-specific patterns run before the generic
    ``KEY=value`` pattern so a literal ``OPENAI_API_KEY=sk-proj-...``
    becomes ``OPENAI_API_KEY=<REDACTED>`` (key name preserved, value
    redacted) rather than the looser fallback.
    """
    if not text:
        return text
    for pattern, replacement in _SECRET_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


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

            # Redact secrets BEFORE building the captured content. A
            # `curl -H "Authorization: Bearer ..."` would otherwise land
            # in the memory store with the live token. Same for any
            # `OPENAI_API_KEY=sk-...` style export, AWS access keys,
            # GitHub PATs, etc.
            command = _redact_secrets(command)
            output_summary = _redact_secrets((tool_output or "")[:200])
            content = f"Ran command: {command}"
            if output_summary:
                content += f"\nOutput: {output_summary}"
            tags = ["command"]

        else:
            # Read tool and unknown tools -- silently ignore
            return _EMPTY_RESPONSE

        if content:
            from attestor._paths import resolve_store_path

            store_path = resolve_store_path()

            from attestor.core import AgentMemory

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
