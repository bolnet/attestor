"""Extraction prompts for the v4 conversation pipeline (roadmap §A.2/A.3).

The prompt content lives as versioned ``.md`` files alongside this module
and is loaded at import via ``attestor.extraction.prompt_loader``. The
public API (``USER_FACT_EXTRACTION_PROMPT``, ``format_user_fact_prompt``,
``PROMPT_VARS``, etc.) is preserved exactly as the inline-string version
exposed it, so existing callers and the prompt-content guard tests keep
working unchanged. To tune a prompt, edit the ``.md`` file and bump the
version (``user_fact_v1`` -> ``user_fact_v2``); the ``PROMPT_VERSIONS``
mapping records which template version produced each extracted fact so
audit replay can compare across versions.

Three prompts, each carrying a load-bearing constraint that must not
drift in future edits:

  USER_FACT_EXTRACTION_PROMPT
      Speaker-locked to the USER turn. The ``IMPORTANT`` line is the
      +53.6 single-session-assistant fix Mem0 published in April 2026 —
      without it, the extractor leaks assistant statements into the
      user-fact stream and audit becomes impossible.

  AGENT_FACT_EXTRACTION_PROMPT
      Speaker-locked to the ASSISTANT turn. Same ``IMPORTANT`` mechanism.
      Categories favor recommendation / decision / commitment / etc. so
      compliance review can audit what the agent said vs what it did.

  MEMORY_UPDATE_PROMPT
      Compares newly extracted facts against existing memories and
      decides ADD / UPDATE / INVALIDATE / NOOP per fact.
"""

from __future__ import annotations

from typing import Final

from attestor.extraction.prompt_loader import load_prompt, prompt_version

# ──────────────────────────────────────────────────────────────────────────
# Prompt name → version registry. Bump the suffix when editing the .md.
# ──────────────────────────────────────────────────────────────────────────

USER_FACT_PROMPT_NAME: Final = "user_fact_v1"
AGENT_FACT_PROMPT_NAME: Final = "agent_fact_v1"
MEMORY_UPDATE_PROMPT_NAME: Final = "memory_update_v1"
SIMPLE_EXTRACTION_PROMPT_NAME: Final = "simple_extraction_v1"
SESSION_EXTRACTION_PROMPT_NAME: Final = "session_extraction_v1"

PROMPT_VERSIONS: Final[dict[str, str]] = {
    "USER_FACT": prompt_version(USER_FACT_PROMPT_NAME),
    "AGENT_FACT": prompt_version(AGENT_FACT_PROMPT_NAME),
    "MEMORY_UPDATE": prompt_version(MEMORY_UPDATE_PROMPT_NAME),
    "SIMPLE_EXTRACTION": prompt_version(SIMPLE_EXTRACTION_PROMPT_NAME),
    "SESSION_EXTRACTION": prompt_version(SESSION_EXTRACTION_PROMPT_NAME),
}

# ──────────────────────────────────────────────────────────────────────────
# Format-template variable names. Tests assert these stay stable.
# ──────────────────────────────────────────────────────────────────────────

PROMPT_VARS: Final[dict[str, set[str]]] = {
    "USER_FACT": {"ts", "user_message", "recent_context_summary"},
    "AGENT_FACT": {"ts", "assistant_message", "recent_context_summary"},
    "MEMORY_UPDATE": {"existing_memories_json", "new_facts_json"},
}


# ──────────────────────────────────────────────────────────────────────────
# Raw template strings — kept as module-level constants for back-compat.
# These are loaded from disk once at import time; .format() is applied by
# the helpers below.
# ──────────────────────────────────────────────────────────────────────────

USER_FACT_EXTRACTION_PROMPT: Final = load_prompt(USER_FACT_PROMPT_NAME)
AGENT_FACT_EXTRACTION_PROMPT: Final = load_prompt(AGENT_FACT_PROMPT_NAME)
MEMORY_UPDATE_PROMPT: Final = load_prompt(MEMORY_UPDATE_PROMPT_NAME)


# ──────────────────────────────────────────────────────────────────────────
# Format helpers — same signatures the inline-string module exposed
# ──────────────────────────────────────────────────────────────────────────


def format_user_fact_prompt(
    ts: str,
    user_message: str,
    recent_context_summary: str = "(none)",
) -> str:
    """Render the USER fact extraction prompt."""
    return USER_FACT_EXTRACTION_PROMPT.format(
        ts=ts,
        user_message=user_message,
        recent_context_summary=recent_context_summary,
    )


def format_agent_fact_prompt(
    ts: str,
    assistant_message: str,
    recent_context_summary: str = "(none)",
) -> str:
    """Render the AGENT (assistant) fact extraction prompt."""
    return AGENT_FACT_EXTRACTION_PROMPT.format(
        ts=ts,
        assistant_message=assistant_message,
        recent_context_summary=recent_context_summary,
    )


def format_memory_update_prompt(
    existing_memories_json: str,
    new_facts_json: str,
) -> str:
    """Render the conflict resolution prompt."""
    return MEMORY_UPDATE_PROMPT.format(
        existing_memories_json=existing_memories_json,
        new_facts_json=new_facts_json,
    )


__all__ = [
    "AGENT_FACT_EXTRACTION_PROMPT",
    "AGENT_FACT_PROMPT_NAME",
    "MEMORY_UPDATE_PROMPT",
    "MEMORY_UPDATE_PROMPT_NAME",
    "PROMPT_VARS",
    "PROMPT_VERSIONS",
    "SESSION_EXTRACTION_PROMPT_NAME",
    "SIMPLE_EXTRACTION_PROMPT_NAME",
    "USER_FACT_EXTRACTION_PROMPT",
    "USER_FACT_PROMPT_NAME",
    "format_agent_fact_prompt",
    "format_memory_update_prompt",
    "format_user_fact_prompt",
    "load_prompt",
    "prompt_version",
]
