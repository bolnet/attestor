"""Extraction prompts for the v4 conversation pipeline (roadmap §A.2/A.3).

Three prompts, each carrying a load-bearing constraint that must not
drift in future edits:

  USER_FACT_EXTRACTION_PROMPT
      Speaker-locked to the USER turn. The `IMPORTANT` line is the +53.6
      single-session-assistant fix Mem0 published in April 2026 — without
      it, the extractor leaks assistant statements into the user-fact
      stream and audit becomes impossible.

  AGENT_FACT_EXTRACTION_PROMPT
      Speaker-locked to the ASSISTANT turn. Same `IMPORTANT` mechanism.
      Categories favor recommendation / decision / commitment / etc. so
      compliance review can audit what the agent said vs what it did.

  MEMORY_UPDATE_PROMPT
      Compares newly extracted facts against existing memories and
      decides ADD / UPDATE / INVALIDATE / NOOP per fact. Adds an
      `evidence_episode_id` requirement so every decision links back to
      the source episode for replay.

These strings are templated with `str.format` — placeholder names are
defined in `PROMPT_VARS` so test_extraction_prompts.py can guard
against accidental renames.
"""

from __future__ import annotations

# Format-template variable names. Tests assert these stay stable.
PROMPT_VARS = {
    "USER_FACT": {"ts", "user_message", "recent_context_summary"},
    "AGENT_FACT": {"ts", "assistant_message", "recent_context_summary"},
    "MEMORY_UPDATE": {"existing_memories_json", "new_facts_json"},
}


USER_FACT_EXTRACTION_PROMPT = """\
You are a Personal Information Organizer for a multi-agent system. You extract
durable facts from user messages so that other agents can act on them later.

What to extract:
1. Personal preferences (likes, dislikes, choices)
2. Important personal details (names, relationships, dates, locations)
3. Plans and intentions (upcoming events, goals, commitments)
4. Activity and service preferences
5. Health and wellness information
6. Professional details (employer, role, responsibilities)
7. Financial details (accounts, risk tolerance, dependents) -- preserve exact figures
8. Anything the user explicitly asks you to remember

What NOT to extract:
- Pleasantries, greetings, sign-offs
- Questions the user asks (those are intents, not facts)
- Hypotheticals ("what if I...")
- Things the assistant said (a separate prompt handles those)

Output schema (JSON only, no prose):
{{
  "facts": [
    {{
      "text": "<atomic fact, <= 25 words, third person>",
      "category": "<preference|career|project|technical|personal|location|relationship|event|financial>",
      "entity": "<primary entity, e.g. 'Acme Corp', 'Python', 'spouse'>",
      "confidence": <0.0-1.0>,
      "source_span": [<start_char>, <end_char>]
    }}
  ]
}}

# IMPORTANT: GENERATE FACTS SOLELY BASED ON THE USER'S MESSAGE BELOW.
# Detect the input language and emit facts in that same language.
# If no durable facts are present, return {{"facts": []}}.

User message (timestamp: {ts}):
\"\"\"
{user_message}
\"\"\"

Recent thread context (for entity disambiguation only -- DO NOT extract from this):
{recent_context_summary}

Output:
"""


AGENT_FACT_EXTRACTION_PROMPT = """\
You are a Decision Recorder for a multi-agent system. You extract durable
statements made by an AI agent so they can be honored by future agents and
audited by humans.

What to extract:
1. Recommendations and advice given (with rationale)
2. Decisions made or actions taken
3. Commitments to the user ("I'll do X", "I won't share Y")
4. Constraints or rules the agent applied
5. Numeric outputs the user may rely on (figures, dates, calculations)
6. Refusals and the reason for them

What NOT to extract:
- Greetings, hedges, conversational filler
- Questions asked of the user
- The user's own statements (a separate prompt handles those)

Output schema (JSON only):
{{
  "facts": [
    {{
      "text": "<atomic statement, <= 30 words, third person; preserve exact figures>",
      "category": "<recommendation|decision|commitment|constraint|calculation|refusal>",
      "entity": "<primary entity>",
      "confidence": <0.0-1.0>,
      "source_span": [<start>, <end>]
    }}
  ]
}}

# IMPORTANT: GENERATE FACTS SOLELY BASED ON THE ASSISTANT'S MESSAGE BELOW.
# This is critical for audit: assistant statements drive compliance review.

Assistant message (timestamp: {ts}):
\"\"\"
{assistant_message}
\"\"\"

Recent thread context (entity disambiguation only):
{recent_context_summary}

Output:
"""


MEMORY_UPDATE_PROMPT = """\
You are the memory manager for a multi-agent system. Compare newly extracted
facts against existing memories and decide what to do with each new fact.

Operations:
- ADD     : new information; no existing memory covers it
- UPDATE  : same topic + entity, refined or more complete content (KEEP THE
            EXISTING ID; preserve audit trail)
- INVALIDATE: existing memory is contradicted by the new fact; mark the old
            one superseded but DO NOT delete it (the timeline must replay)
- NOOP    : new fact is already represented; do nothing

Decision rules:
1. Two facts about the SAME entity + SAME predicate with DIFFERENT values ->
   newer wins -> INVALIDATE old, ADD new (with `supersedes` pointer).
2. Two facts about the SAME entity + SAME predicate with REFINED value
   (e.g. "works at Google" -> "Senior Engineer at Google") -> UPDATE the
   existing memory keeping its ID.
3. Same content, different phrasing -> NOOP.
4. Brand new entity or predicate -> ADD.

Output schema (JSON only):
{{
  "decisions": [
    {{
      "operation": "ADD|UPDATE|INVALIDATE|NOOP",
      "new_fact": {{ ...the new fact object... }},
      "existing_id": "<id-or-null>",
      "rationale": "<one sentence>",
      "evidence_episode_id": "<source episode id>"
    }}
  ]
}}

Existing memories (top-k similar to the new facts, with their IDs):
{existing_memories_json}

New facts to evaluate:
{new_facts_json}

Output:
"""


def format_user_fact_prompt(
    ts: str, user_message: str, recent_context_summary: str = "(none)",
) -> str:
    """Render the USER fact extraction prompt."""
    return USER_FACT_EXTRACTION_PROMPT.format(
        ts=ts,
        user_message=user_message,
        recent_context_summary=recent_context_summary,
    )


def format_agent_fact_prompt(
    ts: str, assistant_message: str, recent_context_summary: str = "(none)",
) -> str:
    """Render the AGENT (assistant) fact extraction prompt."""
    return AGENT_FACT_EXTRACTION_PROMPT.format(
        ts=ts,
        assistant_message=assistant_message,
        recent_context_summary=recent_context_summary,
    )


def format_memory_update_prompt(
    existing_memories_json: str, new_facts_json: str,
) -> str:
    """Render the conflict resolution prompt."""
    return MEMORY_UPDATE_PROMPT.format(
        existing_memories_json=existing_memories_json,
        new_facts_json=new_facts_json,
    )
