"""LLM-based memory extraction (optional, requires openai + OPENROUTER_API_KEY)."""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

from attestor.models import Memory

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_EXTRACTION_MODEL = "openai/gpt-4.1-mini"


def _get_client(api_key: Optional[str] = None):
    """Get an OpenAI client configured for OpenRouter."""
    from openai import OpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError(
            "OPENROUTER_API_KEY not set. Pass api_key or set the env var."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)

_EXTRACTION_PROMPT = """Extract atomic facts from this conversation that would be useful to remember across sessions.

For each fact, provide:
- content: A single, self-contained factual statement
- tags: List of relevant tags
- category: One of "career", "project", "preference", "personal", "technical", "general"
- entity: The primary entity this fact is about (company, person, tool, etc.), or null

Return a JSON array of objects. Only include facts that are clearly stated, not speculative.

Conversation:
{conversation}

Return ONLY valid JSON array, no other text."""

_SESSION_EXTRACTION_PROMPT = """Extract ALL factual information from this conversation between {speaker_a} and {speaker_b}.

Return a JSON object with two keys: "facts" and "relations".

## Facts
For each fact, provide:
- content: A single, atomic, self-contained factual statement. Always include the person's name (never use "she", "he", or "they").
  Examples: "Caroline is single", "Melanie has two children", "Caroline's dog is named Max"
- entity: The primary person or thing this fact is about (e.g., "Caroline", "Melanie")
- category: One of: "personal", "career", "preference", "event", "plan", "location", "health", "general"
- tags: Relevant keywords for search (e.g., ["marital_status", "relationship", "single"])
- event_date: The date this fact refers to if mentioned or inferrable (ISO format YYYY-MM-DD), or null
- confidence: 0.0-1.0 how explicitly stated (1.0 = directly said, 0.7 = strongly implied)

## Relations
Extract entity-relationship triples (subject-predicate-object). For each relation:
- subject: The source entity (person, place, or thing)
- predicate: The relationship type. Use one of: knows, works_at, lives_in, has, owns, is, likes, dislikes, visited, studies_at, member_of, related_to, married_to, sibling_of, parent_of, child_of, friend_of, colleague_of, born_in, moved_to, traveled_to, started, ended, plans_to, wants_to
- object: The target entity
- event_date: ISO date (YYYY-MM-DD) if the relation has a temporal aspect, or null

Examples:
  {{"subject": "Caroline", "predicate": "works_at", "object": "Google", "event_date": "2024-01-15"}}
  {{"subject": "Caroline", "predicate": "friend_of", "object": "Melanie", "event_date": null}}
  {{"subject": "Melanie", "predicate": "moved_to", "object": "New York", "event_date": "2024-03-01"}}

Session date: {session_date}

Rules:
- Extract EVERY factual detail, no matter how small (names, dates, places, numbers, opinions, plans, activities)
- Each fact must be self-contained and readable without the conversation
- Include the person's name in every fact
- Separate compound facts into individual atomic statements
- For temporal references like "next week" or "last month", resolve to dates relative to the session date
- Extract ALL relationships between people, places, organizations, and things mentioned
- Do NOT include greetings, conversational filler, or meta-commentary

Conversation:
{conversation}

Return ONLY a valid JSON object with "facts" and "relations" keys."""


def llm_extract(
    messages: List[Dict[str, Any]],
    model: str = DEFAULT_EXTRACTION_MODEL,
    api_key: Optional[str] = None,
) -> List[Memory]:
    """Extract memories using LLM via OpenRouter."""
    client = _get_client(api_key)

    conversation_text = "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in messages
        if msg.get("content")
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=2048,
        messages=[
            {
                "role": "user",
                "content": _EXTRACTION_PROMPT.format(conversation=conversation_text),
            }
        ],
    )

    facts = _parse_json_response(response.choices[0].message.content)
    return _facts_to_memories(facts)


def llm_extract_session(
    turns: List[Dict[str, Any]],
    speaker_a: str = "A",
    speaker_b: str = "B",
    session_date: str = "",
    model: str = DEFAULT_EXTRACTION_MODEL,
    api_key: Optional[str] = None,
) -> Tuple[List[Memory], List[Dict[str, Any]]]:
    """Extract memories and relation triples from a conversation session.

    Returns (memories, triples) where triples are dicts with
    subject, predicate, object, event_date keys.
    """
    client = _get_client(api_key)

    # Build conversation text
    conversation_lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        # Map speaker labels to names
        if speaker == "A":
            speaker = speaker_a
        elif speaker == "B":
            speaker = speaker_b
        text = turn.get("text", "")
        if text:
            conversation_lines.append(f"{speaker}: {text}")

    conversation_text = "\n".join(conversation_lines)

    prompt = _SESSION_EXTRACTION_PROMPT.format(
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        session_date=session_date,
        conversation=conversation_text,
    )

    response = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}],
    )

    parsed = _parse_extraction_response(response.choices[0].message.content)
    facts = parsed.get("facts", [])
    triples = parsed.get("relations", [])

    memories = _facts_to_memories(facts, default_event_date=session_date)
    # Ensure triples have required keys
    valid_triples = []
    for t in triples:
        if isinstance(t, dict) and "subject" in t and "predicate" in t and "object" in t:
            valid_triples.append({
                "subject": t["subject"],
                "predicate": t["predicate"],
                "object": t["object"],
                "event_date": t.get("event_date"),
            })

    return memories, valid_triples


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences from LLM response."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return text


def _parse_json_response(text: str) -> List[Dict[str, Any]]:
    """Parse JSON array from LLM response, stripping markdown fences if present.

    Handles both plain arrays and objects with a "facts" key (for backward compat).
    """
    text = _strip_markdown_fences(text)
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            # New format: {"facts": [...], "relations": [...]}
            return result.get("facts", [])
        return []
    except (json.JSONDecodeError, ValueError):
        return []


def _parse_extraction_response(text: str) -> Dict[str, Any]:
    """Parse the full extraction response with facts and relations.

    Returns {"facts": [...], "relations": [...]}.
    Handles both old format (plain array) and new format (object with keys).
    """
    text = _strip_markdown_fences(text)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return {
                "facts": result.get("facts", []),
                "relations": result.get("relations", []),
            }
        if isinstance(result, list):
            # Old format: just an array of facts, no relations
            return {"facts": result, "relations": []}
        return {"facts": [], "relations": []}
    except (json.JSONDecodeError, ValueError):
        return {"facts": [], "relations": []}


def _facts_to_memories(
    facts: List[Dict[str, Any]],
    default_event_date: Optional[str] = None,
) -> List[Memory]:
    """Convert extracted fact dicts to Memory objects."""
    memories = []
    for fact in facts:
        if isinstance(fact, dict) and "content" in fact:
            memories.append(
                Memory(
                    content=fact["content"],
                    tags=fact.get("tags", []),
                    category=fact.get("category", "general"),
                    entity=fact.get("entity"),
                    event_date=fact.get("event_date") or default_event_date,
                    confidence=fact.get("confidence", 1.0),
                )
            )
    return memories
