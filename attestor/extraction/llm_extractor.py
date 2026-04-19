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

Return a JSON object with FOUR keys: "facts", "relations", "entities", "concepts".

## Facts (atomic statements)
For each fact, provide:
- content: A single, atomic, self-contained factual statement. Always include the person's name (never use "she", "he", or "they").
  Examples: "Caroline is single", "Melanie has two children", "Caroline's dog is named Max"
- entity: The primary person or thing this fact is about (e.g., "Caroline", "Melanie")
- category: One of: "personal", "career", "preference", "event", "plan", "location", "health", "general"
- tags: Relevant keywords for search (e.g., ["marital_status", "relationship", "single"])
- event_date: The date this fact refers to if mentioned or inferrable (ISO format YYYY-MM-DD), or null
- confidence: 0.0-1.0 how explicitly stated (1.0 = directly said, 0.7 = strongly implied)

## Relations (entity-relationship triples)
For each relation:
- subject: The source entity (person, place, or thing)
- predicate: One of: knows, works_at, lives_in, has, owns, is, likes, dislikes, visited, studies_at, member_of, related_to, married_to, sibling_of, parent_of, child_of, friend_of, colleague_of, born_in, moved_to, traveled_to, started, ended, plans_to, wants_to
- object: The target entity
- event_date: ISO date (YYYY-MM-DD) if the relation has a temporal aspect, or null
- source_quote: Short verbatim quote from the conversation that supports this triple (<=150 chars), or null
- attributes: Optional JSON object of extra structured fields (e.g. {{"percentage": 0.9}}, {{"duration_years": 5}}), or null

Examples:
  {{"subject": "Caroline", "predicate": "works_at", "object": "Google", "event_date": "2024-01-15", "source_quote": "I started at Google last January", "attributes": null}}
  {{"subject": "Caroline", "predicate": "friend_of", "object": "Melanie", "event_date": null, "source_quote": "my friend Melanie and I", "attributes": null}}

## Entities (synthesized profile per person/thing)
For each distinct person, organization, or place mentioned, produce ONE profile aggregating everything known about them in this session:
- name: Canonical name (e.g. "Caroline", "Melanie", "TSMC")
- type: One of: "person", "organization", "place", "animal", "thing"
- profile: A paragraph (2-5 sentences) synthesizing who/what the entity is, based ONLY on this conversation. Include role, key traits, relationships, and recent activities. Always refer to the entity by name.
  Example: "Caroline is a transgender woman and LGBTQ activist working as a counselor. She is single, owns a dog named Max, and is close friends with Melanie. She is studying psychology and pursuing a counseling certification."
- tags: Keywords summarizing the entity (e.g. ["lgbtq", "counselor", "activist"])

## Concepts (synthesized themes / activity clusters / events)
For each recurring theme, activity cluster, or notable event in the conversation, produce ONE concept describing it:
- title: Short descriptive title (e.g. "Caroline's LGBTQ community participation", "Melanie's career transition to product management", "Joint trip to Barcelona 2023")
- description: A paragraph (2-5 sentences) synthesizing what this theme/activity/event is, who is involved, what happened, and when. Use names, not pronouns. Include specific dates, numbers, and places when stated.
  Example: "Caroline actively participates in the LGBTQ community through pride parades, a weekly support group she joined in March 2023, and an annual LGBTQ+ counseling conference. She attended pride events in May 2023 and serves as peer counselor in the support group."
- entities: List of entity names involved (e.g. ["Caroline", "Melanie"])
- tags: Keywords (e.g. ["lgbtq", "community", "activism"])
- event_date: ISO date for point-in-time events, or null for ongoing themes

Session date: {session_date}

Rules:
- Extract EVERY factual detail, no matter how small (names, dates, places, numbers, opinions, plans, activities)
- Each fact must be self-contained and readable without the conversation
- Include the person's name in every fact (never "she", "he", "they")
- Separate compound facts into individual atomic statements
- For temporal references like "next week" or "last month", resolve to dates relative to the session date
- Extract ALL relationships between people, places, organizations, and things mentioned
- Entity profiles and concepts must be SYNTHESIZED from the conversation, not copied verbatim
- A concept groups multiple related facts under one theme (e.g. "books Caroline is reading" rather than one concept per book)
- Do NOT include greetings, conversational filler, or meta-commentary

Conversation:
{conversation}

Return ONLY a valid JSON object with "facts", "relations", "entities", and "concepts" keys."""


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

    For richer output (entity profiles, concepts), see
    :func:`llm_extract_session_full`.
    """
    memories, triples, _, _ = llm_extract_session_full(
        turns=turns,
        speaker_a=speaker_a,
        speaker_b=speaker_b,
        session_date=session_date,
        model=model,
        api_key=api_key,
    )
    return memories, triples


def llm_extract_session_full(
    turns: List[Dict[str, Any]],
    speaker_a: str = "A",
    speaker_b: str = "B",
    session_date: str = "",
    model: str = DEFAULT_EXTRACTION_MODEL,
    api_key: Optional[str] = None,
) -> Tuple[
    List[Memory],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Extract facts + triples + entity profiles + concepts from a session.

    Returns (fact_memories, triples, entity_profiles, concept_profiles).
    - triples: {subject, predicate, object, event_date, source_quote, attributes}
    - entity_profiles: {name, type, profile, tags}
    - concept_profiles: {title, description, entities, tags, event_date}
    """
    client = _get_client(api_key)

    conversation_lines = []
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
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
        max_tokens=8192,
        messages=[{"role": "user", "content": prompt}],
    )

    parsed = _parse_extraction_response(response.choices[0].message.content)
    facts = parsed.get("facts", [])
    triples_raw = parsed.get("relations", [])
    entities_raw = parsed.get("entities", [])
    concepts_raw = parsed.get("concepts", [])

    memories = _facts_to_memories(facts, default_event_date=session_date)

    valid_triples: List[Dict[str, Any]] = []
    for t in triples_raw:
        if not (
            isinstance(t, dict)
            and "subject" in t
            and "predicate" in t
            and "object" in t
        ):
            continue
        valid_triples.append({
            "subject": t["subject"],
            "predicate": t["predicate"],
            "object": t["object"],
            "event_date": t.get("event_date"),
            "source_quote": t.get("source_quote"),
            "attributes": t.get("attributes"),
        })

    valid_entities: List[Dict[str, Any]] = []
    for e in entities_raw:
        if not (isinstance(e, dict) and e.get("name") and e.get("profile")):
            continue
        valid_entities.append({
            "name": str(e["name"]),
            "type": e.get("type", "entity"),
            "profile": str(e["profile"]),
            "tags": list(e.get("tags", [])),
        })

    valid_concepts: List[Dict[str, Any]] = []
    for c in concepts_raw:
        if not (isinstance(c, dict) and c.get("title") and c.get("description")):
            continue
        valid_concepts.append({
            "title": str(c["title"]),
            "description": str(c["description"]),
            "entities": list(c.get("entities", [])),
            "tags": list(c.get("tags", [])),
            "event_date": c.get("event_date"),
        })

    return memories, valid_triples, valid_entities, valid_concepts


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
    """Parse the full extraction response with facts, relations, entities, concepts.

    Returns {"facts": [...], "relations": [...], "entities": [...], "concepts": [...]}.
    Handles both old format (plain array) and new formats.
    """
    empty = {"facts": [], "relations": [], "entities": [], "concepts": []}
    text = _strip_markdown_fences(text)
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return {
                "facts": result.get("facts", []),
                "relations": result.get("relations", []),
                "entities": result.get("entities", []),
                "concepts": result.get("concepts", []),
            }
        if isinstance(result, list):
            return {**empty, "facts": result}
        return dict(empty)
    except (json.JSONDecodeError, ValueError):
        return dict(empty)


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
