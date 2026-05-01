"""LLM-based memory extraction (optional; requires the ``openai`` package
and the API key for whichever provider is configured under
``stack.llm.providers`` in ``configs/attestor.yaml``).

The two LLM prompts used here live as versioned ``.md`` files under
``attestor/extraction/prompts/`` (``simple_extraction_v1.md`` and
``session_extraction_v1.md``). Each extracted ``Memory`` records the
prompt template version in its metadata so audit replay can correlate a
fact back to the exact template that produced it.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from attestor.extraction.prompt_loader import load_prompt, prompt_version
from attestor.models import Memory

logger = logging.getLogger("attestor.extraction.llm_extractor")

# Prompt template names (stems of the .md files under prompts/). Bump
# the suffix when the .md content changes — `prompt_version()` reads
# the suffix and we record it in extracted-fact metadata so audit replay
# can correlate a fact back to the exact template that produced it.
SIMPLE_EXTRACTION_PROMPT_NAME = "simple_extraction_v1"
SESSION_EXTRACTION_PROMPT_NAME = "session_extraction_v1"


def _default_extraction_model() -> str:
    """Resolve the extraction model from ``configs/attestor.yaml``."""
    from attestor.config import get_stack
    return get_stack().models.extraction


# Backwards-compatible attribute access — code that still references
# this constant gets a string at import time. New code should call
# ``_default_extraction_model()`` so config changes take effect without
# a process restart.
DEFAULT_EXTRACTION_MODEL = _default_extraction_model()


def _resolve_client(model: str, api_key: str | None = None) -> tuple[Any, str]:
    """Resolve ``(client, clean_model)`` via the YAML-driven LLM pool.

    YAML is the only source of truth for provider ``base_url`` /
    ``api_key_env``. The ``api_key`` parameter is kept for back-compat:
    when set, a fresh client is built against the pool-resolved
    provider's ``base_url`` (no env fallback).
    """
    from attestor.llm_trace import get_client_for_model, _get_pool, make_client

    if api_key:
        pool = _get_pool()
        head, sep, tail = model.partition("/")
        if sep and head in pool.providers:
            strategy = pool._strategies[head]  # noqa: SLF001 — explicit override path
            clean_model = tail
        else:
            strategy = pool.default_strategy()
            clean_model = model
        client = make_client(base_url=strategy.base_url, api_key=api_key)
        return client, clean_model

    return get_client_for_model(model)


# Prompt templates are loaded once at import from the externalized .md
# files. Bumping a prompt is a content edit + a v1 → v2 rename in
# ``attestor/extraction/prompts/`` and a single-line constant update
# above — no code change in the extraction pipeline.
_EXTRACTION_PROMPT = load_prompt(SIMPLE_EXTRACTION_PROMPT_NAME)
_SESSION_EXTRACTION_PROMPT = load_prompt(SESSION_EXTRACTION_PROMPT_NAME)
_SIMPLE_EXTRACTION_PROMPT_VERSION = prompt_version(SIMPLE_EXTRACTION_PROMPT_NAME)
_SESSION_EXTRACTION_PROMPT_VERSION = prompt_version(SESSION_EXTRACTION_PROMPT_NAME)


def llm_extract(
    messages: list[dict[str, Any]],
    model: str = DEFAULT_EXTRACTION_MODEL,
    api_key: str | None = None,
) -> list[Memory]:
    """Extract memories using the YAML-configured LLM provider."""
    client, clean_model = _resolve_client(model, api_key)

    conversation_text = "\n".join(
        f"{msg.get('role', 'unknown')}: {msg.get('content', '')}"
        for msg in messages
        if msg.get("content")
    )

    from attestor.llm_trace import traced_create
    response = traced_create(
        client,
        role="extraction",
        model=clean_model,
        max_tokens=2048,
        timeout=30,
        max_retries=0,
        messages=[
            {
                "role": "user",
                "content": _EXTRACTION_PROMPT.format(conversation=conversation_text),
            }
        ],
    )

    if not response.choices or not response.choices[0].message:
        logger.warning("LLM returned empty response; skipping extraction")
        return []
    content = response.choices[0].message.content or ""
    facts = _parse_json_response(content)
    return _facts_to_memories(
        facts,
        prompt_template=SIMPLE_EXTRACTION_PROMPT_NAME,
        prompt_template_version=_SIMPLE_EXTRACTION_PROMPT_VERSION,
    )


def llm_extract_session(
    turns: list[dict[str, Any]],
    speaker_a: str = "A",
    speaker_b: str = "B",
    session_date: str = "",
    model: str = DEFAULT_EXTRACTION_MODEL,
    api_key: str | None = None,
) -> tuple[list[Memory], list[dict[str, Any]]]:
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
    turns: list[dict[str, Any]],
    speaker_a: str = "A",
    speaker_b: str = "B",
    session_date: str = "",
    model: str = DEFAULT_EXTRACTION_MODEL,
    api_key: str | None = None,
) -> tuple[
    list[Memory],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Extract facts + triples + entity profiles + concepts from a session.

    Returns (fact_memories, triples, entity_profiles, concept_profiles).
    - triples: {subject, predicate, object, event_date, source_quote, attributes}
    - entity_profiles: {name, type, profile, tags}
    - concept_profiles: {title, description, entities, tags, event_date}
    """
    client, clean_model = _resolve_client(model, api_key)

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

    from attestor.llm_trace import traced_create
    response = traced_create(
        client,
        role="extraction",
        model=clean_model,
        max_tokens=8192,
        timeout=30,
        max_retries=0,
        messages=[{"role": "user", "content": prompt}],
    )

    if not response.choices or not response.choices[0].message:
        logger.warning("LLM returned empty response; skipping extraction")
        return [], [], [], []
    content = response.choices[0].message.content or ""
    parsed = _parse_extraction_response(content)
    facts = parsed.get("facts", [])
    triples_raw = parsed.get("relations", [])
    entities_raw = parsed.get("entities", [])
    concepts_raw = parsed.get("concepts", [])

    memories = _facts_to_memories(
        facts,
        default_event_date=session_date,
        prompt_template=SESSION_EXTRACTION_PROMPT_NAME,
        prompt_template_version=_SESSION_EXTRACTION_PROMPT_VERSION,
    )

    valid_triples: list[dict[str, Any]] = []
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

    valid_entities: list[dict[str, Any]] = []
    for e in entities_raw:
        if not (isinstance(e, dict) and e.get("name") and e.get("profile")):
            continue
        valid_entities.append({
            "name": str(e["name"]),
            "type": e.get("type", "entity"),
            "profile": str(e["profile"]),
            "tags": list(e.get("tags", [])),
        })

    valid_concepts: list[dict[str, Any]] = []
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


def _parse_json_response(text: str) -> list[dict[str, Any]]:
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


def _parse_extraction_response(text: str) -> dict[str, Any]:
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
    facts: list[dict[str, Any]],
    default_event_date: str | None = None,
    *,
    prompt_template: str | None = None,
    prompt_template_version: str | None = None,
) -> list[Memory]:
    """Convert extracted fact dicts to Memory objects.

    When ``prompt_template`` / ``prompt_template_version`` are supplied,
    they're stamped into ``Memory.metadata`` so audit replay can correlate
    each fact back to the exact externalized prompt template that produced
    it. Backwards-compatible: legacy callers that omit them get the same
    metadata shape they always did.
    """
    memories = []
    for fact in facts:
        if not (isinstance(fact, dict) and "content" in fact):
            continue
        metadata: dict[str, Any] = {}
        if fact.get("source_quote"):
            metadata["source_quote"] = fact["source_quote"]
        kind = fact.get("kind")
        if kind in {"list_item", "atomic"}:
            metadata["kind"] = kind
        if prompt_template:
            metadata["prompt_template"] = prompt_template
        if prompt_template_version:
            metadata["prompt_version"] = prompt_template_version
        memories.append(
            Memory(
                content=fact["content"],
                tags=fact.get("tags", []),
                category=fact.get("category", "general"),
                entity=fact.get("entity"),
                event_date=fact.get("event_date") or default_event_date,
                confidence=fact.get("confidence", 1.0),
                metadata=metadata,
            )
        )
    return memories
