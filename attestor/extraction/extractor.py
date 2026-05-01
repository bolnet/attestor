"""Memory extraction from conversations."""

from __future__ import annotations

from typing import Any

from attestor.extraction.rule_based import extract_from_text
from attestor.models import Memory


def extract_memories(
    messages: list[dict[str, Any]],
    use_llm: bool = False,
    model: str | None = None,
) -> list[Memory]:
    """Extract memories from conversation messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        use_llm: If True, use LLM-based extraction (requires the ``openai``
            package + the API key for whichever provider is configured
            under ``stack.llm.providers`` in ``configs/attestor.yaml``).
        model: LLM model to use if use_llm=True.

    Returns:
        List of Memory objects extracted from the conversation.
    """
    if use_llm:
        if model is None:
            from attestor.config import get_stack
            model = get_stack().models.extraction
        return _llm_extract(messages, model)
    return _rule_extract(messages)


def _rule_extract(messages: list[dict[str, Any]]) -> list[Memory]:
    """Extract memories using rule-based patterns."""
    memories = []
    for msg in messages:
        content = msg.get("content", "")
        if not content or msg.get("role") == "system":
            continue
        extracted = extract_from_text(content)
        for item in extracted:
            memories.append(
                Memory(
                    content=item["content"],
                    tags=item["tags"],
                    category=item["category"],
                    entity=item.get("entity"),
                )
            )
    return memories


def _llm_extract(messages: list[dict[str, Any]], model: str) -> list[Memory]:
    """Extract memories using LLM. Requires the ``openai`` package and
    the API key for whichever provider is configured under
    ``stack.llm.providers`` in ``configs/attestor.yaml``."""
    try:
        from attestor.extraction.llm_extractor import llm_extract
        return llm_extract(messages, model)
    except ImportError:
        # Fall back to rule-based if openai not installed
        return _rule_extract(messages)


def extract_from_session(
    turns: list[dict[str, Any]],
    speaker_a: str = "A",
    speaker_b: str = "B",
    session_date: str = "",
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[list[Memory], list[dict[str, Any]]]:
    """Extract memories and relation triples from a conversation session.

    Uses LLM extraction for conversational data (chatbots, companions).
    Falls back to rule-based if openai not available.

    Returns (memories, triples) where triples are dicts with
    subject, predicate, object, event_date keys.
    """
    if model is None:
        from attestor.config import get_stack
        model = get_stack().models.extraction
    try:
        from attestor.extraction.llm_extractor import llm_extract_session

        return llm_extract_session(
            turns=turns,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            session_date=session_date,
            model=model,
            api_key=api_key,
        )
    except ImportError:
        messages = _turns_to_messages(turns, speaker_a, speaker_b)
        return _rule_extract(messages), []


def extract_from_session_full(
    turns: list[dict[str, Any]],
    speaker_a: str = "A",
    speaker_b: str = "B",
    session_date: str = "",
    model: str | None = None,
    api_key: str | None = None,
) -> tuple[
    list[Memory],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Extract facts + triples + entity profiles + concept profiles.

    Returns (fact_memories, triples, entity_profiles, concept_profiles).
    Falls back to (rule_facts, [], [], []) if openai is unavailable.
    """
    if model is None:
        from attestor.config import get_stack
        model = get_stack().models.extraction
    try:
        from attestor.extraction.llm_extractor import llm_extract_session_full

        return llm_extract_session_full(
            turns=turns,
            speaker_a=speaker_a,
            speaker_b=speaker_b,
            session_date=session_date,
            model=model,
            api_key=api_key,
        )
    except ImportError:
        messages = _turns_to_messages(turns, speaker_a, speaker_b)
        return _rule_extract(messages), [], [], []


def _turns_to_messages(
    turns: list[dict[str, Any]],
    speaker_a: str,
    speaker_b: str,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        speaker = turn.get("speaker", "Unknown")
        if speaker == "A":
            speaker = speaker_a
        elif speaker == "B":
            speaker = speaker_b
        messages.append({
            "role": speaker,
            "content": turn.get("text", ""),
        })
    return messages
