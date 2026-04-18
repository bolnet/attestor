"""Memory extraction from conversations."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from attestor.extraction.rule_based import extract_from_text
from attestor.models import Memory


def extract_memories(
    messages: List[Dict[str, Any]],
    use_llm: bool = False,
    model: str = "openai/gpt-4.1-mini",
) -> List[Memory]:
    """Extract memories from conversation messages.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
        use_llm: If True, use LLM-based extraction (requires openai + OPENROUTER_API_KEY).
        model: LLM model to use if use_llm=True.

    Returns:
        List of Memory objects extracted from the conversation.
    """
    if use_llm:
        return _llm_extract(messages, model)
    return _rule_extract(messages)


def _rule_extract(messages: List[Dict[str, Any]]) -> List[Memory]:
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


def _llm_extract(messages: List[Dict[str, Any]], model: str) -> List[Memory]:
    """Extract memories using LLM. Requires openai package + OPENROUTER_API_KEY."""
    try:
        from attestor.extraction.llm_extractor import llm_extract
        return llm_extract(messages, model)
    except ImportError:
        # Fall back to rule-based if openai not installed
        return _rule_extract(messages)


def extract_from_session(
    turns: List[Dict[str, Any]],
    speaker_a: str = "A",
    speaker_b: str = "B",
    session_date: str = "",
    model: str = "openai/gpt-4.1-mini",
    api_key: Optional[str] = None,
) -> Tuple[List[Memory], List[Dict[str, Any]]]:
    """Extract memories and relation triples from a conversation session.

    Uses LLM extraction for conversational data (chatbots, companions).
    Falls back to rule-based if openai not available.

    Returns (memories, triples) where triples are dicts with
    subject, predicate, object, event_date keys.
    """
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
        # Fallback: convert turns to messages format and use rule-based
        messages = []
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
        return _rule_extract(messages), []
