"""Default rule-based extraction (no LLM needed)."""

from __future__ import annotations

import re
from typing import Any, Dict, List

from attestor.models import Memory

# Patterns for extracting structured info from text
_PREFERENCE_PATTERNS = [
    r"(?:prefer|like|love|enjoy|favor)s?\s+(.+?)(?:\s+over\s+(.+?))?(?:\.|$)",
    r"(?:hate|dislike|avoid)s?\s+(.+?)(?:\.|$)",
]

_FACT_PATTERNS = [
    r"(?:work|works|working)\s+(?:at|for)\s+(.+?)(?:\s+as\s+(.+?))?(?:\.|$)",
    r"(?:live|lives|living)\s+(?:in|at)\s+(.+?)(?:\.|$)",
    r"(?:use|uses|using)\s+(.+?)(?:\s+for\s+(.+?))?(?:\.|$)",
]


def extract_from_text(text: str) -> List[Dict[str, Any]]:
    """Extract potential memory facts from plain text using regex patterns.

    Returns list of dicts with keys: content, tags, category, entity.
    """
    results = []

    for pattern in _PREFERENCE_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            content = match.group(0).strip()
            results.append({
                "content": content,
                "tags": ["preference"],
                "category": "preference",
                "entity": None,
            })

    for pattern in _FACT_PATTERNS:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            content = match.group(0).strip()
            entity = match.group(1).strip() if match.group(1) else None
            results.append({
                "content": content,
                "tags": _infer_tags(content),
                "category": _infer_category(content),
                "entity": entity,
            })

    return results


def _infer_tags(content: str) -> List[str]:
    """Infer tags from content keywords."""
    tags = []
    content_lower = content.lower()
    tag_keywords = {
        "career": ["work", "job", "role", "position", "company", "hired"],
        "location": ["live", "city", "country", "move", "relocate"],
        "tech": ["python", "javascript", "react", "database", "api", "code"],
        "preference": ["prefer", "like", "love", "favorite", "enjoy"],
    }
    for tag, keywords in tag_keywords.items():
        if any(kw in content_lower for kw in keywords):
            tags.append(tag)
    return tags or ["general"]


def _infer_category(content: str) -> str:
    """Infer category from content."""
    content_lower = content.lower()
    if any(w in content_lower for w in ["work", "job", "role", "company"]):
        return "career"
    if any(w in content_lower for w in ["live", "city", "move"]):
        return "location"
    if any(w in content_lower for w in ["prefer", "like", "love", "hate"]):
        return "preference"
    return "general"
