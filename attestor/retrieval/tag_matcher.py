"""Layer 1: Tag extraction and matching."""

from __future__ import annotations

import re
from typing import List

# Common stop words to filter out during tag extraction
_STOP_WORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "as", "into", "about", "like",
    "through", "after", "over", "between", "out", "against", "during",
    "without", "before", "under", "around", "among", "what", "which",
    "who", "whom", "this", "that", "these", "those", "am", "or", "and",
    "but", "if", "than", "too", "very", "just", "how", "where", "when",
    "why", "all", "each", "every", "both", "few", "more", "most", "other",
    "some", "such", "no", "not", "only", "own", "same", "so", "then",
    "up", "down", "my", "your", "his", "her", "its", "our", "their",
    "me", "him", "us", "them", "i", "you", "he", "she", "it", "we", "they",
    "tell", "know", "get", "give", "go", "come", "make", "find", "say",
    "think", "see", "want", "look", "use", "user", "user's", "does",
    "current", "currently", "prefer", "prefers", "preferred",
})


def extract_tags(query: str) -> List[str]:
    """Extract potential tags from a natural language query.

    Uses simple keyword extraction: lowercase, strip punctuation,
    remove stop words, keep words >= 2 chars.
    """
    words = re.findall(r"[a-zA-Z0-9_]+", query.lower())
    tags = [w for w in words if w not in _STOP_WORDS and len(w) >= 2]
    return list(dict.fromkeys(tags))  # dedupe preserving order
