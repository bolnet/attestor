"""Extract entities and relations from memory content for the graph layer."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Patterns for extracting relationships from memory content
_RELATION_PATTERNS = [
    # "X uses Y", "X prefers Y"
    (r"(?:user|team|project)?\s*(?:uses?|using)\s+(.+)", "uses"),
    (r"(?:user|team)?\s*(?:prefers?|preferring)\s+(.+)", "prefers"),
    # "works at X", "working at X"
    (r"(?:works?|working)\s+(?:at|for)\s+(.+)", "works_at"),
    # "lives in X", "based in X"
    (r"(?:lives?|living|based)\s+in\s+(.+)", "located_in"),
    # "X is built with Y", "X depends on Y"
    (r"(?:built|made|written)\s+(?:with|in|using)\s+(.+)", "built_with"),
    (r"(?:depends?\s+on|requires?)\s+(.+)", "depends_on"),
]

# Category to entity type mapping
_CATEGORY_TYPE_MAP = {
    "career": "organization",
    "project": "project",
    "preference": "concept",
    "personal": "person",
    "technical": "tool",
    "location": "location",
    "general": "concept",
}


def extract_entities_and_relations(
    content: str,
    tags: List[str],
    entity: Optional[str],
    category: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Extract entities and relations from a memory's content and metadata.

    Returns (nodes, edges) where:
      - nodes: [{"name": str, "type": str, "attributes": dict}, ...]
      - edges: [{"from": str, "to": str, "type": str, "metadata": dict}, ...]
    """
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []

    # Primary entity from the memory
    if entity:
        entity_type = _CATEGORY_TYPE_MAP.get(category, "concept")
        nodes.append({
            "name": entity,
            "type": entity_type,
            "attributes": {"category": category},
        })

    # Extract entities from tags (proper nouns / meaningful keywords)
    for tag in tags:
        if tag and len(tag) > 1:
            # Tags that look like proper nouns or tool names
            if tag[0].isupper() or tag in _KNOWN_TOOLS:
                nodes.append({
                    "name": tag,
                    "type": _guess_type(tag),
                    "attributes": {},
                })
                # Create relation from primary entity to tag entity
                if entity and tag.lower() != entity.lower():
                    edges.append({
                        "from": entity,
                        "to": tag,
                        "type": "related_to",
                        "metadata": {"source": "tag"},
                    })

    # Extract relations from content via patterns
    content_lower = content.lower()
    for pattern, rel_type in _RELATION_PATTERNS:
        match = re.search(pattern, content_lower)
        if match:
            target = match.group(1).strip().rstrip(".")
            # Clean up the target
            target = target.split(",")[0].strip()  # Take first item if list
            if len(target) > 1 and len(target) < 50:
                nodes.append({
                    "name": target,
                    "type": _guess_type(target),
                    "attributes": {},
                })
                source = entity or "user"
                edges.append({
                    "from": source,
                    "to": target,
                    "type": rel_type,
                    "metadata": {"source": "content_pattern"},
                })

    return nodes, edges


# Common developer tools for type detection
_KNOWN_TOOLS = frozenset({
    "python", "javascript", "typescript", "rust", "go", "java", "ruby",
    "react", "vue", "angular", "svelte", "nextjs", "nuxt",
    "django", "flask", "fastapi", "express", "rails",
    "postgresql", "mysql", "sqlite", "mongodb", "redis",
    "docker", "kubernetes", "terraform", "aws", "gcp", "azure",
    "git", "github", "gitlab", "npm", "yarn", "bun", "pip", "cargo",
    "vscode", "vim", "neovim", "emacs", "jetbrains",
    "linux", "macos", "windows",
    "pytest", "jest", "mocha", "cypress",
})


def _guess_type(name: str) -> str:
    """Guess entity type from name."""
    name_lower = name.lower()
    if name_lower in _KNOWN_TOOLS:
        return "tool"
    if name_lower.endswith(("js", "py", "rs", "go", "rb")):
        return "tool"
    if name[0].isupper() and " " not in name:
        return "concept"  # Could be a proper noun
    return "concept"
