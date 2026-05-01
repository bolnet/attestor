"""Extract entities and relations from memory content for the graph layer."""

from __future__ import annotations

import re
from typing import Any

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
    tags: list[str],
    entity: str | None,
    category: str,
    namespace: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Extract entities and relations from a memory's content and metadata.

    Returns (nodes, edges) where:
      - nodes: [{"name": str, "type": str, "namespace": str, "attributes": dict}, ...]
      - edges: [{"from": str, "to": str, "type": str, "namespace": str, "metadata": dict}, ...]

    Every produced node/edge carries the writer's ``namespace`` so the
    graph backend can enforce tenancy without additional plumbing. When
    ``namespace`` is ``None`` the field is omitted from each dict (the
    backend treats absent values as ``"default"`` via coalesce on read).
    """
    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    def _node(name: str, type_: str, attributes: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {"name": name, "type": type_, "attributes": attributes}
        if namespace is not None:
            out["namespace"] = namespace
        return out

    def _edge(
        from_: str, to: str, type_: str, metadata: dict[str, Any],
    ) -> dict[str, Any]:
        out: dict[str, Any] = {
            "from": from_, "to": to, "type": type_, "metadata": metadata,
        }
        if namespace is not None:
            out["namespace"] = namespace
        return out

    # Primary entity from the memory
    if entity:
        entity_type = _CATEGORY_TYPE_MAP.get(category, "concept")
        nodes.append(_node(entity, entity_type, {"category": category}))

    # Extract entities from tags (proper nouns / meaningful keywords)
    for tag in tags:
        if tag and len(tag) > 1:
            # Tags that look like proper nouns or tool names
            if tag[0].isupper() or tag in _KNOWN_TOOLS:
                nodes.append(_node(tag, _guess_type(tag), {}))
                # Create relation from primary entity to tag entity
                if entity and tag.lower() != entity.lower():
                    edges.append(_edge(entity, tag, "related_to", {"source": "tag"}))

    # Extract relations from content via patterns
    content_lower = content.lower()
    for pattern, rel_type in _RELATION_PATTERNS:
        match = re.search(pattern, content_lower)
        if match:
            target = match.group(1).strip().rstrip(".")
            # Clean up the target
            target = target.split(",")[0].strip()  # Take first item if list
            if len(target) > 1 and len(target) < 50:
                nodes.append(_node(target, _guess_type(target), {}))
                source = entity or "user"
                edges.append(
                    _edge(source, target, rel_type, {"source": "content_pattern"})
                )

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
