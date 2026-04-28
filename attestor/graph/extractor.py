"""Extract entities and relations from memory content for the graph layer."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# Date patterns. Multi-hop temporal queries fail when the event description
# and the date live in different memories — vector search finds one but not
# the other (their token / semantic overlap is near-zero). Promoting dates
# to first-class entities lets graph expansion chain them: query about
# `sarah → camping_event` traverses the shared `july_4` date entity to the
# separate calendar memory that gives the actual date.
#
# Reference benchmark failure mode: logs/phase2_temporal_fails.json
# (six "When did X go to Y?" queries that all missed the same way).
_MONTHS = (
    "january|february|march|april|may|june|july|august|september|october|november|december"
    "|jan|feb|mar|apr|jun|jul|aug|sep|sept|oct|nov|dec"
)
# "July 4", "July 4th", "July 4, 2026", "March 21st"
_DATE_DAY_PATTERN = re.compile(
    rf"\b({_MONTHS})\s+(\d{{1,2}})(?:st|nd|rd|th)?(?:,?\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)
# Bare month references like "in July" / "during March 2026"
_DATE_MONTH_PATTERN = re.compile(
    rf"\b({_MONTHS})(?:\s+(\d{{4}}))?\b",
    re.IGNORECASE,
)


def _extract_dates(content: str) -> List[str]:
    """Return canonicalized date entity names found in content.

    Examples:
        "July 4th weekend"          -> ["July 4"]
        "Sunday June 28"            -> ["June 28"]
        "March 2026"                -> ["March 2026"]
        "October 11" + "October"    -> ["October 11", "October"]

    Day-specific dates win over bare months in the same string — we only
    add the bare month entity if no day-pattern already covers it.
    """
    out: List[str] = []
    seen_months: set[str] = set()
    for m in _DATE_DAY_PATTERN.finditer(content):
        month, day, year = m.group(1).capitalize(), m.group(2), m.group(3)
        # Normalize month abbreviations
        month = _MONTH_NORMALIZE.get(month.lower(), month)
        canonical = f"{month} {int(day)}"
        if year:
            canonical += f", {year}"
        out.append(canonical)
        seen_months.add(month.lower())
    for m in _DATE_MONTH_PATTERN.finditer(content):
        month, year = m.group(1).capitalize(), m.group(2)
        month = _MONTH_NORMALIZE.get(month.lower(), month)
        if month.lower() in seen_months:
            continue  # already captured with a day
        canonical = f"{month} {year}" if year else month
        if canonical not in out:
            out.append(canonical)
    return out


_MONTH_NORMALIZE = {
    "jan": "January", "feb": "February", "mar": "March", "apr": "April",
    "jun": "June", "jul": "July", "aug": "August", "sep": "September",
    "sept": "September", "oct": "October", "nov": "November", "dec": "December",
    "january": "January", "february": "February", "march": "March", "april": "April",
    "may": "May", "june": "June", "july": "July", "august": "August",
    "september": "September", "october": "October", "november": "November", "december": "December",
}


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

    # Extract date entities (multi-hop temporal fix). Every date mention
    # becomes a first-class entity, edged from the primary entity (or
    # "user" if none) with type "occurred_on". Graph expansion (step 2
    # of the retrieval pipeline) can now chain across memories that
    # share a date — the event memory and the date memory both link to
    # `July 4`, so a query about Sarah's camping pulls in the calendar
    # memory that says when July 4 weekend was.
    for date_name in _extract_dates(content):
        nodes.append({
            "name": date_name,
            "type": "date",
            "attributes": {},
        })
        source = entity or "user"
        if source.lower() != date_name.lower():
            edges.append({
                "from": source,
                "to": date_name,
                "type": "occurred_on",
                "metadata": {"source": "date_extractor"},
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
