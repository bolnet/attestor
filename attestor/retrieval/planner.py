"""LLM query planner: classify a question into an intent + namespace plan.

The planner reads the user's question and produces a structured plan telling
the retriever which vector namespace(s) to query, which entities are in focus,
and any additional filters.  It is deliberately small: a single LLM call with
a JSON-only response.

Intents (LOCOMO-shaped but useful beyond):
    - FACTUAL_RECALL  : point facts ("where does Caroline live?")
    - ACTIVITY_LIST   : enumerate activities/events ("what books did she read?")
    - CONCEPT_LOOKUP  : thematic summary ("tell me about her LGBTQ work")
    - RELATIVE_DATE   : date arithmetic ("a week before X")
    - ENTITY_LIST     : list related entities ("who are her friends?")

Namespaces: document_chunks, entities, concepts.

The planner never hits the store; it is pure I/O on the question.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PLANNER_MODEL = os.environ.get(
    "PLANNER_MODEL", "anthropic/claude-opus-4.7"
)

VALID_INTENTS = frozenset({
    "FACTUAL_RECALL",
    "ACTIVITY_LIST",
    "CONCEPT_LOOKUP",
    "RELATIVE_DATE",
    "ENTITY_LIST",
})
VALID_NAMESPACES = frozenset({"document_chunks", "entities", "concepts"})

# Fallback plan: whole index, all namespaces, no focus.
_DEFAULT_NAMESPACES: List[str] = ["entities", "concepts", "document_chunks"]


@dataclass(frozen=True)
class QueryPlan:
    """Immutable retrieval plan for a single question."""

    intent: str
    entities: List[str] = field(default_factory=list)
    namespaces: List[str] = field(default_factory=lambda: list(_DEFAULT_NAMESPACES))
    filters: Dict[str, Any] = field(default_factory=dict)

    @property
    def primary_namespace(self) -> str:
        return self.namespaces[0] if self.namespaces else "entities"


_PLANNER_PROMPT = """You are the retrieval planner for a personal-memory agent.
Classify the user's question into ONE intent and pick the namespace(s) to query.

Intents:
- FACTUAL_RECALL  : point facts about a person/entity (job, address, name, status, relationship).
- ACTIVITY_LIST   : list of activities, books read, events attended, places visited, projects done.
- CONCEPT_LOOKUP  : thematic summary or explanation of an ongoing theme/hobby/interest.
- RELATIVE_DATE   : question involves date arithmetic ("the day before", "last week", "next month", "X days after").
- ENTITY_LIST     : list of related people/entities (friends, family, colleagues, pets).

Namespaces (pick 1-3, in priority order):
- entities        : synthesized profiles + atomic facts per person/entity. Best for FACTUAL_RECALL, ENTITY_LIST, specifics about a person.
- concepts        : synthesized themes + activity clusters. Best for ACTIVITY_LIST, CONCEPT_LOOKUP, "all of X's hobbies".
- document_chunks : raw verbatim conversation turns. Best for quotes, exact dates, RELATIVE_DATE reasoning, anything where the phrasing matters.

Namespace guidance:
- FACTUAL_RECALL   → ["entities", "document_chunks"]
- ACTIVITY_LIST    → ["concepts", "document_chunks", "entities"]
- CONCEPT_LOOKUP   → ["concepts", "entities"]
- RELATIVE_DATE    → ["document_chunks", "entities"]
- ENTITY_LIST      → ["entities", "concepts"]

Entities: list ALL distinct named people, places, or organizations that appear in the question (canonical forms, e.g. "Caroline" not "she").

Filters: optional. If the question contains an explicit year or ISO date, include {{"year": 2023}} or {{"date": "2023-05-08"}}.

Return ONLY a JSON object with keys: intent, entities, namespaces, filters.

Question: {question}

JSON:"""


def _get_client(api_key: Optional[str] = None):
    from openai import OpenAI

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY not set")
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def _sanitize_plan(raw: Dict[str, Any]) -> QueryPlan:
    intent = str(raw.get("intent", "FACTUAL_RECALL")).upper().strip()
    if intent not in VALID_INTENTS:
        intent = "FACTUAL_RECALL"

    entities_raw = raw.get("entities", []) or []
    entities = [str(e).strip() for e in entities_raw if str(e).strip()]

    ns_raw = raw.get("namespaces", []) or []
    namespaces: List[str] = []
    for ns in ns_raw:
        ns_str = str(ns).strip()
        if ns_str in VALID_NAMESPACES and ns_str not in namespaces:
            namespaces.append(ns_str)
    if not namespaces:
        namespaces = list(_DEFAULT_NAMESPACES)

    filters_raw = raw.get("filters", {}) or {}
    filters = dict(filters_raw) if isinstance(filters_raw, dict) else {}

    return QueryPlan(
        intent=intent,
        entities=entities,
        namespaces=namespaces,
        filters=filters,
    )


def plan_query(
    question: str,
    *,
    model: str = DEFAULT_PLANNER_MODEL,
    api_key: Optional[str] = None,
) -> QueryPlan:
    """Ask the planner LLM to classify `question` into a QueryPlan.

    On any error (missing key, malformed JSON, bad intent), returns a broad
    fallback plan that queries all three namespaces so the retriever never
    goes blind.
    """
    try:
        client = _get_client(api_key)
    except ValueError:
        return QueryPlan(intent="FACTUAL_RECALL")

    prompt = _PLANNER_PROMPT.format(question=question)
    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception:
        return QueryPlan(intent="FACTUAL_RECALL")

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(l for l in lines if not l.strip().startswith("```"))
        text = text.strip()

    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        brace_start = text.find("{")
        brace_end = text.rfind("}")
        if brace_start >= 0 and brace_end > brace_start:
            try:
                parsed = json.loads(text[brace_start : brace_end + 1])
            except (json.JSONDecodeError, ValueError):
                return QueryPlan(intent="FACTUAL_RECALL")
        else:
            return QueryPlan(intent="FACTUAL_RECALL")

    if not isinstance(parsed, dict):
        return QueryPlan(intent="FACTUAL_RECALL")

    return _sanitize_plan(parsed)
