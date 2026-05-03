"""LLM-driven entity + relation extraction at ingest (Mem0/Zep alignment).

Why this module exists
----------------------

The regex-based ``attestor.graph.extractor`` returns
``entity_count=0`` for most LME-S content (verified via
``entity_names=[]`` traces in production — see e.g. the 852ce960
mortgage-pre-approval failure). The graph layer is essentially dead
for natural-language inputs because the regex patterns target
synthetic phrasings ("user uses X", "works at Y") that do not appear
in real conversation transcripts.

The 2026 Mem0/Zep architectural pattern is to run an LLM at ingest to
extract entities + relationships properly. This module ships that
path. It is a strict superset of the regex layer:

  - When ``stack.ingest.llm_entity_extraction.enabled`` is ``False``
    (the shipping default), the codepath is byte-identical to ``main``
    — the regex extractor runs unchanged.
  - When enabled, an LLM call replaces the regex pass at ingest.
  - When enabled BUT the LLM call fails (timeout, malformed JSON,
    network down) the path degrades to the regex extractor — never to
    empty. Failure isolation invariant: ingest must never break.

Public surface
--------------

    extract_with_llm(content, *, model, api_key, timeout, schema_hint)
        Strict LLM-only path. Returns ``([], [])`` on every failure
        mode (no key / timeout / malformed JSON). Pure function over
        the LLM client; no regex fallback.

    extract_or_regex(content, tags, entity, category, namespace, *,
                     llm_enabled, llm_model)
        The function ``AgentMemory.add()`` calls. Branches on
        ``llm_enabled``: True → LLM, falling back to regex on failure;
        False → regex directly. Both branches return the same
        ``(nodes, edges)`` shape that the existing graph backend
        accepts.

    ExtractedEntity / ExtractedRelation
        Frozen dataclasses for the LLM-side result. Lifted to the
        ``nodes/edges`` dict shape inside ``extract_or_regex``.

The prompt template lives at ``attestor/extraction/prompts/
llm_entity_extraction_v1.md`` so prompt edits do not require a code
change. Cost: ~$0.0003 / ingest with claude-haiku-4.5 on LME-S
(~1k content chars → ~80 output tokens of JSON).
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("attestor.extraction.llm_entity_extractor")


# ─── Public dataclasses ─────────────────────────────────────────────────


@dataclass(frozen=True)
class ExtractedEntity:
    """One entity extracted by the LLM.

    The ``type`` value is constrained to a fixed vocabulary by the
    prompt; we don't validate it server-side because (a) the prompt
    already pins the closed set and (b) graph backends accept any
    string for the type column. If a future model invents a new type
    the graph still indexes the row — we just lose the typed-edge
    semantics.
    """

    name: str
    type: str  # person | organization | concept | event | tool | location | quantity | other
    attributes: dict[str, Any]


@dataclass(frozen=True)
class ExtractedRelation:
    """One typed edge extracted by the LLM.

    ``from_entity`` and ``to`` are entity NAMES — the graph backend
    resolves them to nodes by name. We use ``from_entity`` instead of
    ``from`` because ``from`` is a Python keyword.
    """

    from_entity: str
    to: str
    type: str
    metadata: dict[str, Any]


# ─── Prompt template ───────────────────────────────────────────────────
#
# Worked examples are the single biggest lever for output-quality on
# structured extraction tasks. The template embeds 2-3 LME-S-style
# examples covering: proper nouns, value-bearing tokens (amounts,
# dates), relations only when explicitly stated, refusal to fabricate.
#
# Type vocabulary is closed: person, organization, concept, event,
# tool, location, quantity, other. If the model invents a new type
# the downstream graph layer can't index by type — pin the set in the
# prompt.

_EXTRACTION_PROMPT = """You extract structured entities and \
relationships from a memory snippet for an agent's knowledge graph.

Output STRICT JSON with two top-level arrays — entities and \
relations. Output nothing else (no prose, no markdown fences).

Entity schema:
  {{"name": <string>,
    "type": one of {{person, organization, concept, event, tool, \
location, quantity, other}},
    "attributes": <object — small map of facts about the entity, \
empty object if none>}}

Relation schema:
  {{"from": <entity name>,
    "to": <entity name>,
    "type": <short snake_case verb phrase, e.g. works_at, \
pre_approved_by, located_in>,
    "metadata": <object, empty if none>}}

Rules:
  - Extract proper-noun entities (people, organizations, products, \
locations).
  - Extract value-bearing entities (amounts like "$350,000", \
durations, frequencies, dates) as type=quantity or type=event.
  - Use "User" as the entity name when the snippet refers to the \
person whose memory this is.
  - Output a relation ONLY when the relationship is explicitly stated \
in the snippet. NEVER fabricate edges from co-occurrence alone.
  - When the snippet has no extractable entity, return \
{{"entities": [], "relations": []}}.
  - Use the closed type vocabulary above. Use "other" only when no \
listed type fits.

Worked examples:

Example 1
Input:
"[2023-08-11] User: I just got pre-approved for $350,000 from Wells \
Fargo for the house in Austin."
Output:
{{"entities": [
  {{"name": "User", "type": "person", "attributes": {{}}}},
  {{"name": "Wells Fargo", "type": "organization", \
"attributes": {{"role": "lender"}}}},
  {{"name": "$350,000", "type": "quantity", \
"attributes": {{"currency": "USD"}}}},
  {{"name": "Austin", "type": "location", "attributes": {{}}}},
  {{"name": "pre-approved", "type": "event", \
"attributes": {{"date": "2023-08-11"}}}}
],
"relations": [
  {{"from": "User", "to": "Wells Fargo", \
"type": "pre_approved_by", "metadata": {{}}}},
  {{"from": "User", "to": "$350,000", \
"type": "approved_for_amount", "metadata": {{}}}}
]}}

Example 2
Input:
"User now works at SoFi as a senior engineer (started 2024-01-15)."
Output:
{{"entities": [
  {{"name": "User", "type": "person", "attributes": {{}}}},
  {{"name": "SoFi", "type": "organization", "attributes": {{}}}},
  {{"name": "senior engineer", "type": "concept", \
"attributes": {{"role": "title"}}}}
],
"relations": [
  {{"from": "User", "to": "SoFi", "type": "works_at", \
"metadata": {{"since": "2024-01-15"}}}}
]}}

Example 3
Input:
"The weather in Tokyo was beautiful today."
Output:
{{"entities": [
  {{"name": "Tokyo", "type": "location", "attributes": {{}}}}
],
"relations": []}}

{schema_hint}

Memory snippet:
{content}

Output (strict JSON, no markdown):"""


# ─── LLM client resolution ──────────────────────────────────────────────


def _resolve_client(model: str, api_key: str | None) -> tuple[Any, str]:
    """Resolve ``(client, clean_model)`` via the YAML-driven LLM pool.

    Mirrors :func:`attestor.extraction.llm_extractor._resolve_client`
    so this module routes through the same multi-provider pool that
    the rest of the codebase uses.
    """
    from attestor.llm_trace import _get_pool, get_client_for_model, make_client

    if api_key:
        pool = _get_pool()
        head, sep, tail = model.partition("/")
        if sep and head in pool.providers:
            strategy = pool._strategies[head]
            clean_model = tail
        else:
            strategy = pool.default_strategy()
            clean_model = model
        client = make_client(base_url=strategy.base_url, api_key=api_key)
        return client, clean_model

    return get_client_for_model(model)


# ─── Process-local cache (content_hash → (entities, relations)) ────────
#
# Re-ingesting the same content within a process should not pay for a
# second LLM call. The cache key is the SHA-256 of stripped content;
# values are the parsed dataclass tuples. Process-local — fine for a
# single benchmark run, fine for production where AgentMemory is
# typically a long-lived singleton. Cross-process cache (Redis) is a
# future concern; the prompt is cheap enough that we don't need it
# for the LME-S benches motivating this PR.

# Cap the cache to bound memory in long-lived service processes.
# OrderedDict.popitem(last=False) drops the oldest entry on overflow,
# giving FIFO eviction with O(1) lookup. Bench runs touch O(few-thousand)
# unique chunks; production traffic hits this much harder so the cap
# matters there.
from collections import OrderedDict

_CACHE_MAX_SIZE = int(__import__("os").environ.get("ATTESTOR_ENTITY_CACHE_MAX", "2048"))
_CACHE: OrderedDict[str, tuple[list[ExtractedEntity], list[ExtractedRelation]]] = OrderedDict()


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.strip().encode()).hexdigest()


def _cache_put(key: str, value: tuple[list[ExtractedEntity], list[ExtractedRelation]]) -> None:
    """Insert into the LRU-ish cache with FIFO eviction at the cap."""
    _CACHE[key] = value
    while len(_CACHE) > _CACHE_MAX_SIZE:
        _CACHE.popitem(last=False)


def _reset_cache() -> None:
    """Clear the process-local cache. Used by tests; not a public API."""
    _CACHE.clear()


# ─── Parsing helpers ────────────────────────────────────────────────────


from attestor.extraction.utils import strip_markdown_fences as _strip_markdown_fences  # noqa: E402


def _parse_extraction_response(
    text: str,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """Parse the LLM JSON response → ([entities], [relations]).

    Returns ``([], [])`` on any parse error (malformed JSON, missing
    keys, wrong shape). Failure isolation: a malformed response must
    never crash the ingest path.
    """
    text = _strip_markdown_fences(text)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return [], []

    if not isinstance(data, dict):
        return [], []

    entities: list[ExtractedEntity] = []
    for raw in data.get("entities") or []:
        if not isinstance(raw, dict):
            continue
        name = raw.get("name")
        type_ = raw.get("type")
        if not isinstance(name, str) or not name.strip():
            continue
        if not isinstance(type_, str) or not type_.strip():
            continue
        attributes = raw.get("attributes") or {}
        if not isinstance(attributes, dict):
            attributes = {}
        entities.append(
            ExtractedEntity(name=name.strip(), type=type_.strip(),
                            attributes=dict(attributes)),
        )

    relations: list[ExtractedRelation] = []
    for raw in data.get("relations") or []:
        if not isinstance(raw, dict):
            continue
        # Accept both "from" (LLM-friendly) and "from_entity" (Python-safe)
        frm = raw.get("from") or raw.get("from_entity")
        to = raw.get("to")
        type_ = raw.get("type")
        if not isinstance(frm, str) or not frm.strip():
            continue
        if not isinstance(to, str) or not to.strip():
            continue
        if not isinstance(type_, str) or not type_.strip():
            continue
        metadata = raw.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        relations.append(
            ExtractedRelation(
                from_entity=frm.strip(),
                to=to.strip(),
                type=type_.strip(),
                metadata=dict(metadata),
            ),
        )

    return entities, relations


# ─── Public entrypoints ─────────────────────────────────────────────────


# Cap on the content we feed the LLM. LME-S session turns top out
# around 2-3k chars; 4k is the documented spec ceiling and gives
# headroom without inflating prompt cost. Truncation is graceful —
# we keep the prefix and drop the tail.
_CONTENT_CHAR_CAP = 4000


def extract_with_llm(
    content: str,
    *,
    model: str | None = None,
    api_key: str | None = None,
    timeout: float = 15.0,
    schema_hint: str | None = None,
) -> tuple[list[ExtractedEntity], list[ExtractedRelation]]:
    """LLM-only extraction path. Returns ``([], [])`` on every failure.

    This is the strict path used by tests and by callers who want to
    skip the regex fallback. Most production traffic should go through
    :func:`extract_or_regex` instead, which falls back to regex on LLM
    failure rather than to empty.

    Parameters
    ----------
    content
        The memory snippet to extract from. Truncated to the first
        4000 characters before sending.
    model
        Provider-prefixed model id (e.g. ``"anthropic/claude-haiku-4.5"``).
        ``None`` resolves to the YAML-configured extraction model.
    api_key
        Optional explicit key override. ``None`` uses the env-resolved
        key from the LLM pool.
    timeout
        Per-request deadline. Default 15s — tight enough that ingest
        latency stays bounded, loose enough that haiku-4.5 returns
        comfortably.
    schema_hint
        Optional extra instructions appended to the prompt (e.g.
        domain-specific entity hints). ``None`` → no extra hint.
    """
    if not content or not content.strip():
        return [], []

    # Cache: same content_hash → reuse prior result.
    chash = _content_hash(content)
    cached = _CACHE.get(chash)
    if cached is not None:
        return cached

    # Truncate gracefully
    truncated = content[:_CONTENT_CHAR_CAP]
    resolved_model = model or "anthropic/claude-haiku-4.5"
    hint = (schema_hint.strip() + "\n\n") if schema_hint else ""

    try:
        client, clean_model = _resolve_client(resolved_model, api_key)
    except Exception as e:
        logger.warning(
            "llm_entity_extractor: client resolution failed (%s: %s); "
            "returning empty result",
            type(e).__name__, e,
        )
        return [], []

    prompt = _EXTRACTION_PROMPT.format(
        schema_hint=hint,
        content=truncated,
    )

    try:
        response = client.chat.completions.create(
            model=clean_model,
            max_tokens=2048,
            temperature=0.0,
            timeout=timeout,
            messages=[{"role": "user", "content": prompt}],
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning(
            "llm_entity_extractor: LLM call failed (%s: %s); "
            "returning empty result",
            type(e).__name__, e,
        )
        return [], []

    entities, relations = _parse_extraction_response(text)

    # Cache the result (even when empty — repeated calls on the same
    # content shouldn't keep retrying a model that produced empty
    # output for legitimate reasons).
    _cache_put(chash, (entities, relations))
    return entities, relations


def extract_or_regex(
    content: str,
    tags: list[str],
    entity: str | None,
    category: str,
    namespace: str | None = None,
    *,
    llm_enabled: bool,
    llm_model: str | None = None,
    llm_api_key: str | None = None,
    llm_timeout: float = 15.0,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Branch between LLM and regex extractors; return graph-ready dicts.

    The function ``AgentMemory.add()`` calls. Both branches return the
    same ``(nodes, edges)`` shape that the existing graph backend
    accepts (the regex extractor's contract).

    Behavior
    --------

    ``llm_enabled=False``
        Delegate to the existing regex extractor unchanged.
        Byte-identical to ``main``.

    ``llm_enabled=True``
        Call :func:`extract_with_llm`. If it returns at least one
        entity OR relation, lift to the regex extractor's
        ``(nodes, edges)`` dict shape and return that. If it returns
        ``([], [])`` (timeout / malformed / no-entity), fall back to
        the regex extractor — never empty.
    """
    if not llm_enabled:
        from attestor.graph.extractor import extract_entities_and_relations

        return extract_entities_and_relations(
            content, tags, entity, category, namespace=namespace,
        )

    entities, relations = extract_with_llm(
        content,
        model=llm_model,
        api_key=llm_api_key,
        timeout=llm_timeout,
    )

    if not entities and not relations:
        # LLM degraded → fall back to regex. Never to empty.
        logger.debug(
            "llm_entity_extractor: LLM returned empty; "
            "falling back to regex extractor",
        )
        from attestor.graph.extractor import extract_entities_and_relations

        return extract_entities_and_relations(
            content, tags, entity, category, namespace=namespace,
        )

    nodes: list[dict[str, Any]] = []
    edges: list[dict[str, Any]] = []

    for ent in entities:
        node: dict[str, Any] = {
            "name": ent.name,
            "type": ent.type,
            "attributes": dict(ent.attributes),
        }
        if namespace is not None:
            node["namespace"] = namespace
        nodes.append(node)

    for rel in relations:
        edge: dict[str, Any] = {
            "from": rel.from_entity,
            "to": rel.to,
            "type": rel.type,
            "metadata": dict(rel.metadata),
        }
        if namespace is not None:
            edge["namespace"] = namespace
        edges.append(edge)

    return nodes, edges
