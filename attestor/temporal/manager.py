"""Temporal logic: timeline queries, supersession, contradiction detection."""

from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone

from attestor.models import Memory
from attestor.store.base import DocumentStore


# Strip values that carry a unit hint (currency symbol, percent, common SI /
# imperial units) so two memories that differ only in such a value collapse
# to the same skeleton. Bare integers ("fact 1" vs "fact 2") are NOT
# stripped — they're often distinct enumerations rather than updates to the
# same fact, and false-firing on them surfaced in test_core regressions.
_NUM_PATTERN = re.compile(
    r"(?:[$€£¥])\s*[\d,]+(?:\.\d+)?\s*(?:[kmbt])?\b"           # currency
    r"|"
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:%|percent|kg|lbs?|mi(?:les)?|km|m|cm|mm|"
    r"hours?|hrs?|days?|years?|yrs?|months?|weeks?|wks?|"
    r"times?|x/(?:day|week|month)|/(?:day|week|month))\b",     # units
    flags=re.IGNORECASE,
)
_DATE_TAG_PATTERN = re.compile(r"\[\d{4}-\d{2}-\d{2}[^\]]*\]")
_WS_PATTERN = re.compile(r"\s+")
# Min content length before the entity-None fallback is allowed to fire.
# Trivially short memories (e.g. "fact 1") are too noisy to safely group.
_MIN_FALLBACK_LEN = 12

# Stopwords pruned from auto-topic extraction. Kept tiny on purpose — false
# negatives (extracted topic too generic) are recoverable, false positives
# (over-eager auto-supersede) are not.
_STOP = frozenset({
    "i", "im", "me", "my", "mine", "myself", "you", "your", "yours", "we",
    "our", "ours", "us", "the", "a", "an", "this", "that", "these", "those",
    "is", "am", "are", "was", "were", "be", "been", "being", "have", "has",
    "had", "do", "does", "did", "for", "from", "to", "of", "on", "in", "at",
    "by", "with", "and", "or", "but", "if", "then", "than", "as", "so",
    "user", "assistant", "now", "just", "really", "very", "quite",
    "today", "yesterday", "tomorrow",
})

# Pattern for unit-bearing values that mark "value-context" memories
# eligible for auto-topic extraction. Time-of-day / duration formats
# (HH:MM, MM:SS) are included — KU has running times like "25:50".
_VALUE_CONTEXT_PATTERN = re.compile(
    r"(?:[$€£¥])\s*[\d,]+(?:\.\d+)?[kmbt]?"        # currency
    r"|"
    r"\b\d+:\d{2}\b"                                 # HH:MM / MM:SS
    r"|"
    r"\b[\d,]+(?:\.\d+)?\s*"
    r"(?:%|percent|kg|lbs?|mi(?:les)?|km|m|cm|mm|"
    r"hours?|hrs?|days?|years?|yrs?|months?|weeks?|wks?|"
    r"times?|x/(?:day|week|month)|/(?:day|week|month))\b",
    flags=re.IGNORECASE,
)
_TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z\-']{3,}")


def _stem(token: str) -> str:
    """Conservative suffix stripping. Catches the common
    inflection ('approved' / 'approval' → 'approv') that breaks naive
    string equality on near-duplicate facts. Not a full stemmer — we
    deliberately leave longer suffixes alone to avoid over-collapsing."""
    t = token.lower().rstrip("'").strip("-")
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ed",
                "es", "ly", "al", "s"):
        if t.endswith(suf) and len(t) - len(suf) >= 4:
            return t[: -len(suf)]
    return t


_AUTO_TOPK = 5


def _auto_topics(content: str) -> set[str]:
    """Return the top-K stemmed tokens from a value-context memory,
    ranked by positional proximity to the unit-bearing value.

    Returns an empty set when the content has no unit-bearing value
    (so the memory isn't eligible for auto-topic supersession) or when
    no meaningful tokens survive stop-word + length filtering.

    Why a set instead of one token: positional proximity alone picks
    different anchors for paraphrased facts (m1 → "wells", m2 → "bumped"
    for Wells Fargo). Using top-K and matching on ANY shared element
    catches the load-bearing noun ("preapprov" appears in both top-Ks)
    without needing semantic understanding.
    """
    if not _VALUE_CONTEXT_PATTERN.search(content):
        return set()
    s = _DATE_TAG_PATTERN.sub("", content.lower())
    value_spans = [m.span() for m in _VALUE_CONTEXT_PATTERN.finditer(s)]
    if not value_spans:
        return set()
    tokens: list[tuple[int, str]] = []
    for m in _TOKEN_PATTERN.finditer(s):
        tok = m.group(0)
        if tok in _STOP or _stem(tok) in _STOP:
            continue
        tokens.append((m.start(), _stem(tok)))
    if not tokens:
        return set()

    def _dist_to_value(pos: int) -> int:
        return min(
            min(abs(pos - vs), abs(pos - ve))
            for vs, ve in value_spans
        )

    tokens.sort(key=lambda p: _dist_to_value(p[0]))
    return {tok for _, tok in tokens[:_AUTO_TOPK]}


def _content_skeleton(content: str) -> str:
    """Lower-case, strip unit-bearing numeric values + inline date tags,
    normalize whitespace.

    Two memories with the same skeleton are talking about the same fact at
    different points in time (or with different values). Used as the
    grouping key for entity-None contradiction detection.
    """
    s = _DATE_TAG_PATTERN.sub("", content.lower())
    s = _NUM_PATTERN.sub("NUM", s)
    s = _WS_PATTERN.sub(" ", s).strip()
    return s


class TemporalManager:
    """Handles temporal queries, contradiction detection, and supersession."""

    def __init__(self, store: DocumentStore):
        self.store = store

    def timeline(
        self, entity: str, namespace: str | None = None
    ) -> list[Memory]:
        """Get all memories about an entity ordered by event_date/created_at."""
        memories = self.store.list_memories(
            entity=entity, namespace=namespace, limit=100_000
        )
        return sorted(
            memories,
            key=lambda m: m.event_date or m.created_at,
        )

    def current_facts(
        self,
        category: str | None = None,
        entity: str | None = None,
        namespace: str | None = None,
    ) -> list[Memory]:
        """Return only active, non-superseded memories."""
        memories = self.store.list_memories(
            status="active", category=category, entity=entity,
            namespace=namespace, limit=100_000,
        )
        return [m for m in memories if m.valid_until is None]

    def check_contradictions(self, new_memory: Memory) -> list[Memory]:
        """Find active memories that potentially contradict the new one.

        Two strategies:

        1. **Entity-tagged path** (existing): same entity + same category +
           same namespace + different content. This is the original v3
           rule and stays intact for entity-rich callers.

        2. **Entity-None fallback** (added 2026-05-02 for KU sample
           852ce960): same category + same namespace + same content
           skeleton (numeric-stripped) + different value. Catches LME-S
           knowledge-update pairs like "pre-approved for $350,000" vs
           "pre-approved for $400,000" where the LME extractor never
           tagged an entity. Does NOT fire on "Likes Python" vs "Likes
           JavaScript" because their skeletons differ — so this is safe
           for plain preference memories.
        """
        if new_memory.entity:
            candidates = self.store.list_memories(
                status="active",
                category=new_memory.category,
                entity=new_memory.entity,
                namespace=new_memory.namespace,
                limit=100_000,
            )
        else:
            # Entity-None fallback: skeleton match. Narrow by
            # category+namespace, then filter by content-skeleton equality
            # (same template, different unit-bearing value). Catches
            # "pre-approved for $350,000" vs "pre-approved for $400,000"
            # without false-firing on "Likes Python" vs "Likes JavaScript"
            # (different skeletons).
            #
            # We DELIBERATELY do not have a semantic-similarity fallback
            # for cross-template paraphrases. Empirical 2026-05-03 data on
            # Pinecone llama-text-embed-v2 (1024-D) on the actual KU
            # 852ce960 wording:
            #     - "Likes Python" vs "Likes JavaScript"            d=0.086
            #     - "I'm pre-approved for $350k from Wells Fargo"
            #         vs "My pre-approval was bumped to $400k"      d=0.229
            # The structurally-similar preference pair is closer than the
            # semantically-equivalent paraphrase pair, so no single
            # threshold distinguishes them. The cross-template paraphrase
            # case needs an entity-extractor upgrade (auto-tag amount-
            # bearing memories) or LLM-judged contradiction — see
            # tests/test_temporal_supersession_gaps.py xfail Gap 5.
            if len(new_memory.content.strip()) < _MIN_FALLBACK_LEN:
                return []
            new_skel = _content_skeleton(new_memory.content)
            new_topics = _auto_topics(new_memory.content)
            same_category = self.store.list_memories(
                status="active",
                category=new_memory.category,
                namespace=new_memory.namespace,
                limit=100_000,
            )
            # Pass 1 — skeleton match. Same template, different
            # unit-bearing value. Disabled when no values were actually
            # stripped (skeleton == content), since two distinct facts
            # without values shouldn't be auto-collapsed.
            candidates: list[Memory] = []
            if new_skel != new_memory.content.strip().lower():
                candidates = [
                    c for c in same_category
                    if not c.entity
                    and _content_skeleton(c.content) == new_skel
                ]
            # Pass 2 — auto-topic match. Cross-template paraphrases
            # anchored on a noun near a unit-bearing value. Two memories
            # share a topic when their top-K topic sets intersect — e.g.
            # m1 = {"wells", "fargo", "preapprov"}, m2 = {"bump",
            # "preapprov", ...}: the shared "preapprov" closes the loop
            # without needing semantic understanding.
            if not candidates and new_topics:
                candidates = [
                    c for c in same_category
                    if not c.entity
                    and _auto_topics(c.content) & new_topics
                ]

        contradictions = []
        for existing in candidates:
            if existing.valid_until is not None:
                continue
            if existing.id == new_memory.id:
                continue
            if existing.content.strip() != new_memory.content.strip():
                contradictions.append(existing)
        return contradictions

    def supersede(self, old_memory: Memory, new_memory_id: str) -> Memory:
        """Mark old memory as superseded by a new one."""
        old_memory = replace(
            old_memory,
            status="superseded",
            valid_until=datetime.now(timezone.utc).isoformat(),
            superseded_by=new_memory_id,
        )
        return self.store.update(old_memory)
