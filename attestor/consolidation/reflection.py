"""Cross-thread reflection (Phase 7.3, roadmap §E.2).

Periodic synthesis over a user's recent facts. Produces:

  - stable_preferences   — patterns appearing in 3+ episodes
  - stable_constraints   — rules the user repeatedly invokes
  - changed_beliefs      — preferences that have shifted (old → new)
  - contradictions_for_review — same predicate, conflicting values, no
                                clear winner; HUMAN REVIEW only — never
                                auto-resolved (regulated chat systems)

This is LangMem's procedural memory shape: turning observable behavior
into explicit preferences the agent can rely on. Reflection runs after
the per-episode consolidator and is much cheaper (one LLM call across
N facts vs one per episode).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from attestor.extraction.round_extractor import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    _strip_markdown_fences,
)
from attestor.models import Memory

logger = logging.getLogger("attestor.consolidation.reflection")


REFLECTION_PROMPT = """\
You are a Senior Memory Curator reviewing facts collected across many
conversations with the same user. Identify:

1. STABLE PREFERENCES -- patterns appearing in 3+ episodes
2. STABLE CONSTRAINTS -- rules the user repeatedly invokes
3. CHANGED BELIEFS -- preferences that have shifted; mark old ones for INVALIDATE
4. CONTRADICTIONS -- same predicate, conflicting values, no clear winner
   (these must be flagged for HUMAN REVIEW; do NOT auto-resolve)

Output schema (JSON only):
{{
  "stable_preferences": [
    {{"text": "...", "evidence": ["mem_id1", "mem_id2", "mem_id3"], "confidence": 0.0-1.0}}
  ],
  "stable_constraints": [
    {{"text": "...", "evidence": ["mem_id1", ...], "confidence": 0.0-1.0}}
  ],
  "changed_beliefs": [
    {{"old": "mem_id", "new": "mem_id", "reason": "..."}}
  ],
  "contradictions_for_review": [
    {{"facts": ["mem_id1", "mem_id2"], "rationale": "..."}}
  ]
}}

# IMPORTANT: Do NOT auto-resolve contradictions. Flag them for human review.
# Only emit a stable_preference if at least 3 distinct evidence ids support it.

Facts to review (last 30 days, user_id {user_id}):
{facts_json}

Output:
"""


@dataclass(frozen=True)
class StablePattern:
    """A stable preference or constraint backed by evidence."""
    text: str
    evidence: List[str]
    confidence: float


@dataclass(frozen=True)
class ChangedBelief:
    """One belief that shifted: old memory_id → new memory_id."""
    old: str
    new: str
    reason: str


@dataclass(frozen=True)
class Contradiction:
    """Same predicate, conflicting values — for human review only."""
    facts: List[str]
    rationale: str


@dataclass(frozen=True)
class ReflectionResult:
    """Output of one reflect() call. Empty lists on LLM failure."""
    stable_preferences: List[StablePattern] = field(default_factory=list)
    stable_constraints: List[StablePattern] = field(default_factory=list)
    changed_beliefs: List[ChangedBelief] = field(default_factory=list)
    contradictions_for_review: List[Contradiction] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def total_patterns(self) -> int:
        return len(self.stable_preferences) + len(self.stable_constraints)


def _fact_to_dict(m: Memory) -> Dict[str, Any]:
    return {
        "id": m.id,
        "content": m.content,
        "category": m.category,
        "entity": m.entity,
        "valid_from": m.valid_from,
        "confidence": m.confidence,
        "source_episode_id": m.source_episode_id,
    }


def _parse_pattern_list(raw: Any) -> List[StablePattern]:
    out: List[StablePattern] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        text = entry.get("text")
        evidence = entry.get("evidence")
        confidence = entry.get("confidence", 0.5)
        if not isinstance(text, str) or not text.strip():
            continue
        if not isinstance(evidence, list):
            continue
        ev = [str(e) for e in evidence if e]
        try:
            conf = float(confidence)
        except (TypeError, ValueError):
            conf = 0.5
        conf = max(0.0, min(1.0, conf))
        out.append(StablePattern(
            text=text.strip(), evidence=ev, confidence=conf,
        ))
    return out


def _parse_changed_list(raw: Any) -> List[ChangedBelief]:
    out: List[ChangedBelief] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        old = entry.get("old")
        new = entry.get("new")
        reason = entry.get("reason", "")
        if not (isinstance(old, str) and isinstance(new, str)):
            continue
        out.append(ChangedBelief(
            old=old, new=new, reason=reason if isinstance(reason, str) else "",
        ))
    return out


def _parse_contradictions(raw: Any) -> List[Contradiction]:
    out: List[Contradiction] = []
    if not isinstance(raw, list):
        return out
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        facts = entry.get("facts")
        rationale = entry.get("rationale", "")
        if not isinstance(facts, list) or len(facts) < 2:
            continue
        out.append(Contradiction(
            facts=[str(f) for f in facts if f],
            rationale=rationale if isinstance(rationale, str) else "",
        ))
    return out


class ReflectionEngine:
    """Run REFLECTION_PROMPT over a list of facts.

    The engine is stateless — pass facts in, get patterns out. The
    consolidator decides when to run it (e.g., every N episodes).
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def reflect(
        self, facts: List[Memory], *, user_id: str = "(unknown)",
    ) -> ReflectionResult:
        """Synthesize patterns from a fact set. Empty/error → empty result."""
        if not facts:
            return ReflectionResult()
        if self._client is None:
            return ReflectionResult(error="no llm client configured")

        facts_json = json.dumps(
            [_fact_to_dict(f) for f in facts], ensure_ascii=False,
        )
        prompt = REFLECTION_PROMPT.format(
            user_id=user_id, facts_json=facts_json,
        )
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            logger.warning("reflection LLM call failed: %s", e)
            return ReflectionResult(error=f"llm: {e}")

        return _parse_response(raw)


def _parse_response(raw: str) -> ReflectionResult:
    text = _strip_markdown_fences(raw)
    if not text:
        return ReflectionResult()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("reflection JSON parse failed: %s; raw=%r", e, raw[:200])
        return ReflectionResult(error="bad json")
    if not isinstance(parsed, dict):
        return ReflectionResult(error="not an object")
    return ReflectionResult(
        stable_preferences=_parse_pattern_list(parsed.get("stable_preferences")),
        stable_constraints=_parse_pattern_list(parsed.get("stable_constraints")),
        changed_beliefs=_parse_changed_list(parsed.get("changed_beliefs")),
        contradictions_for_review=_parse_contradictions(
            parsed.get("contradictions_for_review")
        ),
    )
