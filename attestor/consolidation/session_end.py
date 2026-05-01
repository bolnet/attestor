"""Session-end promotion (Phase 7.4, roadmap §E task list).

When a session ends, decide what to do with each session-scoped memory:

  KEEP_SESSION   — leave it scoped to the session (transient, low-value)
  PROMOTE_PROJECT— widen scope to project (relevant beyond this chat)
  PROMOTE_USER   — widen scope to user (cross-project preference)
  DISCARD        — mark superseded (noise / one-off / hypothetical)

The LLM emits the decision; the apply layer mutates ``memories.scope``
(promotion) or status='superseded' (discard). Session-end consolidation
can run synchronously (small sessions) or be queued for the worker.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from typing import Any

from attestor.extraction.round_extractor import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_MODEL,
    _strip_markdown_fences,
)
from attestor.models import Memory

logger = logging.getLogger("attestor.consolidation.session_end")


VALID_PROMOTIONS = {"KEEP_SESSION", "PROMOTE_PROJECT", "PROMOTE_USER", "DISCARD"}


SESSION_PROMOTION_PROMPT = """\
You are reviewing memories captured during a single conversation session
that just ended. For each memory, decide what to do with it:

- KEEP_SESSION    : transient or low-value; relevant only inside this session
- PROMOTE_PROJECT : useful across other sessions in this project
- PROMOTE_USER    : a stable user preference / fact; useful across all projects
- DISCARD         : noise, one-off, hypothetical — mark superseded (kept for audit)

Decision rules:
1. If the memory is a CONCRETE PREFERENCE that would still apply tomorrow
   in any context (food, language, communication style) -> PROMOTE_USER.
2. If the memory is project-specific knowledge (architecture decision,
   API key, project conventions) -> PROMOTE_PROJECT.
3. If the memory references "today", "this conversation", or only makes
   sense inside this thread -> KEEP_SESSION.
4. If the memory is hypothetical ("what if I"), incomplete, or contradicted
   by a later memory in the same set -> DISCARD.

Output schema (JSON only):
{{
  "decisions": [
    {{
      "memory_id": "<id>",
      "operation": "KEEP_SESSION|PROMOTE_PROJECT|PROMOTE_USER|DISCARD",
      "rationale": "<one short sentence>"
    }}
  ]
}}

Session memories ({memory_count} items):
{memories_json}

Output:
"""


@dataclass(frozen=True)
class PromotionDecision:
    """One operation per memory in the ended session."""
    memory_id: str
    operation: str   # one of VALID_PROMOTIONS
    rationale: str

    def __post_init__(self) -> None:
        if self.operation not in VALID_PROMOTIONS:
            raise ValueError(
                f"PromotionDecision.operation must be one of {VALID_PROMOTIONS}; "
                f"got {self.operation!r}"
            )


@dataclass(frozen=True)
class AppliedPromotion:
    """The outcome of one PromotionDecision."""
    memory_id: str
    operation: str
    new_scope: str | None = None  # set on PROMOTE_PROJECT/PROMOTE_USER
    superseded: bool = False         # set on DISCARD
    error: str | None = None


def _memory_to_dict(m: Memory) -> dict[str, Any]:
    return {
        "id": m.id,
        "content": m.content,
        "category": m.category,
        "entity": m.entity,
        "valid_from": m.valid_from,
        "confidence": m.confidence,
    }


# ──────────────────────────────────────────────────────────────────────────
# Decide
# ──────────────────────────────────────────────────────────────────────────


def decide_promotions(
    memories: list[Memory],
    *,
    client: Any | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> list[PromotionDecision]:
    """Run SESSION_PROMOTION_PROMPT to classify each memory.

    Empty memories → []. LLM error → all-KEEP_SESSION (safe default:
    don't widen scope on uncertainty)."""
    if not memories:
        return []
    if client is None:
        return _all_keep(memories, "no llm client configured")

    prompt = SESSION_PROMOTION_PROMPT.format(
        memory_count=len(memories),
        memories_json=json.dumps(
            [_memory_to_dict(m) for m in memories], ensure_ascii=False,
        ),
    )
    try:
        from attestor.llm_trace import traced_create
        response = traced_create(
            client,
            role="session_end_promotion",
            model=model, max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.choices[0].message.content or ""
    except Exception as e:
        logger.warning("promotion LLM call failed: %s", e)
        return _all_keep(memories, f"llm error: {e}")

    return _parse_decisions(raw, memories)


def _all_keep(memories: list[Memory], reason: str) -> list[PromotionDecision]:
    """Safe default — no scope change. Used when the LLM is unavailable."""
    return [
        PromotionDecision(
            memory_id=m.id, operation="KEEP_SESSION", rationale=reason,
        )
        for m in memories
    ]


def _parse_decisions(
    raw: str, memories: list[Memory],
) -> list[PromotionDecision]:
    text = _strip_markdown_fences(raw)
    if not text:
        return _all_keep(memories, "empty llm response")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("promotion JSON parse failed: %s; raw=%r", e, raw[:200])
        return _all_keep(memories, "bad json")

    raw_decs: list = []
    if isinstance(parsed, dict):
        raw_decs = parsed.get("decisions", [])
    elif isinstance(parsed, list):
        raw_decs = parsed

    by_id: dict[str, dict] = {}
    for d in raw_decs:
        if not isinstance(d, dict):
            continue
        mid = d.get("memory_id")
        if isinstance(mid, str):
            by_id[mid] = d

    out: list[PromotionDecision] = []
    for m in memories:
        d = by_id.get(m.id)
        if d is None:
            out.append(PromotionDecision(
                memory_id=m.id, operation="KEEP_SESSION",
                rationale="LLM omitted decision; default KEEP",
            ))
            continue
        op = d.get("operation")
        if not isinstance(op, str) or op not in VALID_PROMOTIONS:
            out.append(PromotionDecision(
                memory_id=m.id, operation="KEEP_SESSION",
                rationale=f"invalid operation {op!r}; default KEEP",
            ))
            continue
        rationale = d.get("rationale")
        if not isinstance(rationale, str):
            rationale = "(no rationale)"
        out.append(PromotionDecision(
            memory_id=m.id, operation=op, rationale=rationale,
        ))
    return out


# ──────────────────────────────────────────────────────────────────────────
# Apply
# ──────────────────────────────────────────────────────────────────────────


def apply_promotions(
    decisions: list[PromotionDecision],
    *,
    mem: Any,                # AgentMemory (uses _store + _temporal)
) -> list[AppliedPromotion]:
    """Mutate memories.scope (promotions) or mark superseded (discard).

    KEEP_SESSION is a no-op outcome; included in the result list for
    audit symmetry.
    """
    out: list[AppliedPromotion] = []
    for d in decisions:
        try:
            if d.operation == "KEEP_SESSION":
                out.append(AppliedPromotion(
                    memory_id=d.memory_id, operation="KEEP_SESSION",
                ))
                continue
            row = mem._store.get(d.memory_id)
            if row is None:
                out.append(AppliedPromotion(
                    memory_id=d.memory_id, operation=d.operation,
                    error="memory vanished",
                ))
                continue
            if d.operation in {"PROMOTE_PROJECT", "PROMOTE_USER"}:
                new_scope = "project" if d.operation == "PROMOTE_PROJECT" else "user"
                row = replace(row, scope=new_scope)
                mem._store.update(row)
                out.append(AppliedPromotion(
                    memory_id=d.memory_id, operation=d.operation,
                    new_scope=new_scope,
                ))
            elif d.operation == "DISCARD":
                # Mark superseded (timeline preserved). No replacement
                # row — nothing to point to via superseded_by.
                from datetime import datetime, timezone
                row = replace(
                    row,
                    status="superseded",
                    valid_until=datetime.now(timezone.utc).isoformat(),
                )
                mem._store.update(row)
                out.append(AppliedPromotion(
                    memory_id=d.memory_id, operation="DISCARD",
                    superseded=True,
                ))
        except Exception as e:
            logger.warning(
                "apply_promotion %s for %s failed: %s",
                d.operation, d.memory_id, e,
            )
            out.append(AppliedPromotion(
                memory_id=d.memory_id, operation=d.operation,
                error=str(e),
            ))
    return out
