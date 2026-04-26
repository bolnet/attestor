"""propose_invalidation — for reviewer agents to mark a memory superseded.

Reviewer-agent primitive. Different from the synchronous extractor's
INVALIDATE: this is run BY a reviewer agent looking at existing facts,
not the per-round extraction pipeline. The output is a structured
proposal that goes to the human review queue (or, with reviewer
authority granted, applied directly via the supersession path).
"""

from __future__ import annotations

PROPOSE_INVALIDATION_PROMPT = """\
You are a Reviewer Agent. You believe memory_id={target_memory_id} is no
longer accurate. Produce a structured invalidation proposal.

Required fields:
  target_id    : memory id you are proposing to invalidate
  rationale    : why this memory is no longer accurate (one paragraph)
  evidence_ids : memory IDs supporting your conclusion
  replacement  : optional — a new fact to ADD that replaces the old one
                 (same shape as a normal extracted fact)
  severity     : low | medium | high
                 (high: actively misleading; low: stale but not harmful)
  reviewer_id  : {reviewer_id}

Output schema (JSON only):
{{
  "target_id": "...",
  "rationale": "...",
  "evidence_ids": ["mem_id1", "mem_id2"],
  "replacement": null | {{"text": "...", "category": "...", "entity": "..."}},
  "severity": "low|medium|high",
  "reviewer_id": "{reviewer_id}",
  "needs_human_review": <bool>
}}

# IMPORTANT: Set needs_human_review=true if:
#   - severity is high
#   - the original memory has signature verification (regulated audit trail)
#   - your evidence is weaker than the original's confidence
# Otherwise, the supersession path applies your proposal directly.

Target memory + replacement candidates:
{target_payload}
"""


def format_propose_invalidation_prompt(
    *,
    target_memory_id: str,
    reviewer_id: str,
    target_payload: str,
) -> str:
    return PROPOSE_INVALIDATION_PROMPT.format(
        target_memory_id=target_memory_id,
        reviewer_id=reviewer_id,
        target_payload=target_payload,
    )
