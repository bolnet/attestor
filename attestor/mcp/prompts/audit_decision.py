"""audit_decision — fetch a memory + its episode + its supersession chain.

Auditor primitive. Given a memory_id, returns the canonical view that
a compliance reviewer needs:
  - the memory itself (current state)
  - the source episode (verbatim turns that produced it)
  - the supersession chain (every memory in the lineage, valid_from
    ordering preserved)
  - signature verification result
"""

from __future__ import annotations

AUDIT_DECISION_PROMPT = """\
You are an Auditor reviewing memory_id={memory_id}. Produce a structured
audit report.

Sections:
1. CURRENT STATE -- the memory as it exists right now (content, scope,
   confidence, status, agent_id, t_created).
2. PROVENANCE -- the source episode's verbatim user_turn_text and
   assistant_turn_text. Quote exactly.
3. SUPERSESSION CHAIN -- every memory in this lineage, oldest to newest.
   For each: id, content, status, valid_from, valid_until,
   superseded_by. The chain ends at a row whose superseded_by is NULL.
4. SIGNATURE -- whether the row has an Ed25519 signature attached and
   whether it verifies. Unsigned rows are flagged but not failed.
5. FINDINGS -- in plain English, what does this memory tell the
   auditor? Was the decision well-supported? Are any links broken?

# IMPORTANT: Output is for HUMAN reading. Use markdown headers + tables.
# Cite every quoted text with the source episode id.

Memory + episode + lineage payload:
{audit_payload}
"""


def format_audit_decision_prompt(
    *, memory_id: str, audit_payload: str,
) -> str:
    return AUDIT_DECISION_PROMPT.format(
        memory_id=memory_id, audit_payload=audit_payload,
    )
