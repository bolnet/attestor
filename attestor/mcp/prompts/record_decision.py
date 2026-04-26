"""record_decision — capture an agent decision with rationale + evidence.

Wraps a memory_add call with category="decision" and forces the agent
to attach (a) rationale text and (b) evidence_ids citing the memories
the decision was based on. Without this, audit reviewers can't
reconstruct WHY a decision was made.
"""

from __future__ import annotations

RECORD_DECISION_PROMPT = """\
You are recording a decision you just made so future agents and human
auditors can review it.

Required fields:
  decision    : the action taken (one sentence, third person)
  rationale   : why you took it (one paragraph; cite evidence)
  evidence_ids: list of memory IDs the decision was based on
                (use [mem_<id>] format; pull from your recall results)
  reversibility: low | medium | high
  effective_at: ISO datetime; default = now

Output schema (JSON only):
{{
  "decision": "...",
  "rationale": "...",
  "evidence_ids": ["mem_id1", "mem_id2"],
  "reversibility": "low|medium|high",
  "effective_at": "<ISO datetime>"
}}

# IMPORTANT: A decision without evidence_ids is not auditable. If you
# cannot cite at least one memory, recall first or flag the decision
# as "no_prior_context": true.

Decision context:
  agent_id   : {agent_id}
  thread_id  : {thread_id}
  user_query : {user_query}

Output:
"""


def format_record_decision_prompt(
    *, agent_id: str, thread_id: str, user_query: str,
) -> str:
    return RECORD_DECISION_PROMPT.format(
        agent_id=agent_id, thread_id=thread_id, user_query=user_query,
    )
