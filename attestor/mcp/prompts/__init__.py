"""Standard MCP prompts shipped with the Attestor server (Phase 8.2).

Five primitives that bake in the right defaults for multi-agent flows:

  record_decision      — capture an agent decision with rationale + evidence
  handoff_to           — produce a handoff package between agents
  resume_thread        — issue a chronological recall of a thread
  audit_decision       — fetch a memory + its episode + supersession chain
  propose_invalidation — flag an existing memory as superseded with rationale

Each prompt is a templated string + a thin format helper. The MCP server
exposes them via prompts/get; agents using the server "do the right
thing" by default.
"""

from attestor.mcp.prompts.audit_decision import (
    AUDIT_DECISION_PROMPT,
    format_audit_decision_prompt,
)
from attestor.mcp.prompts.handoff import (
    HANDOFF_PROMPT_TEMPLATE,
    format_handoff_prompt,
)
from attestor.mcp.prompts.propose_invalidation import (
    PROPOSE_INVALIDATION_PROMPT,
    format_propose_invalidation_prompt,
)
from attestor.mcp.prompts.record_decision import (
    RECORD_DECISION_PROMPT,
    format_record_decision_prompt,
)
from attestor.mcp.prompts.resume_thread import (
    RESUME_THREAD_PROMPT,
    format_resume_thread_prompt,
)

__all__ = [
    "RECORD_DECISION_PROMPT",
    "HANDOFF_PROMPT_TEMPLATE",
    "RESUME_THREAD_PROMPT",
    "AUDIT_DECISION_PROMPT",
    "PROPOSE_INVALIDATION_PROMPT",
    "format_record_decision_prompt",
    "format_handoff_prompt",
    "format_resume_thread_prompt",
    "format_audit_decision_prompt",
    "format_propose_invalidation_prompt",
]
