"""handoff_to — generate a handoff package between agents.

Cognition's session-handoff pattern (Devin), surfaced as a first-class
MCP primitive. Today every multi-agent team re-invents this badly;
shipping it as a default makes Attestor opinionated about the right
shape (summary + decisions + open questions + constraints + superseded
context).

Why include SUPERSEDED CONTEXT: the receiving agent shouldn't re-raise
questions that were already resolved earlier in the thread. This is
the single biggest source of "the new agent looped on a settled point"
in production multi-agent flows.
"""

from __future__ import annotations

HANDOFF_PROMPT_TEMPLATE = """\
You are handing off a task from {from_agent} to {to_agent}.

Produce a handoff package with:
1. SUMMARY (<= 200 words) -- what the thread is about, current state.
2. KEY DECISIONS -- bullet list, each with memory_id citation in
   [mem_<id>] form.
3. OPEN QUESTIONS -- what {to_agent} needs to decide next.
4. CONSTRAINTS -- any rules or commitments {to_agent} must honor.
5. SUPERSEDED CONTEXT -- facts that USED to be true but are no longer;
   include so {to_agent} doesn't accidentally re-raise resolved questions.

Format as Markdown. Cite every fact with [mem_<id>].

Recall budget: {recall_budget} tokens. Use the recall tool to gather
memories first; build the summary FROM the recall, not from imagination.

Thread context:
  thread_id : {thread_id}
  from_agent: {from_agent}
  to_agent  : {to_agent}
"""


def format_handoff_prompt(
    *,
    from_agent: str,
    to_agent: str,
    thread_id: str,
    recall_budget: int = 4000,
) -> str:
    return HANDOFF_PROMPT_TEMPLATE.format(
        from_agent=from_agent,
        to_agent=to_agent,
        thread_id=thread_id,
        recall_budget=recall_budget,
    )
