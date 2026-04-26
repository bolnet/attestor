"""Default Chain-of-Note reading prompt (Phase 6.1, roadmap §D.2).

Agents should prepend this to their system prompt when consuming a
``ContextPack`` from ``AgentMemory.recall_as_context``. Five steps:
NOTES → SYNTHESIS → CITE → ABSTAIN → CONFLICT.

The ABSTAIN clause is the load-bearing piece: AbstentionBench shows
every frontier model defaults to confabulation when context is
irrelevant unless explicitly instructed otherwise. With this clause,
abstention rates jump dramatically — and that's exactly the
LongMemEval ``abstention`` category Track D targets (≥80%).

The {memories_json} placeholder is filled by ``ContextPack.render_prompt``.
"""

from __future__ import annotations

DEFAULT_CHAIN_OF_NOTE_PROMPT = """\
You have been given a set of retrieved memories. Before answering, do the following:

1. NOTES: For each memory, write one line that says either:
   - "memory:<id> -- relevant: <why and which part>"
   - "memory:<id> -- irrelevant: <one-line reason>"

2. SYNTHESIS: Combine only the relevant notes to form your answer. If the
   memories' validity windows differ, prefer the one whose window covers the
   user's question.

3. CITE: When you state a fact derived from a memory, cite the memory ID in
   square brackets, e.g. [mem_42].

4. ABSTAIN: If no memory is relevant, say "I don't have that information"
   and do not invent. This is correct behavior, not a failure.

5. CONFLICT: If two memories contradict, prefer the one with the later
   `valid_from`, or the higher confidence if dates tie. Note the conflict
   in your answer if it would change the user's decision.

Memories (sorted by relevance):
{memories_json}
"""
