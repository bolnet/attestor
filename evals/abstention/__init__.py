"""AbstentionBench runner (Phase 9.4, roadmap §G).

Tests two failure modes simultaneously:

  - Confabulation: model invents an answer when memory has nothing
    relevant. The Chain-of-Note ABSTAIN clause is supposed to prevent
    this.
  - Over-abstention: model says "I don't know" when memory actually
    contained a usable answer. Loose abstention prompts cause this.

Primary metric is F1 over the abstention DECISION — not answer
correctness. A system that abstains on every query gets perfect
recall and zero precision; the F1 captures the tradeoff.
"""
