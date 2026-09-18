# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Deterministic token-savings ledger for the build benchmark.

Single-run counterfactual: build the software once with attestor recall. At step
``i`` the ledger records two numbers using the canonical ``estimate_tokens``:

* ``recall_tokens``       -- what attestor actually packed for step ``i`` (flat).
* ``full_history_tokens`` -- tokens of the corpus produced through step ``i-1``,
  i.e. the context a memoryless ("naive") agent would have re-fed to do step ``i``.

The naive number is a *computed counterfactual* on the same run, not a second
executed build. Both baselines see the same information units (every step's
captured decisions); the naive agent re-feeds all of them, attestor feeds only
the relevant subset.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from attestor.utils.tokens import estimate_tokens


@dataclass(frozen=True)
class StepRow:
    """One build step's token accounting (immutable)."""

    step: str
    recall_tokens: int
    full_history_tokens: int
    saved_tokens: int
    savings_ratio: float
    n_memories: int = 0


@dataclass
class BuildLedger:
    """Accumulates the build corpus and per-step token-savings rows.

    ``corpus`` is the ordered list of every step's captured decision text -- the
    full information a naive agent would carry forward. Only ``record_step``
    mutates it, and it appends *after* measuring the current step's naive
    baseline so a step never inflates its own counterfactual.
    """

    corpus: list[str] = field(default_factory=list)
    rows: list[StepRow] = field(default_factory=list)

    def full_history_tokens(self) -> int:
        """Tokens of the entire corpus captured so far (the naive context size)."""
        if not self.corpus:
            return 0
        return estimate_tokens("\n".join(self.corpus))

    def record_step(
        self,
        step: str,
        recall_tokens: int,
        decisions: str,
        n_memories: int = 0,
    ) -> StepRow:
        """Record one step, returning its row.

        ``full_history_tokens`` is measured against the corpus *before* this
        step's decisions are appended -- that is exactly the recall-time state a
        naive agent would have faced for this step.
        """
        full_history = self.full_history_tokens()
        saved = full_history - recall_tokens
        ratio = round(full_history / recall_tokens, 2) if recall_tokens > 0 else 0.0
        row = StepRow(
            step=step,
            recall_tokens=recall_tokens,
            full_history_tokens=full_history,
            saved_tokens=saved,
            savings_ratio=ratio,
            n_memories=n_memories,
        )
        self.rows.append(row)
        if decisions:
            self.corpus.append(decisions)
        return row

    def summary(self) -> dict:
        """Cumulative rollup across all recorded steps."""
        cum_naive = sum(r.full_history_tokens for r in self.rows)
        cum_attestor = sum(r.recall_tokens for r in self.rows)
        return {
            "steps": len(self.rows),
            "cumulative_naive_tokens": cum_naive,
            "cumulative_attestor_tokens": cum_attestor,
            "cumulative_savings_ratio": (
                round(cum_naive / cum_attestor, 2) if cum_attestor > 0 else 0.0
            ),
            "total_saved_tokens": cum_naive - cum_attestor,
            "peak_full_history_tokens": max(
                (r.full_history_tokens for r in self.rows), default=0
            ),
            "peak_recall_tokens": max((r.recall_tokens for r in self.rows), default=0),
        }

    def to_json(self) -> dict:
        """Serialize ledger (rows + corpus + summary) for cross-process handoff."""
        return {
            "rows": [asdict(r) for r in self.rows],
            "corpus": list(self.corpus),
            "summary": self.summary(),
        }

    @classmethod
    def from_json(cls, data: dict) -> "BuildLedger":
        """Rehydrate a ledger persisted by ``to_json``."""
        led = cls(corpus=list(data.get("corpus", [])))
        led.rows = [StepRow(**r) for r in data.get("rows", [])]
        return led
