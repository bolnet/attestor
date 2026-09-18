# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Unit tests for the build-bench token-savings ledger (deterministic core).

The ledger implements the *single-run counterfactual*: build software once with
attestor recall, and at every step compare the recall-packed context against the
full-history context a memoryless agent would have had to re-feed.
"""

from __future__ import annotations

from evals.build_bench.ledger import BuildLedger, StepRow


def test_first_step_has_no_history() -> None:
    led = BuildLedger()
    row = led.record_step("0.3.1", recall_tokens=0, decisions="built auth provider", n_memories=0)
    # Nothing was captured before step 1 -> naive baseline is zero, no savings yet.
    assert row.full_history_tokens == 0
    assert row.saved_tokens == 0
    assert row.savings_ratio == 0.0


def test_full_history_grows_recall_stays_flat() -> None:
    led = BuildLedger()
    # Each step captures a chunk of decisions; recall stays small (relevant subset only).
    led.record_step("s1", recall_tokens=0, decisions="alpha " * 100)
    r2 = led.record_step("s2", recall_tokens=50, decisions="beta " * 100)
    r3 = led.record_step("s3", recall_tokens=50, decisions="gamma " * 100)
    # full_history at step i = tokens of corpus BEFORE step i (counterfactual naive context).
    assert r2.full_history_tokens > 0
    assert r3.full_history_tokens > r2.full_history_tokens  # history accumulates
    assert r3.recall_tokens == 50  # recall is flat regardless of history size


def test_savings_ratio_and_saved_tokens() -> None:
    led = BuildLedger()
    led.record_step("s1", recall_tokens=0, decisions="word " * 1000)
    row = led.record_step("s2", recall_tokens=10, decisions="x")
    # full_history ~ 1000 words * 1.3 = 1300 tokens; recall = 10 -> ratio ~130x.
    assert row.full_history_tokens > 1000
    assert row.saved_tokens == row.full_history_tokens - 10
    assert row.savings_ratio == round(row.full_history_tokens / 10, 2)


def test_full_history_measured_before_capturing_current_step() -> None:
    # The decisions of step i must NOT inflate step i's own naive baseline.
    led = BuildLedger()
    row = led.record_step("only", recall_tokens=5, decisions="huge " * 500, n_memories=1)
    assert row.full_history_tokens == 0  # corpus was empty when this step ran


def test_summary_aggregates_cumulative() -> None:
    led = BuildLedger()
    led.record_step("s1", recall_tokens=0, decisions="a " * 200)
    led.record_step("s2", recall_tokens=40, decisions="b " * 200)
    led.record_step("s3", recall_tokens=40, decisions="c " * 200)
    s = led.summary()
    assert s["steps"] == 3
    assert s["cumulative_attestor_tokens"] == 80
    assert s["cumulative_naive_tokens"] > s["cumulative_attestor_tokens"]
    assert s["total_saved_tokens"] == s["cumulative_naive_tokens"] - 80
    assert s["cumulative_savings_ratio"] > 1.0


def test_json_round_trip() -> None:
    led = BuildLedger()
    led.record_step("s1", recall_tokens=0, decisions="hello world")
    led.record_step("s2", recall_tokens=12, decisions="more text here", n_memories=2)
    blob = led.to_json()
    restored = BuildLedger.from_json(blob)
    assert restored.corpus == led.corpus
    assert restored.rows == led.rows
    assert isinstance(restored.rows[0], StepRow)
    # A restored ledger keeps accumulating correctly.
    r = restored.record_step("s3", recall_tokens=12, decisions="z")
    assert r.full_history_tokens > 0
