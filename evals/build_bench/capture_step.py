# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Capture step: persist one build step's decisions + record the ledger row.

This is the only place that mutates the ledger. It (1) loads the persisted
ledger, (2) records the step -- which measures the naive ``full_history_tokens``
against the corpus BEFORE this step's decisions are appended, (3) writes the
decisions into attestor under the benchmark namespace, (4) saves the ledger, and
(5) prints the new row + cumulative summary as JSON on stdout.

Usage:
    python -m evals.build_bench.capture_step <store_path> <namespace> "<decisions>" \\
        --step 0.3.1 --recall-tokens 42 --n-memories 3 --ledger <ledger.json> \\
        [--tags a,b --category build --entity Auth]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from attestor.core import AgentMemory

from evals.build_bench._env import autoload_store_env
from evals.build_bench.ledger import BuildLedger


def _load_ledger(path: Path) -> BuildLedger:
    if path.exists():
        return BuildLedger.from_json(json.loads(path.read_text()))
    return BuildLedger()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_bench.capture_step")
    parser.add_argument("store_path", help="Attestor store path (e.g. ~/.attestor)")
    parser.add_argument("namespace", help="Benchmark namespace (isolated tenant)")
    parser.add_argument(
        "decisions",
        nargs="?",
        default=None,
        help="What this step built / decided (captured text). Omit when using "
        "--decisions-file (preferred for long, multi-line records).",
    )
    parser.add_argument(
        "--decisions-file",
        default=None,
        help="Read the step's decisions from this file instead of the positional arg.",
    )
    parser.add_argument("--step", required=True, help="Step id, e.g. 0.3.1")
    parser.add_argument("--recall-tokens", type=int, required=True)
    parser.add_argument("--n-memories", type=int, default=0)
    parser.add_argument("--ledger", required=True, help="Path to the ledger JSON file")
    parser.add_argument("--tags", default="", help="Comma-separated tags")
    parser.add_argument("--category", default="build", help="Memory category")
    parser.add_argument("--entity", default=None, help="Primary entity for the step")
    args = parser.parse_args(argv)

    logging.getLogger("attestor").setLevel(logging.ERROR)
    autoload_store_env(args.store_path)

    if args.decisions_file:
        decisions = Path(args.decisions_file).expanduser().read_text().strip()
    elif args.decisions is not None:
        decisions = args.decisions
    else:
        parser.error("provide decisions positionally or via --decisions-file")

    ledger_path = Path(args.ledger).expanduser()
    led = _load_ledger(ledger_path)

    # Record BEFORE persisting to attestor so the naive baseline reflects the
    # corpus through the previous step (recall-time state for this step).
    row = led.record_step(
        step=args.step,
        recall_tokens=args.recall_tokens,
        decisions=decisions,
        n_memories=args.n_memories,
    )

    tags = [t.strip() for t in args.tags.split(",") if t.strip()]
    with AgentMemory(args.store_path) as mem:
        mem.add(
            content=decisions,
            tags=tags,
            category=args.category,
            entity=args.entity,
            namespace=args.namespace,
        )

    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    ledger_path.write_text(json.dumps(led.to_json(), indent=2))

    json.dump({"row": row.__dict__, "summary": led.summary()}, sys.stdout)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
