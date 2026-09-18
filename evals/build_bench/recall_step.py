# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Recall step: fetch the attestor context for one build step (Mode A library).

Emits a single JSON object on stdout so the Workflow orchestrator can capture
both the context to inject into the build subagent and the packed token count
(measured with the canonical ``estimate_tokens``):

    {"recall_tokens": 42, "n_memories": 3, "context": "..."}

Usage:
    python -m evals.build_bench.recall_step <store_path> <namespace> "<query>" \\
        [--budget N]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

from attestor.core import AgentMemory
from attestor.utils.tokens import estimate_tokens

from evals.build_bench._env import autoload_store_env


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="build_bench.recall_step")
    parser.add_argument("store_path", help="Attestor store path (e.g. ~/.attestor)")
    parser.add_argument("namespace", help="Benchmark namespace (isolated tenant)")
    parser.add_argument("query", help="What this build step needs to recall")
    parser.add_argument("--budget", type=int, default=1_000_000, help="Token budget")
    args = parser.parse_args(argv)

    logging.getLogger("attestor").setLevel(logging.ERROR)
    autoload_store_env(args.store_path)

    with AgentMemory(args.store_path) as mem:
        results = mem.recall(args.query, budget=args.budget, namespace=args.namespace)

    packed = sum(estimate_tokens(r.memory.content) for r in results)
    context = "\n".join(f"[{r.match_source}] {r.memory.content}" for r in results)

    json.dump(
        {"recall_tokens": packed, "n_memories": len(results), "context": context},
        sys.stdout,
    )
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
