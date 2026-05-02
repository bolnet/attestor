"""Sweep the 8 LME-S Cat-5 / Cat-6 wrong samples (retrieval-rooted in the
original 26-wrong root-cause analysis) through any configured (model, mode)
pair, with ORACLE FACTS substituted for production retrieval.

The goal is the diagnostic that confirms or refutes the "retrieval-rooted"
classification:

  * If haiku-4.5 with oracle facts (the LME-S ``answer_session_ids`` — the
    1-3 sessions the benchmark itself marks as containing the gold-relevant
    turns) STILL gets these wrong, the failure is *not* retrieval — the
    answerer can't compose / disambiguate even when fed the right facts.
    Re-categorize.
  * If haiku-4.5 gets them right with oracle facts, production retrieval
    is the real bottleneck (top_k / BM25 / reranker work).

Cat 5 — Retrieved wrong facts / wrong context window (6 samples)
Cat 6 — Close but wrong (2 samples)

Mirrors ``experiments/cat2_sweep.py`` end-to-end; reuses ``code_exec_poc``
helpers (``_build_prompt`` already injects the oracle facts via
``_format_oracle_context``).

Two tool-modes default-supported:
  - ``none``       — pure prompt + reasoning. Control. THIS IS THE PRIMARY
                     SIGNAL for Cat 5/6 since these aren't calc errors.
  - ``chat-tools`` — portable function tools (compute_diff / sum_durations).
                     Useful for #2 (wake-up-time) since composition involves
                     time arithmetic.
  - ``anthropic-native`` — Anthropic Messages API + code_execution.

Usage:
    set -a && source .env && set +a
    .venv/bin/python experiments/cat56_sweep.py \\
        --model openrouter/anthropic/claude-haiku-4.5 \\
        --tool-mode chat-tools
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
from typing import Any

# Make ``attestor`` importable when run with the .venv python directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Reuse helpers from the POC.
from experiments.cat2_sweep import (  # noqa: E402
    _classify,
    _gold_match_hint,
    _last_text,
    _run_baseline_for_mode,
    _run_with_tools,
)
from experiments.code_exec_poc import _load_sample  # noqa: E402


# (label, qid, brief, gold-fragment-for-context)
CAT56_SAMPLES: list[tuple[str, str, str]] = [
    ("#1",  "a3838d2b",      "charity events before Run for the Cure"),
    ("#2",  "gpt4_2c50253f", "wake up time Tue/Thu (composition?)"),
    ("#17", "gpt4_d6585ce8", "concert order past 2 months"),
    ("#21", "gpt4_d6585ce9", "music event last Saturday"),
    ("#22", "gpt4_1e4a8aec", "gardening activity 2 weeks ago"),
    ("#25", "9a707b82",      "cooking for friend couple days ago"),
    ("#24", "gpt4_59149c78", "art event 2 weeks ago, where"),
    ("#26", "eac54add",      "business milestone 4 weeks ago"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        default="openrouter/anthropic/claude-haiku-4.5",
        help=(
            "Model id. For anthropic-native mode, pass a bare claude id "
            "(e.g. claude-haiku-4-5-20251001)."
        ),
    )
    parser.add_argument(
        "--tool-mode",
        default="chat-tools",
        choices=["chat-tools", "anthropic-native"],
    )
    parser.add_argument(
        "--out",
        default=None,
        help=(
            "Where to write the per-sample JSON. Defaults to "
            "experiments/cat56_sweep_<safe_model>_<mode>.json"
        ),
    )
    args = parser.parse_args()

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    out_path = (
        pathlib.Path(args.out)
        if args.out
        else pathlib.Path(__file__).resolve().parent
        / f"cat56_sweep_{safe}_{args.tool_mode}.json"
    )

    print(f"\nSweeping {len(CAT56_SAMPLES)} Cat-5/6 (retrieval-rooted) samples")
    print(f"  model    : {args.model}")
    print(f"  tool-mode: {args.tool_mode}")
    print(f"  out      : {out_path}")
    print(
        "  oracle facts: yes (answer_session_ids only — production retrieval "
        "BYPASSED)\n"
    )

    rows: list[dict[str, Any]] = []
    for label, qid, brief in CAT56_SAMPLES:
        print(f"--- {label} {qid}  {brief} ---")
        sample = _load_sample(qid)
        gold = str(sample.get("answer") or "")
        try:
            base = _run_baseline_for_mode(sample, args.model, args.tool_mode)
            tools = _run_with_tools(sample, args.model, args.tool_mode)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows.append(
                {
                    "label": label,
                    "qid": qid,
                    "brief": brief,
                    "gold": gold,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue
        base_match = _gold_match_hint(gold, base["answer"])
        tools_match = _gold_match_hint(gold, tools["answer"])
        tag = _classify(
            base_match, tools_match, base["answer"], tools["answer"], gold
        )
        # Verdict per Cat 5/6 framing:
        #   ANSWERER_FIXABLE — oracle facts in front of model, base or tools
        #     gets it right → production retrieval was the real bottleneck.
        #   STILL_WRONG     — oracle facts in front of model, both modes
        #     wrong → not just retrieval; answerer can't compose. Re-cat.
        ok_set = {
            "MATCH",
            "NUMERIC_MATCH",
            "WORD_NUMERIC_MATCH",
            "NUMERIC_WORD_MATCH",
        }
        base_ok = base_match in ok_set
        tools_ok = tools_match in ok_set
        if base_ok or tools_ok:
            verdict = "ANSWERER_FIXABLE"  # = production retrieval needs fix
        else:
            verdict = "ANSWERER_LIMIT"  # = re-categorize, not pure retrieval
        row = {
            "label": label,
            "qid": qid,
            "brief": brief,
            "gold": gold,
            "base_answer": _last_text(base["answer"])[:300],
            "base_match": base_match,
            "tools_answer": _last_text(tools["answer"])[:300],
            "tools_match": tools_match,
            "tools_calls": tools["tool_calls"],
            "tag": tag,
            "verdict": verdict,
            "base_ms": base["latency_ms"],
            "tools_ms": tools["latency_ms"],
        }
        rows.append(row)
        print(f"  gold        : {gold[:120]!r}")
        print(f"  base        : {row['base_answer']!r}     [{base_match}]")
        print(
            f"  with tools  : {row['tools_answer']!r}    [{tools_match}]  "
            f"({tools['tool_calls']} tool calls)"
        )
        print(f"  tag         : {tag}    verdict: {verdict}")
        print()

    print("\n" + "=" * 78)
    print(
        f"SUMMARY  model={args.model}  tool-mode={args.tool_mode}  "
        f"oracle-facts=yes"
    )
    print("=" * 78)
    print(
        f"{'#':5s} {'qid':18s} {'brief':40s} {'tag':14s} {'verdict':18s}"
    )
    for r in rows:
        tag = r.get("tag") or ("ERROR" if "error" in r else "?")
        verdict = r.get("verdict") or ("ERROR" if "error" in r else "?")
        print(
            f"{r['label']:5s} {r['qid']:18s} {r['brief']:40s} "
            f"{tag:14s} {verdict:18s}"
        )
    verdicts: dict[str, int] = {}
    for r in rows:
        v = r.get("verdict") or ("ERROR" if "error" in r else "?")
        verdicts[v] = verdicts.get(v, 0) + 1
    print("\nVerdict counts:")
    for k, v in sorted(verdicts.items(), key=lambda x: -x[1]):
        print(f"  {k:18s}  {v}")

    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
