"""Knowledge-update wrong-sample sweep — claude-haiku-4.5 + oracle facts.

Mirrors the per-bucket harness in `cat2_sweep.py` / `cat56_sweep.py`. Three
wrong samples from `logs/lme_knowledge-update_GOLDEN.json` (the 2026-05-02
KU 10-sample smoke at 70% accuracy):

    6a1eabeb  charity 5K personal best         gold 25:50          got 27:12
    945e3d21  yoga class frequency             gold 3x/week        got 2x/week
    9ea5eabc  most recent family trip          gold Paris          got Hawaii

All three are characteristic knowledge-update failures: the user's stated
fact CHANGED across sessions and the production answerer (gpt-5.5 +
voyage retrieval) returned the older value instead of the most-recent.
This sweep tests whether haiku-4.5 with oracle answer-relevant sessions
gets the most-recent fact right — i.e. whether the failure is
retrieval-side (production retrieval surfaces older sessions and not the
most-recent) or answerer-side (model can't pick the most-recent fact
even when handed all of them).

Usage:
    set -a && source .env && set +a
    .venv/bin/python experiments/cat_ku_sweep.py
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import sys
import time
from typing import Any

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

from experiments.code_exec_poc import (  # noqa: E402
    _build_prompt,
    _execute_tool,
    _load_sample,
    _openai_client_for,
    CHAT_TOOLS,
    PROMPT_INSTRUCTION_TAIL_FUNCTIONS,
)
from experiments.cat2_sweep import (  # noqa: E402
    _classify,
    _gold_match_hint,
    _last_text,
    _run_chat_tools,
    _run_none_chat,
)


KU_SAMPLES = [
    ("KU#1",  "6a1eabeb", "charity 5K personal best (gold 25:50)"),
    ("KU#2",  "945e3d21", "yoga class frequency (gold 3x/week)"),
    ("KU#3",  "9ea5eabc", "most recent family trip (gold Paris)"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="openrouter/anthropic/claude-haiku-4.5")
    parser.add_argument("--tool-mode", default="chat-tools",
                        choices=["chat-tools", "none"])
    args = parser.parse_args()

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    out_path = pathlib.Path(__file__).resolve().parent / (
        f"cat_ku_sweep_{safe}_{args.tool_mode}.json"
    )

    print(f"\nKnowledge-update sweep")
    print(f"  model    : {args.model}")
    print(f"  tool-mode: {args.tool_mode}")
    print()

    rows: list[dict[str, Any]] = []
    for label, qid, brief in KU_SAMPLES:
        print(f"--- {label} {qid}  {brief} ---")
        sample = _load_sample(qid)
        gold = str(sample.get("answer") or "")
        try:
            base = _run_none_chat(sample, args.model)
            tools = _run_chat_tools(sample, args.model) if args.tool_mode == "chat-tools" else base
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows.append({"label": label, "qid": qid, "brief": brief, "gold": gold, "error": str(e)})
            continue

        base_match = _gold_match_hint(gold, base["answer"])
        tools_match = _gold_match_hint(gold, tools["answer"])
        tag = _classify(base_match, tools_match, base["answer"], tools["answer"], gold)

        row = {
            "label": label, "qid": qid, "brief": brief, "gold": gold,
            "base_answer": _last_text(base["answer"])[:300],
            "base_match": base_match,
            "tools_answer": _last_text(tools["answer"])[:300],
            "tools_match": tools_match,
            "tools_calls": tools["tool_calls"] if isinstance(tools.get("tool_calls"), int) else 0,
            "tag": tag,
            "base_ms": base["latency_ms"], "tools_ms": tools["latency_ms"],
        }
        rows.append(row)
        print(f"  gold        : {gold[:120]!r}")
        print(f"  base        : {row['base_answer']!r}     [{base_match}]")
        if args.tool_mode == "chat-tools":
            print(f"  with tools  : {row['tools_answer']!r}    [{tools_match}]  ({row['tools_calls']} tool calls)")
        print(f"  tag         : {tag}")
        print()

    print("\n" + "=" * 78)
    print(f"SUMMARY  model={args.model}  tool-mode={args.tool_mode}")
    print("=" * 78)
    for r in rows:
        tag = r.get("tag") or ("ERROR" if "error" in r else "?")
        print(f"  {r['label']:6s} {r['qid']:18s} {r['brief']:42s} {tag}")
    tags_count: dict[str, int] = {}
    for r in rows:
        tag = r.get("tag") or ("ERROR" if "error" in r else "?")
        tags_count[tag] = tags_count.get(tag, 0) + 1
    print()
    print("Tag counts:")
    for k, v in sorted(tags_count.items(), key=lambda x: -x[1]):
        print(f"  {k:14s}  {v}")

    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
