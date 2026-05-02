"""Sweep the 5 LME-S Cat-4 (event-ordering) wrong samples through any
configured (model, mode) pair.

Cat-4 samples are ordering questions ("what was the order of X, Y, Z" /
"which came first, A or B?"). Reuses the same harness as cat2_sweep, but
the gold-match heuristic targets ENTITY ORDER rather than numeric tokens.

Per-sample heuristic hint:
    REAL_FIX    — none-mode wrong, with-tools fixed it (model can order
                  given oracle facts + a date-math tool, but couldn't on
                  its own).
    BOTH_OK     — both correct.
    STILL_WRONG — neither matches gold; failure is either retrieval-
                  rooted (incomplete entity set) or answerer-rooted
                  (wrong order of right entities).
    REGRESSION  — base correct, tools made it worse.

Examples:
    .venv/bin/python experiments/cat4_sweep.py \
        --model openrouter/anthropic/claude-haiku-4.5 \
        --tool-mode chat-tools
"""

from __future__ import annotations

import argparse
import json
import pathlib
import re
import sys
import time
from typing import Any

# Make ``attestor`` importable when run with the .venv python directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Reuse helpers from the POC + cat2 sweep.
from experiments.cat2_sweep import (  # noqa: E402
    _run_none_chat,
    _run_chat_tools,
    _run_anthropic_native,
    _run_anthropic_none,
    _run_baseline_for_mode,
    _run_with_tools,
    _last_text,
)
from experiments.code_exec_poc import _load_sample  # noqa: E402


# (label, qid, brief, gold_entities_in_order)
# gold_entities lists the discriminating tokens in the correct order so we
# can score "right entities, right order" vs "right entities, wrong order"
# vs "missing entities".
CAT4_SAMPLES: list[tuple[str, str, str, list[str]]] = [
    ("#6",  "gpt4_2f584639", "necklace vs photo album",
     ["photo album", "necklace"]),
    ("#9",  "gpt4_7f6b06db", "3 trips past 3 months",
     ["muir woods", "big sur", "yosemite"]),
    ("#10", "gpt4_18c2b244", "Walmart -> Ibotta -> ShopRite",
     ["walmart", "ibotta", "shoprite"]),
    ("#15", "gpt4_45189cb4", "January sports order",
     ["nba", "college football", "nfl"]),
    ("#18", "gpt4_f420262c", "airlines earliest to latest",
     ["jetblue", "delta", "united", "american airlines"]),
]


def _entity_positions(text: str, entities: list[str]) -> list[tuple[str, int]]:
    """Return [(entity, first-occurrence-index)] for entities present in text.

    Order in returned list reflects order of appearance in `text`.
    """
    text_low = (text or "").lower()
    found: list[tuple[str, int]] = []
    for ent in entities:
        idx = text_low.find(ent.lower())
        if idx >= 0:
            found.append((ent, idx))
    found.sort(key=lambda x: x[1])
    return found


def _ordering_match(gold_entities: list[str], answer: str) -> tuple[str, dict[str, Any]]:
    """Score an ordering answer.

    Returns (tag, detail). Tag is one of:
        MATCH               all gold entities present in correct order
        WRONG_ORDER         all gold entities present, wrong order
        PARTIAL_RIGHT_ORDER  some gold entities present, in correct relative order
        PARTIAL_WRONG_ORDER  some gold entities present, but order broken
        NO_MATCH             no gold entities present
    """
    ans = _last_text(answer or "")
    found = _entity_positions(ans, gold_entities)
    found_names = [e for e, _ in found]
    detail = {
        "found": found_names,
        "expected": gold_entities,
        "n_found": len(found_names),
        "n_expected": len(gold_entities),
    }
    if not found_names:
        return "NO_MATCH", detail
    # Order check on the SUBSET found
    expected_subset = [e for e in gold_entities if e in found_names]
    is_correct_order = found_names == expected_subset
    if len(found_names) == len(gold_entities):
        return ("MATCH" if is_correct_order else "WRONG_ORDER"), detail
    return ("PARTIAL_RIGHT_ORDER" if is_correct_order else "PARTIAL_WRONG_ORDER"), detail


def _classify_ordering(base_tag: str, tools_tag: str) -> str:
    """Recovery tag for cat-4 sweep."""
    base_ok = base_tag == "MATCH"
    tools_ok = tools_tag == "MATCH"
    if not base_ok and tools_ok:
        return "REAL_FIX"
    if base_ok and tools_ok:
        return "BOTH_OK"
    if base_ok and not tools_ok:
        return "REGRESSION"
    return "STILL_WRONG"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        default="openrouter/anthropic/claude-haiku-4.5",
        help="Model id, e.g. openrouter/anthropic/claude-haiku-4.5",
    )
    parser.add_argument(
        "--tool-mode",
        default="chat-tools",
        choices=["chat-tools", "anthropic-native"],
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path; defaults to experiments/cat4_sweep_<model>_<mode>.json",
    )
    args = parser.parse_args()

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    out_path = (
        pathlib.Path(args.out)
        if args.out
        else pathlib.Path(__file__).resolve().parent
        / f"cat4_sweep_{safe}_{args.tool_mode}.json"
    )

    print(f"\nSweeping {len(CAT4_SAMPLES)} Cat-4 (event-ordering) samples")
    print(f"  model    : {args.model}")
    print(f"  tool-mode: {args.tool_mode}")
    print(f"  out      : {out_path}")
    print()

    rows: list[dict[str, Any]] = []
    t_start = time.monotonic()
    for label, qid, brief, gold_entities in CAT4_SAMPLES:
        print(f"--- {label} {qid}  {brief} ---")
        sample = _load_sample(qid)
        gold = str(sample.get("answer") or "")
        try:
            base = _run_baseline_for_mode(sample, args.model, args.tool_mode)
            tools = _run_with_tools(sample, args.model, args.tool_mode)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows.append({
                "label": label, "qid": qid, "brief": brief, "gold": gold,
                "error": f"{type(e).__name__}: {e}",
            })
            continue
        base_tag, base_detail = _ordering_match(gold_entities, base["answer"])
        tools_tag, tools_detail = _ordering_match(gold_entities, tools["answer"])
        recovery_tag = _classify_ordering(base_tag, tools_tag)
        row = {
            "label": label,
            "qid": qid,
            "brief": brief,
            "gold": gold,
            "gold_entities": gold_entities,
            "base_answer": _last_text(base["answer"])[:600],
            "base_tag": base_tag,
            "base_detail": base_detail,
            "tools_answer": _last_text(tools["answer"])[:600],
            "tools_tag": tools_tag,
            "tools_detail": tools_detail,
            "tools_calls": tools["tool_calls"],
            "recovery_tag": recovery_tag,
            "base_ms": base["latency_ms"],
            "tools_ms": tools["latency_ms"],
        }
        rows.append(row)
        print(f"  gold        : {gold[:120]!r}")
        print(f"  base        : {row['base_answer'][:160]!r}    [{base_tag}]")
        print(f"  with tools  : {row['tools_answer'][:160]!r}   [{tools_tag}]  ({tools['tool_calls']} calls)")
        print(f"  recovery    : {recovery_tag}")
        print()

    elapsed = round(time.monotonic() - t_start, 1)
    print("\n" + "=" * 78)
    print(f"SUMMARY  model={args.model}  tool-mode={args.tool_mode}  total={elapsed}s")
    print("=" * 78)
    print(f"{'#':5s} {'qid':18s} {'brief':35s} {'recovery':16s}")
    for r in rows:
        rt = r.get("recovery_tag") or ("ERROR" if "error" in r else "?")
        print(f"{r['label']:5s} {r['qid']:18s} {r['brief']:35s} {rt:16s}")
    tags_count: dict[str, int] = {}
    for r in rows:
        rt = r.get("recovery_tag") or ("ERROR" if "error" in r else "?")
        tags_count[rt] = tags_count.get(rt, 0) + 1
    print()
    print("Recovery tag counts:")
    for k, v in sorted(tags_count.items(), key=lambda x: -x[1]):
        print(f"  {k:16s}  {v}")

    out_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
