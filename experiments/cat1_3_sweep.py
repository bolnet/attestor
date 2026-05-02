"""Sweep the 6 LME-S Cat-1 (abstention-when-calculable) + Cat-3
(should-abstain-but-guessed) wrong samples through ``none`` vs
``chat-tools`` modes. Mirrors :mod:`experiments.cat2_sweep` but with
sample-aware match logic for the abstention buckets.

The 6 samples:

Cat 1 — Abstained when answer was calculable (4 samples):
    #3   d01c6aa8         "How old was I when I moved to the US?"   gold: 27
    #5   6613b389         "Months before anniversary Rachel got engaged?"  gold: 2
    #12  gpt4_7abb270c    "Order of six museums visited"
    #23  gpt4_f420262d    "Airline I flew on Valentine's day?"   gold: American Airlines

Cat 3 — Gold says "insufficient info" but AI guessed (2 samples):
    #7   gpt4_70e84552_abs   "Fixing fence or purchasing cows from Peter?"
    #8   gpt4_fe651585_abs   "Who became parent first, Tom or Alex?"

Tags emitted (per-sample):
    BOTH_OK       — both modes matched gold
    REAL_CALC     — base wrong, with-tools matched (real lift)
    STILL_WRONG   — neither matched gold (retrieval / dataset issue OR answerer
                    refused to commit even with tools)
    REGRESSION    — base ok, tools wrong

Usage:
    .venv/bin/python experiments/cat1_3_sweep.py \
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

# Reuse the helpers from the POC + cat2 sweep.
from experiments.code_exec_poc import (  # noqa: E402
    _load_sample,
)
from experiments.cat2_sweep import (  # noqa: E402
    _run_chat_tools,
    _run_none_chat,
    _last_text,
)


CAT1_3_SAMPLES: list[tuple[str, str, str, str]] = [
    # (label, qid, brief, bucket)
    ("#3",  "d01c6aa8",          "age when moved to US",                "cat1"),
    ("#5",  "6613b389",          "months before anniversary Rachel engaged",  "cat1"),
    ("#12", "gpt4_7abb270c",     "order of 6 museums visited",          "cat1"),
    ("#23", "gpt4_f420262d",     "airline on Valentine's day",          "cat1"),
    ("#7",  "gpt4_70e84552_abs", "fixing fence vs cows from Peter",     "cat3"),
    ("#8",  "gpt4_fe651585_abs", "Tom or Alex became parent first",     "cat3"),
]


# ── Sample-aware matching ─────────────────────────────────────────────────

ABSTAIN_PATTERNS = (
    "i don't know",            # canonical LME-S abstention per ANSWER_PROMPT
    "i do not know",
    "not enough info",
    "not enough information",
    "insufficient info",
    "insufficient information",
    "cannot determine",
    "can't determine",
    "cannot be determined",
    "no information",
    "did not mention",
    "didn't mention",
    "is not mentioned",
    "isn't mentioned",
    "no mention of",
    "not mentioned",
    "unable to determine",
    "unable to answer",
    "no record",
    "no record of",
    "i don't have",
    "i do not have",
    "no evidence",
)


def _is_abstention(answer: str) -> bool:
    """True if the answer text looks like a refusal-to-commit."""
    a = (answer or "").lower()
    return any(pat in a for pat in ABSTAIN_PATTERNS)


def _match_cat1(gold: str, answer: str) -> bool:
    """Cat-1 = should have computed. Match if answer contains gold."""
    if not gold:
        return False
    ans = _last_text(answer or "").lower()
    g_low = gold.lower().strip()

    # Numeric gold (#3 = "27", #5 = "2") — require digit token presence,
    # but also guard against the digit appearing only inside a date.
    if re.fullmatch(r"\d+", g_low):
        # Look for the digit as a standalone number/word in the answer.
        if re.search(rf"\b{re.escape(g_low)}\b", ans):
            # Avoid matching the gold digit purely as part of a YYYY date.
            if not re.search(rf"\b\d*{re.escape(g_low)}\d*\b(?:\W|$)", ans):
                return True
            # Re-check: simpler — ensure at least one boundary form like
            # "27 years", "27 months", "= 27", "is 27", standalone "27".
            patterns = [
                rf"\b{re.escape(g_low)}\s*(year|month|day|week)",
                rf"(=|is|was|were|been)\s*{re.escape(g_low)}\b",
                rf"^{re.escape(g_low)}$",
                rf"answer[^\n]*?\b{re.escape(g_low)}\b",
            ]
            return any(re.search(p, ans) for p in patterns)
        return False

    # String gold — substring match on the most distinctive token.
    # For #23 ("American Airlines"), require both words.
    # For #12, require the first and last museum (order constraint
    # is too complex for substring matching; we accept "ordered list
    # contains both endpoints in correct relative order").
    if "american airlines" in g_low:
        return "american airlines" in ans

    if "science museum" in g_low and "natural history" in g_low:
        # All 6 museums must appear AND the order must match the gold
        # sequence. Cheap test: check "science museum" appears before
        # "natural history" in the answer.
        sci = ans.find("science museum")
        nat = ans.find("natural history")
        if sci == -1 or nat == -1:
            return False
        # All six museums named in gold must be present.
        gold_museums = [m.strip() for m in gold.split(",")]
        if not all(m.lower() in ans for m in gold_museums):
            return False
        return sci < nat

    return g_low in ans


def _match_cat3(answer: str) -> bool:
    """Cat-3 = should have abstained. Match if answer is an abstention."""
    return _is_abstention(answer)


def _classify(
    bucket: str, gold: str, base_ans: str, tools_ans: str
) -> tuple[str, bool, bool]:
    """Return (tag, base_ok, tools_ok) for one sample."""
    if bucket == "cat1":
        base_ok = _match_cat1(gold, base_ans)
        tools_ok = _match_cat1(gold, tools_ans)
    else:  # cat3
        base_ok = _match_cat3(base_ans)
        tools_ok = _match_cat3(tools_ans)

    if base_ok and tools_ok:
        tag = "BOTH_OK"
    elif not base_ok and tools_ok:
        tag = "REAL_CALC"
    elif base_ok and not tools_ok:
        tag = "REGRESSION"
    else:
        tag = "STILL_WRONG"
    return tag, base_ok, tools_ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--model",
        default="openrouter/anthropic/claude-haiku-4.5",
        help="Model id; defaults to openrouter/anthropic/claude-haiku-4.5.",
    )
    parser.add_argument(
        "--tool-mode",
        default="chat-tools",
        choices=["chat-tools"],
        help="Only chat-tools is wired here (the cat-1/3 buckets aren't claude-only).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Where to write the per-sample JSON. Defaults to "
        "experiments/cat1_3_sweep_<safe_model>_<mode>.json",
    )
    args = parser.parse_args()

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    out_path = (
        pathlib.Path(args.out)
        if args.out
        else (
            pathlib.Path(__file__).resolve().parent
            / f"cat1_3_sweep_{safe}_{args.tool_mode}.json"
        )
    )

    print(f"\nSweeping {len(CAT1_3_SAMPLES)} Cat-1 + Cat-3 samples")
    print(f"  model    : {args.model}")
    print(f"  tool-mode: {args.tool_mode}")
    print(f"  out      : {out_path}")
    print()

    rows: list[dict[str, Any]] = []
    t_start = time.monotonic()
    for label, qid, brief, bucket in CAT1_3_SAMPLES:
        print(f"--- {label} {qid}  [{bucket}]  {brief} ---")
        sample = _load_sample(qid)
        gold = str(sample.get("answer") or "")
        try:
            base = _run_none_chat(sample, args.model)
            tools = _run_chat_tools(sample, args.model)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR: {type(e).__name__}: {e}")
            rows.append(
                {
                    "label": label,
                    "qid": qid,
                    "brief": brief,
                    "bucket": bucket,
                    "gold": gold,
                    "error": f"{type(e).__name__}: {e}",
                }
            )
            continue
        tag, base_ok, tools_ok = _classify(
            bucket, gold, base["answer"], tools["answer"]
        )
        row = {
            "label": label,
            "qid": qid,
            "brief": brief,
            "bucket": bucket,
            "gold": gold,
            "base_answer": _last_text(base["answer"])[:300],
            "base_ok": base_ok,
            "tools_answer": _last_text(tools["answer"])[:300],
            "tools_ok": tools_ok,
            "tools_calls": tools["tool_calls"],
            "tag": tag,
            "base_ms": base["latency_ms"],
            "tools_ms": tools["latency_ms"],
        }
        rows.append(row)
        print(f"  gold       : {gold[:150]!r}")
        print(f"  base       : {row['base_answer']!r}     [ok={base_ok}]")
        print(
            f"  with tools : {row['tools_answer']!r}    "
            f"[ok={tools_ok}]  ({tools['tool_calls']} tool calls)"
        )
        print(f"  tag        : {tag}")
        print()

    elapsed = round(time.monotonic() - t_start, 1)
    print("\n" + "=" * 78)
    print(f"SUMMARY  model={args.model}  tool-mode={args.tool_mode}  ({elapsed}s)")
    print("=" * 78)
    print(f"{'#':5s} {'qid':22s} {'bucket':6s} {'brief':40s} {'tag':14s}")
    for r in rows:
        tag = r.get("tag") or ("ERROR" if "error" in r else "?")
        print(
            f"{r['label']:5s} {r['qid']:22s} {r['bucket']:6s} "
            f"{r['brief']:40s} {tag:14s}"
        )
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
