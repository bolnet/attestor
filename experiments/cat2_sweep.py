"""Sweep the 7 LME-S Cat-2 (date arithmetic) wrong samples through any
configured (model, mode) pair. Two modes default-supported:

    1. ``none``              — no tools, plain Chat Completions or Anthropic
                               Messages depending on the model. Control cell.
    2. ``chat-tools``        — portable OpenAI-compatible function tools
                               (compute_diff / sum_durations). Works on any
                               Chat Completions endpoint that supports
                               function-calling.
    3. ``anthropic-native``  — Anthropic Messages API + the server-side
                               code_execution_20250522 tool. claude-* only.

Per-sample heuristic hint:
    REAL_CALC   — none-mode wrong, with-tools fixed it (real calc error)
    FORMAT      — none had right number wrong unit ("3 weeks 1 day" vs "3 weeks")
    BOTH_OK     — both correct; no calc-error class issue
    STILL_WRONG — neither matches gold; usually retrieval or date-extraction

Examples:
    .venv/bin/python experiments/cat2_sweep.py    # default: gpt-5.5 baseline + chat-tools
    .venv/bin/python experiments/cat2_sweep.py --model claude-sonnet-4-5-20250929 --tool-mode anthropic-native
    .venv/bin/python experiments/cat2_sweep.py --model claude-haiku-4-5-20251001 --tool-mode anthropic-native
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

# Make ``attestor`` importable when run with the .venv python directly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

# Reuse the helpers from the POC.
from experiments.code_exec_poc import (  # noqa: E402
    _build_prompt,
    _execute_tool,
    _load_sample,
    _openai_client_for,
    CHAT_TOOLS,
    PROMPT_INSTRUCTION_TAIL_FUNCTIONS,
    PROMPT_INSTRUCTION_TAIL_NATIVE,
)


CAT2_SAMPLES = [
    ("#4",  "gpt4_cd90e484", "binoculars before goldfinches"),
    ("#11", "gpt4_a1b77f9c", "reading + listening total weeks"),
    ("#13", "gpt4_4fc4f797", "suspension feedback → test"),
    ("#14", "gpt4_61e13b3c", "Farmers' Market → Spring Fling"),
    ("#16", "370a8ff4",      "flu recovery → 10th jog (weeks)"),
    ("#19", "gpt4_21adecb5", "undergrad → thesis (months)"),
    ("#20", "gpt4_85da3956", "weeks since Summer Nights"),
]

MAX_TOKENS = 2048
MAX_ROUNDS = 6


# ── plain Chat Completions (works for any OpenAI-compatible endpoint) ─────

def _run_none_chat(sample: dict[str, Any], model: str) -> dict[str, Any]:
    client, clean_model = _openai_client_for(model)
    prompt = _build_prompt(sample, "")
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=clean_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=MAX_TOKENS,
    )
    latency = round((time.monotonic() - t0) * 1000, 1)
    msg = resp.choices[0].message
    return {
        "mode": "none-chat",
        "answer": (msg.content or "").strip(),
        "latency_ms": latency,
        "tool_calls": 0,
    }


def _run_chat_tools(sample: dict[str, Any], model: str) -> dict[str, Any]:
    client, clean_model = _openai_client_for(model)
    prompt = _build_prompt(sample, PROMPT_INSTRUCTION_TAIL_FUNCTIONS)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    tool_call_count = 0
    final_text = ""
    t0 = time.monotonic()
    for _round in range(MAX_ROUNDS):
        resp = client.chat.completions.create(
            model=clean_model,
            messages=messages,
            tools=CHAT_TOOLS,
            max_tokens=MAX_TOKENS,
        )
        choice = resp.choices[0]
        msg = choice.message
        if msg.content:
            final_text = msg.content.strip()
        assistant_turn: dict[str, Any] = {
            "role": "assistant",
            "content": msg.content or "",
        }
        if msg.tool_calls:
            assistant_turn["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in msg.tool_calls
            ]
        messages.append(assistant_turn)
        if not msg.tool_calls:
            break
        for tc in msg.tool_calls:
            tool_call_count += 1
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = _execute_tool(tc.function.name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })
    latency = round((time.monotonic() - t0) * 1000, 1)
    return {
        "mode": "chat-tools",
        "answer": final_text,
        "latency_ms": latency,
        "tool_calls": tool_call_count,
    }


# ── Anthropic native (Messages API + server-side code_execution) ───────────

def _run_anthropic_native(sample: dict[str, Any], model: str) -> dict[str, Any]:
    """Anthropic Messages API + server-side code_execution sandbox."""
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)
    prompt = _build_prompt(sample, PROMPT_INSTRUCTION_TAIL_NATIVE)
    t0 = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "code_execution_20250522", "name": "code_execution"}],
        extra_headers={"anthropic-beta": "code-execution-2025-05-22"},
    )
    latency = round((time.monotonic() - t0) * 1000, 1)
    final_text = ""
    tool_calls = 0
    for block in response.content:
        bt = getattr(block, "type", "")
        if bt == "text":
            final_text = block.text.strip()
        elif bt == "server_tool_use":
            tool_calls += 1
    return {
        "mode": "anthropic-native",
        "answer": final_text,
        "latency_ms": latency,
        "tool_calls": tool_calls,
    }


def _run_anthropic_none(sample: dict[str, Any], model: str) -> dict[str, Any]:
    """Anthropic Messages API, no tools — control for the anthropic-native cell."""
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set")
    client = Anthropic(api_key=api_key)
    prompt = _build_prompt(sample, "")
    t0 = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        messages=[{"role": "user", "content": prompt}],
    )
    latency = round((time.monotonic() - t0) * 1000, 1)
    final_text = ""
    for block in response.content:
        if getattr(block, "type", "") == "text":
            final_text = block.text.strip()
    return {
        "mode": "none-anthropic",
        "answer": final_text,
        "latency_ms": latency,
        "tool_calls": 0,
    }


def _run_baseline_for_mode(sample: dict[str, Any], model: str, tool_mode: str) -> dict[str, Any]:
    if tool_mode == "anthropic-native":
        return _run_anthropic_none(sample, model)
    return _run_none_chat(sample, model)


def _run_with_tools(sample: dict[str, Any], model: str, tool_mode: str) -> dict[str, Any]:
    if tool_mode == "anthropic-native":
        return _run_anthropic_native(sample, model)
    if tool_mode == "chat-tools":
        return _run_chat_tools(sample, model)
    raise ValueError(f"unknown tool_mode {tool_mode!r}")


def _last_text(answer: str) -> str:
    """Strip the <reasoning>...</reasoning> block; take what's after."""
    if "</reasoning>" in answer:
        answer = answer.split("</reasoning>", 1)[1]
    return answer.strip()


def _gold_match_hint(gold: str, answer: str) -> str:
    """Coarse heuristic: does the answer match the gold's primary number/word?"""
    gold = gold or ""
    ans = _last_text(answer or "").lower()
    g_low = gold.lower()
    # First: full string substring (digits or words)
    if g_low and g_low.split(",")[0].split(".")[0].strip() in ans:
        return "MATCH"
    # Try numeric tokens
    g_nums = re.findall(r"\d+", gold)
    a_nums = re.findall(r"\d+", ans)
    # Take leading numeric tokens to match (rough)
    for gn in g_nums[:1]:
        if gn in a_nums:
            return "NUMERIC_MATCH"
    # Word-number mapping for "two", "three", etc.
    word_to_num = {"one":"1","two":"2","three":"3","four":"4","five":"5",
                   "six":"6","seven":"7","eight":"8","nine":"9","ten":"10",
                   "eleven":"11","twelve":"12"}
    for w, n in word_to_num.items():
        if w in g_low and n in a_nums:
            return "WORD_NUMERIC_MATCH"
        if w in ans and any(g == n for g in g_nums):
            return "NUMERIC_WORD_MATCH"
    return "NO_MATCH"


def _classify(none_match: str, tools_match: str, none_ans: str, tools_ans: str, gold: str) -> str:
    """Tag the sample's recovery type."""
    none_ok = none_match in {"MATCH", "NUMERIC_MATCH", "WORD_NUMERIC_MATCH", "NUMERIC_WORD_MATCH"}
    tools_ok = tools_match in {"MATCH", "NUMERIC_MATCH", "WORD_NUMERIC_MATCH", "NUMERIC_WORD_MATCH"}
    if not none_ok and tools_ok:
        # Tool fixed something. Was it a calc error or a format mismatch?
        # Heuristic: if none_ans contains the gold's primary numeric
        # token but as part of a richer phrase (e.g. "3 weeks 1 day"
        # while gold = "3 weeks"), it's a FORMAT mismatch — the math was
        # right but the granularity was wrong.
        g_nums = re.findall(r"\d+", gold or "")
        if g_nums:
            principal = g_nums[0]
            none_text = _last_text(none_ans).lower()
            # Look for the gold number ALONG WITH extra units in none_ans
            if principal in none_text and any(
                f"{principal} {unit}" in none_text
                for unit in ("week", "weeks", "month", "months", "day", "days", "year", "years")
            ):
                return "FORMAT"
        return "REAL_CALC"
    if none_ok and tools_ok:
        return "BOTH_OK"
    if not none_ok and not tools_ok:
        return "STILL_WRONG"
    return "REGRESSION"  # tools made it worse


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", default="openrouter/openai/gpt-5.5",
                        help="Model id. For anthropic-native mode, pass a "
                             "bare claude id (e.g. claude-sonnet-4-5-20250929).")
    parser.add_argument("--tool-mode", default="chat-tools",
                        choices=["chat-tools", "anthropic-native"])
    parser.add_argument("--out", default=None,
                        help="Where to write the per-sample JSON. Defaults to "
                             "experiments/cat2_sweep_<safe_model>_<mode>.json")
    args = parser.parse_args()

    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", args.model)
    out_path = pathlib.Path(args.out) if args.out else (
        pathlib.Path(__file__).resolve().parent / f"cat2_sweep_{safe}_{args.tool_mode}.json"
    )

    print(f"\nSweeping {len(CAT2_SAMPLES)} Cat-2 samples")
    print(f"  model    : {args.model}")
    print(f"  tool-mode: {args.tool_mode}")
    print(f"  out      : {out_path}")
    print()

    rows: list[dict[str, Any]] = []
    for label, qid, brief in CAT2_SAMPLES:
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
        base_match = _gold_match_hint(gold, base["answer"])
        tools_match = _gold_match_hint(gold, tools["answer"])
        tag = _classify(base_match, tools_match, base["answer"], tools["answer"], gold)
        row = {
            "label": label,
            "qid": qid,
            "brief": brief,
            "gold": gold,
            "base_answer": _last_text(base["answer"])[:200],
            "base_match": base_match,
            "tools_answer": _last_text(tools["answer"])[:200],
            "tools_match": tools_match,
            "tools_calls": tools["tool_calls"],
            "tag": tag,
            "base_ms": base["latency_ms"],
            "tools_ms": tools["latency_ms"],
        }
        rows.append(row)
        print(f"  gold        : {gold[:120]!r}")
        print(f"  base        : {row['base_answer']!r}     [{base_match}]")
        print(f"  with tools  : {row['tools_answer']!r}    [{tools_match}]  ({tools['tool_calls']} tool calls)")
        print(f"  tag         : {tag}")
        print()

    print("\n" + "=" * 78)
    print(f"SUMMARY  model={args.model}  tool-mode={args.tool_mode}")
    print("=" * 78)
    print(f"{'#':5s} {'qid':18s} {'brief':35s} {'tag':14s}")
    for r in rows:
        tag = r.get("tag") or ("ERROR" if "error" in r else "?")
        print(f"{r['label']:5s} {r['qid']:18s} {r['brief']:35s} {tag:14s}")
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
