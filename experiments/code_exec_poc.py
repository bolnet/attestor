"""POC: portable tool-augmented answerer for LME-S Cat-2 calc errors.

Three modes, any model:

    --mode none              No tools. Pure prompt + reasoning. (Control.)
    --mode chat-tools        OpenAI-compatible Chat Completions function tools
                             — date-math helpers (compute_diff / month_diff /
                             sum_durations). Portable across every chat-
                             completion API: OpenAI direct, OpenRouter
                             (Anthropic / Mistral / Meta / DeepSeek / ...),
                             Anthropic via OpenRouter, etc. The runtime
                             executes the tools locally and feeds results back.
    --mode anthropic-native  Anthropic Messages API + the server-side
                             code_execution_20250522 tool. Only works for
                             claude-* models hitting the Anthropic endpoint
                             directly. Anthropic runs the sandbox.
    --mode openai-native     OpenAI Responses API + the built-in
                             code_interpreter tool. Only works on direct
                             OpenAI endpoints (api.openai.com), NOT
                             OpenRouter pass-through. OpenAI runs the sandbox.

The two non-control modes deliver the same goal — deterministic arithmetic
without LLM round-tripping — through different provider surfaces. ``chat-
tools`` is the portable path; ``anthropic-native`` is the upgrade path for
providers that ship a managed sandbox.

Loads one wrong sample from the LME-S source dataset, hands the oracle
facts (the 1-3 answer-relevant sessions only) to the model, prints the
full trace, compares the final answer to gold. Oracle facts isolate the
answerer lever from retrieval.

Usage:
    .venv/bin/python experiments/code_exec_poc.py                                  # default: claude-4.5 + anthropic-native, sample #14
    .venv/bin/python experiments/code_exec_poc.py --mode none                      # claude-4.5 control
    .venv/bin/python experiments/code_exec_poc.py --mode chat-tools --model openrouter/openai/gpt-5.5
    .venv/bin/python experiments/code_exec_poc.py --mode chat-tools --model openrouter/anthropic/claude-sonnet-4.5

The 7 Cat-2 (date arithmetic) wrong samples from the 2026-05-02 133q run:
    gpt4_cd90e484  binoculars before goldfinches      gold: 2 weeks
    <unknown>      total weeks reading + listening    gold: 8 weeks
    <unknown>      days between suspension and test   gold: 38
    gpt4_61e13b3c  Farmers' Market vs Spring Fling    gold: 3 weeks   <-- default
    <unknown>      weeks since flu recovery to 10th   gold: 15
    <unknown>      months between undergrad/thesis    gold: 6
    <unknown>      weeks since Summer Nights          gold: 3
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time
from typing import Any

# Allow running this script directly (.venv python) without `pip install -e .`
# — the .venv on disk was created from the previous repo root and the
# project itself isn't site-installed.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

DEFAULT_QID = "gpt4_61e13b3c"  # Farmers' Market → Spring Fling, gold=3 weeks
DEFAULT_MODEL = "claude-sonnet-4-5-20250929"
DEFAULT_MODE = "anthropic-native"

# Reuse the production answerer prompt verbatim — same tone, same rules,
# same worked examples — so any quality delta is attributable to the tool,
# not to a different prompt.
PROMPT_INSTRUCTION_TAIL_NATIVE = (
    "\n\nWhen the question requires date arithmetic, ordering by date, "
    "counting events that match a filter, or any other deterministic "
    "computation, you MUST call the code_execution tool. Do not try to "
    "do the math in your head. Show your final answer AFTER the tool "
    "result, in the format the prompt's WORKED EXAMPLES specify."
)
PROMPT_INSTRUCTION_TAIL_FUNCTIONS = (
    "\n\nFor date arithmetic (gaps, durations, ordering, counting events "
    "that match a date filter), you MUST call the provided functions "
    "(compute_diff / month_diff / sum_durations). Do not do the math in "
    "your head. After the function returns, give your final answer in "
    "the format the prompt's WORKED EXAMPLES specify."
)


# ── Portable function-tools (Chat Completions schema) ────────────────────
# Same JSON schema works for OpenAI direct, OpenRouter, Anthropic via
# OpenRouter, and any other provider that speaks OpenAI-compatible
# function calling.

CHAT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "compute_diff",
            "description": (
                "Calendar-aware difference between two dates in the requested unit. "
                "Semantics by unit:\n"
                "  - days  : exact day count (|d2 - d1| in days).\n"
                "  - weeks : days / 7 rounded per round_mode (default 'nearest'). "
                "Note 22 days = 3 weeks rounded nearest, NOT '3 weeks 1 day'.\n"
                "  - months: CALENDAR-MONTHS-ELAPSED, i.e. the number of month "
                "transitions between the two dates. e.g. 2022-11-17 → 2023-05-15 "
                "is 6 months (Nov→Dec→Jan→Feb→Mar→Apr→May). This matches the "
                "everyday-speech meaning of 'X months between events' that LME-S "
                "gold answers use. Use round_mode='down' for strict "
                "completed-months only (Nov-17→May-15 down → 5).\n"
                "  - years : calendar-years elapsed, anniversary-aware.\n"
                "Use for ANY 'how many days/weeks/months/years between/since X and Y' "
                "question. Do not do the math in your head — call this function."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "d1": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "d2": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "unit": {"type": "string", "enum": ["days", "weeks", "months", "years"]},
                    "round_mode": {
                        "type": "string",
                        "enum": ["nearest", "down", "up"],
                        "description": (
                            "Rounding when unit is weeks/months/years. "
                            "Default 'nearest'. Use 'down' for floor (only "
                            "completed units), 'up' for ceiling."
                        ),
                    },
                },
                "required": ["d1", "d2", "unit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "sum_durations",
            "description": (
                "Sum a list of (start_date, end_date) durations in the "
                "requested unit. Use for 'total weeks/days I did X' "
                "questions where the user did the activity in multiple "
                "non-overlapping spans."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "spans": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "start": {"type": "string", "description": "ISO YYYY-MM-DD"},
                                "end": {"type": "string", "description": "ISO YYYY-MM-DD"},
                            },
                            "required": ["start", "end"],
                        },
                    },
                    "unit": {"type": "string", "enum": ["days", "weeks", "months"]},
                    "round_mode": {
                        "type": "string",
                        "enum": ["nearest", "down", "up"],
                    },
                },
                "required": ["spans", "unit"],
            },
        },
    },
]


def _round(value: float, mode: str) -> int:
    if mode == "down":
        import math
        return math.floor(value)
    if mode == "up":
        import math
        return math.ceil(value)
    return round(value)


def _execute_tool(name: str, args: dict[str, Any]) -> str:
    """Run one of the date-math functions. Returns a JSON-serialisable string."""
    from datetime import date as _date
    try:
        if name == "compute_diff":
            d1 = _date.fromisoformat(args["d1"])
            d2 = _date.fromisoformat(args["d2"])
            unit = args["unit"]
            mode = args.get("round_mode", "nearest")
            days = abs((d2 - d1).days)
            if unit == "days":
                value: int = days
            elif unit == "weeks":
                value = _round(days / 7, mode)
            elif unit == "months":
                # Calendar-months-elapsed semantics — counts month transitions.
                # Matches the everyday-speech meaning of "X months between"
                # that LongMemEval gold answers use (e.g. 2022-11-17 →
                # 2023-05-15 = 6 months, not 5.93). round_mode='down' yields
                # strict completed-months (5 in that example).
                a, b = sorted([d1, d2])
                full_months = (b.year - a.year) * 12 + (b.month - a.month)
                if mode == "down":
                    value = full_months - 1 if b.day < a.day else full_months
                elif mode == "up":
                    value = full_months + 1 if b.day > a.day else full_months
                else:  # 'nearest' — calendar-month-elapsed (same as 'up' shy of full month)
                    value = full_months
            elif unit == "years":
                a, b = sorted([d1, d2])
                years = b.year - a.year
                if (b.month, b.day) < (a.month, a.day):
                    if mode == "down":
                        years -= 1
                    elif mode == "nearest":
                        # if we're past the half-year mark, count it as +1 of the next year
                        days_into_next = (b - _date(a.year + years, a.month, a.day)).days
                        if days_into_next > 182:
                            years += 1
                value = years
            else:
                return json.dumps({"error": f"unknown unit {unit!r}"})
            return json.dumps({"unit": unit, "value": value, "raw_days": days})
        if name == "sum_durations":
            spans = args.get("spans") or []
            total_days = 0
            for s in spans:
                a = _date.fromisoformat(s["start"])
                b = _date.fromisoformat(s["end"])
                total_days += abs((b - a).days)
            unit = args["unit"]
            mode = args.get("round_mode", "nearest")
            if unit == "days":
                value = total_days
            elif unit == "weeks":
                value = _round(total_days / 7, mode)
            elif unit == "months":
                value = _round(total_days / 30, mode)
            else:
                return json.dumps({"error": f"unknown unit {unit!r}"})
            return json.dumps({"unit": unit, "value": value, "raw_days": total_days})
        return json.dumps({"error": f"unknown tool {name!r}"})
    except Exception as e:  # noqa: BLE001
        return json.dumps({"error": f"{type(e).__name__}: {e}"})


def _load_sample(qid: str) -> dict[str, Any]:
    cache = pathlib.Path.home() / ".cache/attestor/longmemeval/longmemeval_s_cleaned.json"
    if not cache.exists():
        sys.exit(f"LME-S source not found at {cache}; run a benchmark once to populate the cache.")
    with cache.open() as f:
        data = json.load(f)
    for sample in data:
        if sample.get("question_id") == qid:
            return sample
    sys.exit(f"qid {qid!r} not found in LME-S source.")


def _format_oracle_context(sample: dict[str, Any]) -> str:
    """Build a context block from the answer-relevant sessions only.

    LME-S labels each sample with ``answer_session_ids`` — the 1-3 sessions
    that contain the gold-answer-relevant turns. We zip those against
    ``haystack_dates`` to date-prefix the turns the way the production
    ANSWER_PROMPT expects.
    """
    answer_ids = set(sample.get("answer_session_ids") or [])
    dates = sample.get("haystack_dates") or []
    sids = sample.get("haystack_session_ids") or []
    sessions = sample.get("haystack_sessions") or []

    lines: list[str] = []
    for date, sid, turns in zip(dates, sids, sessions):
        if sid not in answer_ids:
            continue
        for turn in turns:
            role = turn.get("role", "?")
            content = (turn.get("content") or "").strip().replace("\n", " ")
            if not content:
                continue
            lines.append(f"  - [{date}] ({role}): {content}")
    return "\n".join(lines) if lines else "(no facts)"


def _build_prompt(sample: dict[str, Any], tail: str) -> str:
    from attestor.longmemeval.prompts import ANSWER_PROMPT
    context = _format_oracle_context(sample)
    return ANSWER_PROMPT.format(
        question=sample["question"],
        question_date=sample.get("question_date") or "(unknown)",
        context=context,
    ) + tail


def _print_trace(response: Any) -> None:
    """Walk every content block in the response and pretty-print it."""
    print("=" * 78)
    print(f"stop_reason   : {response.stop_reason}")
    print(f"usage         : input={response.usage.input_tokens} output={response.usage.output_tokens}")
    print(f"content blocks: {len(response.content)}")
    print("=" * 78)
    for i, block in enumerate(response.content):
        bt = getattr(block, "type", "?")
        print(f"\n--- block {i}: type={bt} ---")
        if bt == "text":
            print(block.text)
        elif bt == "tool_use":
            print(f"  name={block.name}  id={block.id}")
            inp = getattr(block, "input", {}) or {}
            for k, v in inp.items():
                v_str = v if isinstance(v, str) else json.dumps(v)
                print(f"  {k}:")
                for line in v_str.splitlines():
                    print(f"    {line}")
        elif bt == "code_execution_tool_result":
            content = getattr(block, "content", None)
            if content is None:
                print("  (no content)")
                continue
            for sub in (content if isinstance(content, list) else [content]):
                stype = getattr(sub, "type", "?")
                if stype == "code_execution_result":
                    rc = getattr(sub, "return_code", None)
                    stdout = getattr(sub, "stdout", "") or ""
                    stderr = getattr(sub, "stderr", "") or ""
                    print(f"  return_code={rc}")
                    if stdout:
                        print(f"  stdout:")
                        for line in stdout.splitlines():
                            print(f"    {line}")
                    if stderr:
                        print(f"  stderr:")
                        for line in stderr.splitlines():
                            print(f"    {line}")
                else:
                    print(f"  ({stype}): {sub}")
        else:
            # Future-proof: print whatever attributes the block has.
            for attr in ("text", "input", "content"):
                if hasattr(block, attr):
                    print(f"  {attr}={getattr(block, attr)}")


def _run_anthropic_native(sample: dict[str, Any], model: str, max_tokens: int) -> tuple[str, float, int]:
    from anthropic import Anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("ANTHROPIC_API_KEY not set in environment.")
    client = Anthropic(api_key=api_key)
    prompt = _build_prompt(sample, PROMPT_INSTRUCTION_TAIL_NATIVE)
    t0 = time.monotonic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
        tools=[{"type": "code_execution_20250522", "name": "code_execution"}],
        extra_headers={"anthropic-beta": "code-execution-2025-05-22"},
    )
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    _print_trace(response)
    final_text = ""
    for block in response.content:
        if getattr(block, "type", "") == "text":
            final_text = block.text.strip()
    return final_text, elapsed_ms, response.usage.output_tokens


def _openai_client_for(model: str) -> tuple[Any, str]:
    """Pick the right OpenAI-compatible endpoint based on the model id prefix.

    Same scheme as the production codebase: ``provider/<rest>`` routes to
    that provider's OpenAI-compatible endpoint; bare ids go to
    OpenRouter as the catch-all.
    """
    from openai import OpenAI
    head, sep, tail = model.partition("/")
    if not sep:
        # Bare model id → OpenRouter default
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit("OPENROUTER_API_KEY not set; needed for bare model id")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), model
    if head == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            sys.exit("OPENAI_API_KEY not set")
        return OpenAI(api_key=api_key), tail
    if head == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            sys.exit("OPENROUTER_API_KEY not set")
        return OpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1"), tail
    if head == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            sys.exit("ANTHROPIC_API_KEY not set")
        # Anthropic exposes an OpenAI-compatible /chat/completions surface.
        return OpenAI(api_key=api_key, base_url="https://api.anthropic.com/v1/"), tail
    sys.exit(f"unknown provider prefix {head!r} in model {model!r}")


def _run_chat_tools(sample: dict[str, Any], model: str, max_tokens: int, max_rounds: int = 6) -> tuple[str, float, int]:
    """Portable Chat Completions function-tool loop. Works on any
    OpenAI-compatible endpoint that supports function calling."""
    client, clean_model = _openai_client_for(model)
    prompt = _build_prompt(sample, PROMPT_INSTRUCTION_TAIL_FUNCTIONS)
    messages: list[dict[str, Any]] = [{"role": "user", "content": prompt}]
    print("=" * 78)
    total_output_tokens = 0
    t0 = time.monotonic()
    final_text = ""
    for round_no in range(max_rounds):
        resp = client.chat.completions.create(
            model=clean_model,
            messages=messages,
            tools=CHAT_TOOLS,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message
        if resp.usage:
            total_output_tokens += resp.usage.completion_tokens or 0

        print(f"\n--- round {round_no} (finish_reason={choice.finish_reason}) ---")
        if msg.content:
            print(f"text: {msg.content.strip()[:600]}")
            final_text = msg.content.strip()

        # Append the assistant turn so the model sees its own tool_calls
        # echoed back when we feed tool results.
        assistant_turn: dict[str, Any] = {"role": "assistant", "content": msg.content or ""}
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
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError as e:
                args = {"_raw": tc.function.arguments, "_err": str(e)}
            print(f"  tool_call: {tc.function.name}({json.dumps(args)})")
            result = _execute_tool(tc.function.name, args)
            print(f"  → {result}")
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    print("=" * 78)
    return final_text, elapsed_ms, total_output_tokens


def _run_openai_native(sample: dict[str, Any], model: str, max_tokens: int) -> tuple[str, float, int]:
    """OpenAI Responses API + code_interpreter (managed sandbox).

    Requires direct OpenAI access — model id may be either a bare 'gpt-5.5'
    or the explicit 'openai/gpt-5.5' prefix; OpenRouter pass-through is
    NOT supported because Responses API isn't an OpenRouter surface.
    """
    from openai import OpenAI
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("OPENAI_API_KEY not set; openai-native mode needs direct OpenAI access")
    head, sep, tail = model.partition("/")
    if sep and head not in {"openai"}:
        sys.exit(f"openai-native only supports openai/<model> or bare model id; got {model!r}")
    clean_model = tail if sep else model
    client = OpenAI(api_key=api_key)

    prompt = _build_prompt(sample, PROMPT_INSTRUCTION_TAIL_NATIVE)
    instructions = (
        "You are answering an LME-S temporal-reasoning question. The full "
        "rubric, worked examples, and rules are below in the user input. "
        "When arithmetic is needed you MUST call the python tool — do not "
        "compute in your head."
    )

    t0 = time.monotonic()
    response = client.responses.create(
        model=clean_model,
        tools=[{"type": "code_interpreter", "container": {"type": "auto"}}],
        instructions=instructions,
        input=prompt,
        max_output_tokens=max_tokens,
    )
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)

    print("=" * 78)
    # Walk the output list — each item is a typed block (reasoning / message /
    # tool_use / tool_result). Print them all so we can see the trace.
    output = response.output if hasattr(response, "output") else []
    for i, item in enumerate(output):
        itype = getattr(item, "type", "?")
        print(f"\n--- output[{i}] type={itype} ---")
        if itype == "message":
            for c in (item.content or []):
                ctype = getattr(c, "type", "?")
                if ctype in {"output_text", "text"}:
                    print((getattr(c, "text", "") or "")[:600])
                else:
                    print(f"  ({ctype})")
        elif itype == "code_interpreter_call":
            code = getattr(item, "code", None) or getattr(item, "input", "") or ""
            for line in (code or "").splitlines()[:30]:
                print(f"  | {line}")
            if hasattr(item, "outputs") and item.outputs:
                for out in item.outputs:
                    otype = getattr(out, "type", "?")
                    if otype == "logs":
                        print(f"  stdout:")
                        for line in (getattr(out, "logs", "") or "").splitlines():
                            print(f"    {line}")
        elif itype == "reasoning":
            for c in (getattr(item, "summary", []) or []):
                txt = getattr(c, "text", "")
                if txt:
                    print(f"  reasoning: {txt[:300]}")
        else:
            for attr in ("text", "code", "input", "output_text"):
                if hasattr(item, attr):
                    val = getattr(item, attr)
                    if isinstance(val, str) and val:
                        print(f"  {attr}: {val[:400]}")
    print("=" * 78)

    final_text = getattr(response, "output_text", "") or ""
    out_tokens = 0
    if hasattr(response, "usage") and response.usage:
        out_tokens = getattr(response.usage, "output_tokens", 0) or 0
    return final_text, elapsed_ms, out_tokens


def _run_no_tools(sample: dict[str, Any], model: str, max_tokens: int) -> tuple[str, float, int]:
    """Plain Chat Completions, no tools, any model. Control cell."""
    client, clean_model = _openai_client_for(model)
    prompt = _build_prompt(sample, "")  # no tail instruction
    t0 = time.monotonic()
    resp = client.chat.completions.create(
        model=clean_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
    )
    elapsed_ms = round((time.monotonic() - t0) * 1000, 1)
    msg = resp.choices[0].message
    final_text = (msg.content or "").strip()
    output_tokens = (resp.usage.completion_tokens if resp.usage else 0) or 0
    print("=" * 78)
    print(f"text: {final_text[:600]}")
    print("=" * 78)
    return final_text, elapsed_ms, output_tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--qid", default=DEFAULT_QID, help="LME-S question_id to test")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=(
        "Model id. For 'anthropic-native' mode use a bare claude id "
        "(e.g. claude-sonnet-4-5-20250929). For 'chat-tools' or 'none' use "
        "provider-prefixed ids (openai/gpt-5.5, openrouter/anthropic/"
        "claude-sonnet-4.5, etc)."
    ))
    parser.add_argument(
        "--mode", default=DEFAULT_MODE,
        choices=["anthropic-native", "openai-native", "chat-tools", "none"],
        help=(
            "anthropic-native: Anthropic Messages API + code_execution server tool. "
            "openai-native: OpenAI Responses API + code_interpreter sandbox. "
            "chat-tools: portable OpenAI-compatible function tools (compute_diff / "
            "sum_durations). none: no tools, control cell."
        ),
    )
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()

    sample = _load_sample(args.qid)
    print(f"\nSAMPLE  qid={args.qid}  cat={sample.get('question_type')}")
    print(f"QUESTION  {sample['question']}")
    print(f"DATE      {sample.get('question_date')}")
    print(f"GOLD      {sample.get('answer')!r}")
    print(f"FACTS     {len(sample.get('answer_session_ids') or [])} oracle session(s) "
          f"out of {len(sample.get('haystack_session_ids') or [])} haystack")
    print(f"MODE      {args.mode}    MODEL    {args.model}")
    print()

    if args.mode == "anthropic-native":
        final_text, elapsed_ms, out_tokens = _run_anthropic_native(sample, args.model, args.max_tokens)
    elif args.mode == "openai-native":
        final_text, elapsed_ms, out_tokens = _run_openai_native(sample, args.model, args.max_tokens)
    elif args.mode == "chat-tools":
        final_text, elapsed_ms, out_tokens = _run_chat_tools(sample, args.model, args.max_tokens)
    else:
        final_text, elapsed_ms, out_tokens = _run_no_tools(sample, args.model, args.max_tokens)

    print()
    print(f"latency: {elapsed_ms} ms   output_tokens: {out_tokens}")
    print(f"final answer (last text block):")
    print(f"  {final_text[:400]}")
    gold = str(sample.get("answer") or "")
    print(f"gold: {gold!r}")
    print(f"contains gold token? {gold.lower() in final_text.lower()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
