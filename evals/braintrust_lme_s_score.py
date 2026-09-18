"""Braintrust score: post-hoc LME-S judge on submission JSONLs.

Mirrors the upstream LongMemEval evaluator
(https://github.com/xiaowu0162/LongMemEval/blob/main/src/evaluation/evaluate_qa.py)
exactly:
  - same per-task prompt templates (incl. temporal off-by-one tolerance,
    knowledge-update updated-answer rule, abstention variant for `_abs`
    question_ids)
  - same judge call shape (temperature=0, max_tokens=10, n=1)
  - same scoring (`'yes' in response.lower()` → 1.0, else 0.0)

Differences from upstream:
  - Routed through OpenRouter when ``OPENAI_API_KEY`` is unset (the WIP
    YAML pins the answerer to ``openrouter/openai/gpt-4o-2024-08-06`` so
    judging via the same path keeps the leaderboard-comparable model
    while sharing one billing surface).
  - Logs each sample to Braintrust under ``attestor-lme-s/lme-<cat>:<suffix>``
    with input/output/expected + score + judge reasoning, instead of
    writing a flat ``.eval-results-*`` text file.

Run example:

    set -a && source .env && set +a

    .venv/bin/python evals/braintrust_lme_s_score.py \\
        --submission-dir submissions/lme_s_gpt4o_20260508_1854 \\
        --suffix submission-2026-05-09 \\
        --parallel 4
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname).1s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("eval.lme.score")
logging.getLogger("httpx").setLevel(logging.WARNING)

from braintrust import Eval
from openai import AsyncOpenAI


# ── LME-S categories in scope (Option B) ─────────────────────────────────

CATEGORIES: tuple[str, ...] = (
    "knowledge-update",
    "single-session-user",
    "temporal-reasoning",
    "multi-session",
)

CATEGORY_QUESTION_COUNTS: dict[str, int] = {
    "knowledge-update":     78,
    "single-session-user":  70,
    "temporal-reasoning":  133,
    "multi-session":       133,
}

PROJECT = "attestor-lme-s"

DEFAULT_REF_PATH = Path(
    "/Users/aarjay/Documents/longmemeval-bench/LongMemEval/data/longmemeval_s_cleaned.json"
)


# ── Upstream LME-S judge prompt templates ─────────────────────────────────
# Verbatim from longmemeval-bench/LongMemEval/src/evaluation/evaluate_qa.py
# `get_anscheck_prompt`. Do not paraphrase — leaderboard parity depends on
# byte-identical prompt strings.

_T_DEFAULT = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. \n\n"
    "Question: {}\n\n"
    "Correct Answer: {}\n\n"
    "Model Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_T_TEMPORAL = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response is equivalent to the correct answer or contains "
    "all the intermediate steps to get the correct answer, you should also "
    "answer yes. If the response only contains a subset of the information "
    "required by the answer, answer no. In addition, do not penalize "
    "off-by-one errors for the number of days. If the question asks for the "
    "number of days/weeks/months, etc., and the model makes off-by-one errors "
    "(e.g., predicting 19 days when the answer is 18), the model's response "
    "is still correct. \n\n"
    "Question: {}\n\n"
    "Correct Answer: {}\n\n"
    "Model Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_T_KU = (
    "I will give you a question, a correct answer, and a response from a model. "
    "Please answer yes if the response contains the correct answer. Otherwise, "
    "answer no. If the response contains some previous information along with "
    "an updated answer, the response should be considered as correct as long "
    "as the updated answer is the required answer.\n\n"
    "Question: {}\n\n"
    "Correct Answer: {}\n\n"
    "Model Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_T_PREFERENCE = (
    "I will give you a question, a rubric for desired personalized response, "
    "and a response from a model. Please answer yes if the response satisfies "
    "the desired response. Otherwise, answer no. The model does not need to "
    "reflect all the points in the rubric. The response is correct as long as "
    "it recalls and utilizes the user's personal information correctly.\n\n"
    "Question: {}\n\n"
    "Rubric: {}\n\n"
    "Model Response: {}\n\n"
    "Is the model response correct? Answer yes or no only."
)

_T_ABSTENTION = (
    "I will give you an unanswerable question, an explanation, and a response "
    "from a model. Please answer yes if the model correctly identifies the "
    "question as unanswerable. The model could say that the information is "
    "incomplete, or some other information is given but the asked information "
    "is not.\n\n"
    "Question: {}\n\n"
    "Explanation: {}\n\n"
    "Model Response: {}\n\n"
    "Does the model correctly identify the question as unanswerable? "
    "Answer yes or no only."
)


def build_judge_prompt(
    qtype: str, question: str, answer: str, hypothesis: str, abstention: bool
) -> str:
    if abstention:
        return _T_ABSTENTION.format(question, answer, hypothesis)
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return _T_DEFAULT.format(question, answer, hypothesis)
    if qtype == "temporal-reasoning":
        return _T_TEMPORAL.format(question, answer, hypothesis)
    if qtype == "knowledge-update":
        return _T_KU.format(question, answer, hypothesis)
    if qtype == "single-session-preference":
        return _T_PREFERENCE.format(question, answer, hypothesis)
    raise NotImplementedError(f"unsupported question_type: {qtype!r}")


# ── Judge plumbing ────────────────────────────────────────────────────────


def _build_judge_client(judge_model: str) -> tuple[AsyncOpenAI, str]:
    """Pick OpenAI direct vs OpenRouter based on env. Returns (client, real_model)."""
    if os.environ.get("OPENAI_API_KEY"):
        client = AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])
        # Strip any provider prefix the user passed.
        if judge_model.startswith("openai/"):
            judge_model = judge_model.split("/", 1)[1]
        return client, judge_model
    if os.environ.get("OPENROUTER_API_KEY"):
        client = AsyncOpenAI(
            api_key=os.environ["OPENROUTER_API_KEY"],
            base_url="https://openrouter.ai/api/v1",
        )
        # OpenRouter wants `openai/gpt-4o-2024-08-06`; strip Attestor's
        # `openrouter/` prefix if present.
        if judge_model.startswith("openrouter/"):
            judge_model = judge_model[len("openrouter/"):]
        return client, judge_model
    raise SystemExit("Neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set.")


async def _judge_one(
    client: AsyncOpenAI,
    model: str,
    qtype: str,
    question: str,
    answer: str,
    hypothesis: str,
    abstention: bool,
    *,
    max_retries: int = 5,
) -> dict[str, Any]:
    prompt = build_judge_prompt(qtype, question, answer, hypothesis, abstention)
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                # 16 is the minimum the Responses API accepts (gpt-4.1+ /
                # gpt-5.x reject <16). Upstream LME-S uses 10 with gpt-4o,
                # but the verdict is "yes"/"no" so the extra budget is a
                # no-op for the legacy model and a hard requirement for
                # the newer ones.
                max_tokens=16,
                n=1,
                timeout=60,
            )
            text = (resp.choices[0].message.content or "").strip()
            return {"text": text, "yes": "yes" in text.lower(), "ok": True}
        except Exception as e:  # noqa: BLE001
            last_err = e
            await asyncio.sleep(min(2 ** attempt, 30))
    return {"text": "", "yes": False, "ok": False, "error": repr(last_err)}


# ── Main flow per category ────────────────────────────────────────────────


def _load_submission(jsonl: Path) -> list[dict[str, Any]]:
    rows = []
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_reference(ref_path: Path) -> dict[str, dict[str, Any]]:
    with ref_path.open() as f:
        refs = json.load(f)
    return {r["question_id"]: r for r in refs}


async def _judge_category(
    submission_jsonl: Path,
    category: str,
    judge_model: str,
    qid2ref: dict[str, dict[str, Any]],
    parallel: int,
) -> list[dict[str, Any]]:
    hyps = _load_submission(submission_jsonl)
    expected = CATEGORY_QUESTION_COUNTS[category]
    if len(hyps) != expected:
        log.warning(
            "category=%s: submission has %d rows, expected %d",
            category, len(hyps), expected,
        )
    log.info("category=%s: judging %d rows (parallel=%d)…",
             category, len(hyps), parallel)

    client, real_model = _build_judge_client(judge_model)
    sem = asyncio.Semaphore(parallel)
    rows: list[dict[str, Any]] = [None] * len(hyps)  # type: ignore[list-item]
    done_count = 0
    t0 = time.monotonic()

    async def _one(idx: int, h: dict[str, Any]) -> None:
        nonlocal done_count
        qid = h["question_id"]
        ref = qid2ref.get(qid)
        if ref is None:
            log.warning("qid=%s not in reference data — skipping", qid)
            return
        # Strip `_abs` suffix to find the base qtype if needed; reference
        # data uses the unsuffixed qid as the source of truth for qtype.
        ref_for_type = ref
        if ref_for_type["question_type"] not in CATEGORIES + ("single-session-assistant", "single-session-preference"):
            log.warning("qid=%s has odd question_type=%s", qid, ref_for_type["question_type"])

        async with sem:
            verdict = await _judge_one(
                client, real_model,
                ref["question_type"], ref["question"], str(ref["answer"]), h["hypothesis"],
                abstention=("_abs" in qid),
            )
        rows[idx] = {
            "_qid": qid,
            "input": {"question_id": qid, "question": ref["question"]},
            "output": h["hypothesis"],
            "expected": str(ref["answer"]),
            "metadata": {
                "question_id": qid,
                "question_type": ref["question_type"],
                "abstention": "_abs" in qid,
                "judge_model": real_model,
                "judge_text": verdict.get("text"),
                "judge_ok": verdict.get("ok", True),
                "judge_error": verdict.get("error"),
            },
            "score": 1.0 if verdict["yes"] else 0.0,
        }
        done_count += 1
        if done_count % 25 == 0 or done_count == len(hyps):
            dt = time.monotonic() - t0
            log.info("  %d/%d (%.1fs, %.2fs/sample)",
                     done_count, len(hyps), dt, dt / max(done_count, 1))

    await asyncio.gather(*(_one(i, h) for i, h in enumerate(hyps)))
    return [r for r in rows if r is not None]


def _upload_to_braintrust(
    rows: list[dict[str, Any]],
    category: str,
    experiment_suffix: str,
    judge_model: str,
    submission_jsonl: Path,
) -> None:
    short = {
        "temporal-reasoning":   "temporal",
        "multi-session":        "multi-session",
        "knowledge-update":     "knowledge-update",
        "single-session-user":  "single-session-user",
    }[category]
    experiment_name = f"lme-{short}:{experiment_suffix}"

    correct = sum(1 for r in rows if r["score"] >= 0.5)
    judge_failures = sum(1 for r in rows if not r["metadata"].get("judge_ok", True))
    accuracy = correct / max(len(rows), 1)
    log.info(
        "category=%s: accuracy %.4f (%d/%d correct, %d judge failures)",
        category, accuracy, correct, len(rows), judge_failures,
    )

    # Pre-build qid → row for O(1) task lookup inside Eval (mirrors the
    # pattern in evals/braintrust_longmemeval.py for parity).
    rows_by_qid = {r["_qid"]: r for r in rows}

    def _scorer(input, output, expected, metadata, _rows=rows_by_qid):
        r = _rows.get(input.get("question_id"))
        return {
            "name": f"lme_judge:{judge_model.replace('/', '_')}",
            "score": float(r["score"]) if r else 0.0,
            "metadata": {
                "judge_text": (r or {}).get("metadata", {}).get("judge_text"),
                "judge_ok": (r or {}).get("metadata", {}).get("judge_ok"),
                "abstention": (r or {}).get("metadata", {}).get("abstention"),
            },
        }

    Eval(
        PROJECT,
        data=lambda: [
            {"input": r["input"], "expected": r["expected"], "metadata": r["metadata"]}
            for r in rows
        ],
        task=lambda row_input: rows_by_qid[row_input["question_id"]]["output"],
        scores=[_scorer],
        experiment_name=experiment_name,
        metadata={
            "category": category,
            "judge_model": judge_model,
            "submission_path": str(submission_jsonl),
            "samples": len(rows),
            "judge_failures": judge_failures,
            "accuracy": accuracy,
            "answerer_model": "openrouter/openai/gpt-4o-2024-08-06",
            "embedder": "voyage-4",
            "scope": "lme-s-option-b",
        },
    )
    log.info("uploaded experiment %s/%s (%d rows)", PROJECT, experiment_name, len(rows))


# ── CLI ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score LME-S submission JSONLs via the official judge protocol, log to Braintrust.",
    )
    parser.add_argument(
        "--submission-dir", type=Path, required=True,
        help="Directory holding lme_s_<category>.jsonl files (4 expected).",
    )
    parser.add_argument(
        "--ref-path", type=Path, default=DEFAULT_REF_PATH,
        help=f"Path to longmemeval_s_cleaned.json (default: {DEFAULT_REF_PATH}).",
    )
    parser.add_argument(
        "--suffix", default="submission",
        help="Experiment-name suffix (lme-<cat>:<suffix>).",
    )
    parser.add_argument(
        "--judge-model", default="openrouter/openai/gpt-4o-2024-08-06",
        help="Judge model. With OPENAI_API_KEY set, prefix is stripped and called direct; "
             "otherwise routed through OpenRouter.",
    )
    parser.add_argument(
        "--parallel", type=int, default=4,
        help="Concurrent judge calls (default 4; OpenRouter handles 4 comfortably).",
    )
    parser.add_argument(
        "--categories", nargs="+", choices=list(CATEGORIES), default=list(CATEGORIES),
        help="Categories to judge (default: all 4).",
    )
    parser.add_argument(
        "--upload-only", action="store_true",
        help=(
            "Skip the LLM-judge step. For each category, read the previously "
            "saved <submission_dir>/lme_s_<category>.judge-rows.json and "
            "upload to Braintrust. Useful after a Braintrust outage or when "
            "moving experiments to a different project."
        ),
    )
    parser.add_argument(
        "--upload-datasets", action="store_true",
        help=(
            "Before judging, upload LME-S source samples (questions + answers) "
            "as Braintrust Datasets (lme-<category>-v1). One-time per project."
        ),
    )
    args = parser.parse_args()

    if not os.environ.get("BRAINTRUST_API_KEY"):
        raise SystemExit("BRAINTRUST_API_KEY not set. Run: set -a && source .env && set +a")
    if not args.ref_path.exists():
        raise SystemExit(f"reference file not found: {args.ref_path}")

    if args.upload_datasets:
        # Reuse the dataset uploader from the live-bench harness so naming
        # stays single-source-of-truth and per-category sample counts are
        # validated against the cache. The sibling file lives in this same
        # directory; add it to sys.path so the import works whether the
        # script is invoked as ``python evals/braintrust_lme_s_score.py``
        # or as a module.
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from braintrust_longmemeval import upload_dataset  # type: ignore
        for cat in args.categories:
            log.info("uploading dataset for category=%s …", cat)
            upload_dataset(cat)

    log.info("loading reference QA from %s", args.ref_path)
    qid2ref = _load_reference(args.ref_path)
    log.info("loaded %d reference questions", len(qid2ref))

    overall: dict[str, dict[str, Any]] = {}
    upload_failures: list[tuple[str, str]] = []
    for cat in args.categories:
        jsonl = args.submission_dir / f"lme_s_{cat}.jsonl"
        rows_path = args.submission_dir / f"lme_s_{cat}.judge-rows.json"

        if args.upload_only:
            if not rows_path.exists():
                log.warning("--upload-only set but %s missing for %s — skipping",
                            rows_path, cat)
                continue
            with rows_path.open() as f:
                payload = json.load(f)
            rows = payload["rows"]
            log.info("category=%s: loaded %d rows from %s (no LLM)",
                     cat, len(rows), rows_path)
        else:
            if not jsonl.exists():
                log.warning("missing submission JSONL for %s: %s — skipping", cat, jsonl)
                continue
            rows = asyncio.run(_judge_category(
                jsonl, cat, args.judge_model, qid2ref, args.parallel
            ))
            # Checkpoint to disk BEFORE upload so an upload failure doesn't
            # discard $-spent LLM judgments. Re-runnable via --upload-only.
            rows_path.write_text(json.dumps({
                "category": cat,
                "judge_model": args.judge_model,
                "suffix": args.suffix,
                "rows": rows,
            }, indent=2))
            log.info("checkpointed %d rows to %s", len(rows), rows_path)

        try:
            _upload_to_braintrust(rows, cat, args.suffix, args.judge_model, jsonl)
        except Exception as e:  # noqa: BLE001
            log.error("upload failed for category=%s: %r — rows preserved at %s",
                      cat, e, rows_path)
            upload_failures.append((cat, repr(e)))

        correct = sum(1 for r in rows if r["score"] >= 0.5)
        overall[cat] = {
            "accuracy": correct / max(len(rows), 1),
            "correct": correct,
            "total": len(rows),
            "judge_failures": sum(1 for r in rows if not r["metadata"].get("judge_ok", True)),
        }

    log.info("=" * 60)
    log.info("FINAL — submission=%s suffix=%s judge=%s",
             args.submission_dir.name, args.suffix, args.judge_model)
    for cat, stats in overall.items():
        log.info(
            "  %-22s %.4f  (%3d/%3d correct, %d judge errors)",
            cat, stats["accuracy"], stats["correct"], stats["total"], stats["judge_failures"],
        )


if __name__ == "__main__":
    main()
