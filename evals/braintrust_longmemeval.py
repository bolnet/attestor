"""Braintrust eval: attestor recall pipeline on LongMemEval-S.

Per ``feedback_eval_scope_lme_s_only`` and ``feedback_lme_s_four_category_split``,
this harness is LME-S only and runs each category as its OWN experiment with
its OWN dataset:

  - lme-temporal-v1            (133 questions, temporal-reasoning)
  - lme-multi-session-v1       (133 questions, multi-session)
  - lme-knowledge-update-v1    ( 78 questions, knowledge-update)
  - lme-single-session-user-v1 ( 70 questions, single-session-user)

LME-S produces 414 of 500 questions in scope (Option B locked in 2026-05-02).

Two run modes:

  1. ``--from-json PATH`` (no LLM cost, instant) — parse a persisted
     ``LMERunReport`` JSON from a prior run (e.g. logs/*.json) and upload
     it as a Braintrust experiment. Use to backfill historical bench data
     or to upload a run produced by another harness.

  2. live mode (default) — call ``attestor.longmemeval.run_async`` directly,
     pre-compute predictions + judgments, then upload. Mirrors the
     ``braintrust_locomo.py`` pattern. Costs apply per the configured
     answerer + judge models.

Run examples:

    set -a && source .env && set +a

    # Backfill today's golden-config bench from logs (free, instant):
    .venv/bin/python evals/braintrust_longmemeval.py \\
        --category temporal-reasoning \\
        --from-json logs/lme_temporal_10q_voyage_baseline_smoke.json \\
        --suffix golden-2026-05-02

    # Live 10q run from scratch:
    .venv/bin/python evals/braintrust_longmemeval.py \\
        --category temporal-reasoning \\
        --max-samples 10 \\
        --suffix smoke
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname).1s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("eval.lme")
logging.getLogger("attestor").setLevel(logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

from braintrust import Eval, init_dataset

from attestor import AgentMemory
from attestor.config import build_backend_config, get_stack, reset_stack, set_stack
from attestor.longmemeval import load_or_download, run_async


# ────────────────────────────────────────────────────────────────────────────
# Config — single source of truth for project + dataset names
# ────────────────────────────────────────────────────────────────────────────

PROJECT = "attestor-lme-s"

# Dataset name per LME-S category. Locked to Option B per
# ``feedback_lme_s_four_category_split`` — single-session-assistant and
# single-session-preference are explicitly out of scope.
DATASETS: dict[str, str] = {
    "temporal-reasoning":   "lme-temporal-v1",
    "multi-session":        "lme-multi-session-v1",
    "knowledge-update":     "lme-knowledge-update-v1",
    "single-session-user":  "lme-single-session-user-v1",
}

CATEGORY_QUESTION_COUNTS: dict[str, int] = {
    "temporal-reasoning":   133,
    "multi-session":        133,
    "knowledge-update":      78,
    "single-session-user":   70,
}


# ────────────────────────────────────────────────────────────────────────────
# Dataset upload (one-time per category)
# ────────────────────────────────────────────────────────────────────────────

def upload_dataset(category: str) -> None:
    """One-shot upload of LME-S samples for ``category`` to Braintrust.

    Idempotent — Braintrust handles row uniqueness by content hash. Run once
    per category before launching experiments; re-running is safe.
    """
    if category not in DATASETS:
        raise ValueError(
            f"Unknown LME-S category: {category!r}. "
            f"In-scope categories: {sorted(DATASETS)}"
        )

    log.info("loading LME-S samples for category=%s …", category)
    all_samples = list(load_or_download(variant="s"))
    samples = [s for s in all_samples if s.question_type == category]
    expected = CATEGORY_QUESTION_COUNTS[category]
    if len(samples) != expected:
        raise RuntimeError(
            f"Expected {expected} samples for {category}, got {len(samples)}. "
            "Did the upstream LongMemEval dataset change? Confirm cache "
            "freshness at ~/.cache/attestor/longmemeval/."
        )

    dataset_name = DATASETS[category]
    log.info("uploading %d samples to Braintrust dataset %s/%s …",
             len(samples), PROJECT, dataset_name)
    ds = init_dataset(project=PROJECT, name=dataset_name)
    inserted = 0
    for s in samples:
        ds.insert(
            input={
                "question": s.question,
                "question_id": s.question_id,
                "question_date": s.question_date,
            },
            expected=str(s.answer),
            metadata={
                "question_id": s.question_id,
                "question_type": s.question_type,
                "answer_session_ids": list(s.answer_session_ids),
                "haystack_session_count": len(s.haystack_sessions),
            },
        )
        inserted += 1
    log.info("dataset upload complete: %d rows in %s/%s", inserted, PROJECT, dataset_name)


# ────────────────────────────────────────────────────────────────────────────
# Backfill mode — upload an already-run LMERunReport JSON to Braintrust
# ────────────────────────────────────────────────────────────────────────────

def upload_from_json(json_path: Path, category: str, experiment_suffix: str) -> None:
    """Upload a persisted ``LMERunReport`` JSON as a Braintrust experiment.

    Used to backfill experiments without re-running the bench. Pulls per-sample
    judgments verbatim from the JSON — no judge LLM is invoked.
    """
    if category not in DATASETS:
        raise ValueError(f"Unknown category: {category!r}")
    log.info("loading persisted bench JSON: %s", json_path)
    with json_path.open() as f:
        report = json.load(f)

    # Validate the JSON corresponds to the requested category.
    sample_categories = {s.get("category") for s in report.get("samples", [])}
    if not sample_categories.issubset({category}):
        raise RuntimeError(
            f"JSON contains samples from categories {sample_categories}, "
            f"but --category was {category!r}. Mismatch."
        )

    rows = []
    for s in report["samples"]:
        rows.append(_sample_to_braintrust_row(s, report))

    metadata = _build_experiment_metadata(report, category, json_path)
    judge_names = list(report.get("judge_models") or [])
    experiment_name = _experiment_name(category, experiment_suffix)

    log.info("uploading %d rows as experiment %s/%s …",
             len(rows), PROJECT, experiment_name)
    Eval(
        PROJECT,
        data=lambda: [
            {"input": r["input"], "expected": r["expected"], "metadata": r["metadata"]}
            for r in rows
        ],
        task=lambda row_input: _row_lookup(rows, row_input)["output"],
        scores=_build_judge_scorers(judge_names, rows),
        experiment_name=experiment_name,
        metadata=metadata,
    )
    log.info("experiment uploaded: %s/%s (%d rows)", PROJECT, experiment_name, len(rows))


# ────────────────────────────────────────────────────────────────────────────
# Live mode — run from scratch then upload
# ────────────────────────────────────────────────────────────────────────────

def run_live(
    category: str,
    max_samples: int | None,
    experiment_suffix: str,
    *,
    embedder_override: dict[str, Any] | None,
    self_consistency_override: dict[str, Any] | None,
) -> None:
    """Execute a fresh bench run and upload as a Braintrust experiment.

    ``embedder_override`` and ``self_consistency_override`` flip stack fields
    for this run only via ``set_stack()``. YAML is never modified.
    """
    if category not in DATASETS:
        raise ValueError(f"Unknown category: {category!r}")

    # Override stack if requested
    base = get_stack()
    overrides: dict[str, Any] = {}
    if embedder_override:
        overrides["embedder"] = replace(base.embedder, **embedder_override)
    if self_consistency_override:
        overrides["self_consistency"] = replace(
            base.self_consistency, **self_consistency_override
        )
    if overrides:
        set_stack(replace(base, **overrides))
        log.info("stack overrides applied: %s", sorted(overrides))

    try:
        log.info("loading LME-S samples (category=%s) …", category)
        all_samples = list(load_or_download(variant="s"))
        samples = [s for s in all_samples if s.question_type == category]
        if max_samples is not None:
            samples = samples[:max_samples]
        log.info("running %d / %d samples for category=%s",
                 len(samples), CATEGORY_QUESTION_COUNTS[category], category)

        backend_config = build_backend_config(get_stack())
        backend_config["backend_configs"]["postgres"]["skip_schema_init"] = False

        store_path = f"lme-{category}-{int(time.time())}"

        def mem_factory() -> AgentMemory:
            return AgentMemory(store_path, config=backend_config)

        log_path = Path(f"logs/lme_{category}_{experiment_suffix}.json")
        log_path.parent.mkdir(parents=True, exist_ok=True)

        report = asyncio.run(run_async(
            samples=samples,
            mem_factory=mem_factory,
            parallel=1,
            verbose=True,
            output_path=log_path,
        ))
    finally:
        reset_stack()

    # Persisted JSON exists at log_path; reuse the backfill path for upload.
    upload_from_json(log_path, category, experiment_suffix)


# ────────────────────────────────────────────────────────────────────────────
# Helpers — shared between live and backfill modes
# ────────────────────────────────────────────────────────────────────────────

def _experiment_name(category: str, suffix: str) -> str:
    """Stable experiment name: ``lme-<category>:<suffix>``."""
    short = {
        "temporal-reasoning":   "temporal",
        "multi-session":        "multi-session",
        "knowledge-update":     "knowledge-update",
        "single-session-user":  "single-session-user",
    }[category]
    return f"lme-{short}:{suffix}"


def _sample_to_braintrust_row(sample: dict[str, Any], report: dict[str, Any]) -> dict[str, Any]:
    """Convert one persisted sample dict into a Braintrust-ready row."""
    qid = sample["question_id"]
    return {
        "input": {
            "question": sample["question"],
            "question_id": qid,
        },
        "expected": str(sample["gold"]),
        "output": str(sample.get("answer") or ""),
        "metadata": {
            "question_id": qid,
            "category": sample.get("category"),
            "ingest_turns": sample.get("ingest_turns"),
            "ingest_memories": sample.get("ingest_memories"),
            "retrieved_count": sample.get("retrieved_count"),
            "retrieval_hit": sample.get("retrieval_hit"),
            "retrieval_overlap": sample.get("retrieval_overlap"),
            "answer_latency_ms": sample.get("answer_latency_ms"),
            "predicted_mode": sample.get("predicted_mode"),
            "judgments": sample.get("judgments", {}),
        },
        "_qid": qid,
    }


def _row_lookup(rows: list[dict[str, Any]], row_input: dict[str, Any]) -> dict[str, Any]:
    qid = row_input.get("question_id")
    for r in rows:
        if r["_qid"] == qid:
            return r
    raise KeyError(f"no precomputed row for question_id={qid!r}")


def _build_judge_scorers(judge_names: list[str], rows: list[dict[str, Any]]) -> list:
    """Return one Braintrust scorer per judge model.

    Each scorer reads the persisted ``judgments[judge].correct`` flag from the
    row's metadata. Score = 1.0 if correct, 0.0 otherwise. No live LLM call.
    """
    scorers = []
    for judge in judge_names:
        # Capture ``judge`` by default-arg trick (closure-over-loop hazard).
        def _scorer(input, output, expected, metadata, _j=judge, _rows=rows):
            verdict = (metadata.get("judgments") or {}).get(_j) or {}
            correct = bool(verdict.get("correct"))
            return {
                "name": f"judge:{_safe_name(_j)}",
                "score": 1.0 if correct else 0.0,
                "metadata": {
                    "judge_label": verdict.get("label"),
                    "judge_reasoning": verdict.get("reasoning"),
                },
            }
        scorers.append(_scorer)

    # Composite: any-judge-correct + all-judges-correct
    def _any_correct(input, output, expected, metadata, _judges=tuple(judge_names)):
        verdicts = metadata.get("judgments") or {}
        any_ok = any(bool((verdicts.get(j) or {}).get("correct")) for j in _judges)
        return {"name": "any_judge_correct", "score": 1.0 if any_ok else 0.0}

    def _all_correct(input, output, expected, metadata, _judges=tuple(judge_names)):
        verdicts = metadata.get("judgments") or {}
        all_ok = all(bool((verdicts.get(j) or {}).get("correct")) for j in _judges) if _judges else False
        return {"name": "all_judges_correct", "score": 1.0 if all_ok else 0.0}

    if len(judge_names) > 1:
        scorers.extend([_any_correct, _all_correct])
    return scorers


def _safe_name(model: str) -> str:
    """Sanitize a model id for Braintrust scorer name (no slashes)."""
    return model.replace("/", "_").replace(":", "_")


def _build_experiment_metadata(
    report: dict[str, Any], category: str, source_path: Path | None
) -> dict[str, Any]:
    """Pack run-level metadata for the experiment surface in Braintrust UI."""
    rc = report.get("run_config") or {}
    prov = report.get("provenance") or {}
    return {
        "category": category,
        "answer_model": report.get("answer_model"),
        "judge_models": report.get("judge_models"),
        "total_samples": report.get("total"),
        "by_category": report.get("by_category"),
        "by_judge": report.get("by_judge"),
        "started_at": report.get("started_at"),
        "completed_at": report.get("completed_at"),
        "schema_version": report.get("schema_version"),
        "git_sha": prov.get("git_sha"),
        "attestor_version": prov.get("attestor_version"),
        "run_config": rc,
        "source_json": str(source_path) if source_path else None,
    }


# ────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Braintrust eval for LongMemEval-S (one category at a time).",
    )
    parser.add_argument(
        "--category",
        required=True,
        choices=sorted(DATASETS.keys()),
        help="LME-S question category (one experiment per category).",
    )
    parser.add_argument(
        "--upload-dataset",
        action="store_true",
        help=(
            "One-shot: upload all LME-S samples for --category to Braintrust "
            "as the versioned dataset, then exit. Run once per category."
        ),
    )
    parser.add_argument(
        "--from-json",
        type=Path,
        default=None,
        help=(
            "Backfill mode: parse a persisted LMERunReport JSON and upload "
            "as an experiment. No LLM calls are made."
        ),
    )
    parser.add_argument("--max-samples", type=int, default=10,
                        help="Live mode: cap on samples per category (default 10).")
    parser.add_argument("--suffix", default="smoke",
                        help="Experiment name suffix (e.g. smoke, golden, ablation:foo).")
    parser.add_argument("--embedder-provider", default=None,
                        help="Live mode override: stack.embedder.provider.")
    parser.add_argument("--embedder-model", default=None,
                        help="Live mode override: stack.embedder.model.")
    parser.add_argument("--self-consistency", action="store_true",
                        help="Live mode override: enable self_consistency (k=5, judge_pick).")
    args = parser.parse_args()

    if not os.environ.get("BRAINTRUST_API_KEY"):
        raise SystemExit(
            "BRAINTRUST_API_KEY not set. Run: set -a && source .env && set +a"
        )

    if args.upload_dataset:
        upload_dataset(args.category)
        return

    if args.from_json:
        if not args.from_json.exists():
            raise SystemExit(f"--from-json path does not exist: {args.from_json}")
        upload_from_json(args.from_json, args.category, args.suffix)
        return

    # Live mode — needs answer + judge model API keys.
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise SystemExit("OPENROUTER_API_KEY not set (live mode needs answerer).")

    embedder_override: dict[str, Any] | None = None
    if args.embedder_provider or args.embedder_model:
        embedder_override = {}
        if args.embedder_provider:
            embedder_override["provider"] = args.embedder_provider
        if args.embedder_model:
            embedder_override["model"] = args.embedder_model

    sc_override: dict[str, Any] | None = None
    if args.self_consistency:
        sc_override = {"enabled": True, "k": 5, "voter": "judge_pick", "temperature": 0.7}

    run_live(
        args.category,
        args.max_samples,
        args.suffix,
        embedder_override=embedder_override,
        self_consistency_override=sc_override,
    )


if __name__ == "__main__":
    main()
