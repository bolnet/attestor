"""Benchmark CLI commands: ``locomo``, ``mab``, ``longmemeval``."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attestor.cli._common import _load_env_file, _parse_backend_config
from attestor.core import AgentMemory


def _cmd_locomo(args: argparse.Namespace) -> None:
    # Load env file if provided. Provider keys are resolved by the LLM
    # pool from ``configs/attestor.yaml`` (``llm.providers.*.api_key_env``)
    # at first use — we don't read or pre-validate any specific env var
    # here.
    if args.env_file:
        _load_env_file(args.env_file)

    from attestor.config import get_stack
    from attestor.locomo import run_locomo, print_locomo

    stack = get_stack()
    judge_model = args.judge_model or stack.models.judge
    answer_model = args.answer_model or stack.models.answerer
    extraction_model = args.extraction_model or stack.models.extraction

    print("Running LOCOMO benchmark...")
    print("(Long Conversation Memory -- industry-standard benchmark)\n")

    results = run_locomo(
        data_path=args.data,
        judge_model=judge_model,
        answer_model=answer_model,
        extraction_model=extraction_model,
        use_extraction=args.use_extraction,
        resolve_pronouns=args.resolve_pronouns,
        max_conversations=args.max_conversations,
        max_questions_per_conv=args.max_questions,
        recall_budget=args.budget,
        verbose=args.verbose,
        backend_config=_parse_backend_config(args),
    )

    print_locomo(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


def _cmd_mab(args: argparse.Namespace) -> None:
    # Load env file if provided. Provider keys are resolved by the LLM
    # pool from ``configs/attestor.yaml`` (``llm.providers.*.api_key_env``)
    # at first use — we don't read or pre-validate any specific env var
    # here.
    if args.env_file:
        _load_env_file(args.env_file)

    from attestor.mab import run_mab, print_mab

    print("Running MemoryAgentBench benchmark...")
    print("(ICLR 2026 benchmark -- Accurate Retrieval, Conflict Resolution, etc.)\n")

    results = run_mab(
        categories=args.categories,
        max_examples=args.max_examples,
        max_questions=args.max_questions,
        chunk_size=args.chunk_size,
        context_max_tokens=args.context_max_tokens,
        answer_model=args.answer_model,
        recall_budget=args.budget,
        verbose=args.verbose,
        skip_examples=args.skip_examples,
        backend_config=_parse_backend_config(args),
    )

    print_mab(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


def _cmd_longmemeval(args: argparse.Namespace) -> None:
    """Run the LongMemEval benchmark against Attestor."""
    # Load env file if provided. Provider keys are resolved by the LLM
    # pool from ``configs/attestor.yaml`` (``llm.providers.*.api_key_env``)
    # at first use — we don't pre-read any specific env var here. If the
    # configured provider's key is missing the pool raises with a clear
    # message at first chat call.
    if args.env_file:
        _load_env_file(args.env_file)

    from attestor.longmemeval import (
        load_longmemeval,
        load_or_download,
        run,
    )

    # Data source: fixture > --data > auto-download
    dataset_path: Path | None = None
    if args.fixture:
        fixture_path = Path(__file__).parent.parent.parent.parent / "tests" / "fixtures" / "lme_mini.json"
        if not fixture_path.exists():
            print(f"ERROR: bundled fixture missing at {fixture_path}", file=sys.stderr)
            sys.exit(2)
        samples = load_longmemeval(fixture_path)
        dataset_path = fixture_path
        print(f"[fixture] loaded {len(samples)} samples from {fixture_path}")
    elif args.data:
        samples = load_longmemeval(args.data)
        dataset_path = Path(args.data)
        print(f"loaded {len(samples)} samples from {args.data}")
    else:
        samples = load_or_download(variant=args.variant)
        print(f"loaded {len(samples)} samples (variant={args.variant})")

    if args.categories:
        allowed = set(args.categories)
        samples = [s for s in samples if s.question_type in allowed]
        print(f"filtered to {len(samples)} samples matching {sorted(allowed)}")

    if args.max_samples is not None:
        samples = samples[: args.max_samples]
        print(f"capped to {len(samples)} samples")

    from attestor.config import get_stack

    stack = get_stack()
    # Judge panel: CLI flag > YAML (judge + verifier) > legacy DEFAULT_JUDGES
    judge_models = args.judge_model or [stack.models.judge, stack.models.verifier]
    answer_model = args.answer_model or stack.models.answerer
    distill_model = args.distill_model or stack.models.distill
    verify_model = args.verify_model or stack.models.verifier

    from attestor._paths import resolve_store_path
    import threading

    backend_config = _parse_backend_config(args)
    store_path = resolve_store_path(getattr(args, "path", None))

    # Pre-warm so the first factory call doesn't race with others.
    # AgentMemory.__init__ reads + writes ~/.attestor/config.json every time,
    # which is a write-truncate race under concurrent construction. The
    # lock makes construction atomic without serializing the (long) ingest
    # / answer / judge work that follows.
    _factory_lock = threading.Lock()
    AgentMemory(store_path, config=backend_config).close()

    def mem_factory() -> AgentMemory:
        """Fresh AgentMemory per sample — per-task Postgres/Neo4j connections."""
        with _factory_lock:
            return AgentMemory(store_path, config=backend_config)

    print(
        f"Running LongMemEval: answer={answer_model} "
        f"judges={judge_models} samples={len(samples)} budget={args.budget} "
        f"parallel={args.parallel}"
    )
    report = run(
        samples,
        mem_factory=mem_factory,
        answer_model=answer_model,
        judge_models=judge_models,
        budget=args.budget,
        use_extraction=args.use_extraction,
        use_distillation=args.use_distillation,
        distill_model=distill_model,
        max_facts=args.max_facts,
        parallel=args.parallel,
        verify=args.verify,
        verify_model=verify_model,
        verbose=args.verbose,
        output_path=args.output,
        dataset_path=dataset_path,
    )

    # Pretty print summary
    print("\n=== LongMemEval summary ===")
    print(f"total samples: {report.total}")
    for jm, bucket in report.by_judge.items():
        if jm.startswith("_"):
            continue  # skip meta entries like _inter_judge_agreement
        print(f"  judge={jm}: {bucket['correct']}/{bucket['total']} ({bucket['accuracy']}%)")

    agreement = report.by_judge.get("_inter_judge_agreement")
    if agreement:
        print("\n  inter-judge agreement:")
        for pair, stats in agreement.items():
            print(
                f"    {pair}: agreement={stats['agreement_pct']}% "
                f"(both_correct={stats['both_correct']}, both_wrong={stats['both_wrong']})"
            )

    dim = getattr(report, "by_dimension", {}) or {}
    if dim:
        print("\n  Dimension B (multi-dimensional scoring):")
        retr = dim.get("retrieval", {})
        if retr.get("total"):
            print(
                f"    retrieval precision: {retr['hits']}/{retr['total']} ({retr['precision']}%)"
            )
        mode = dim.get("mode_distribution", {})
        if mode:
            counts = mode.get("counts", {})
            print(
                f"    mode distribution: fact={counts.get('fact', 0)} "
                f"recommendation={counts.get('recommendation', 0)} "
                f"unknown={counts.get('unknown', 0)}"
            )
        pers = dim.get("personalization", {})
        if pers.get("total"):
            print(
                f"    personalization (recommendation samples only): "
                f"{pers['correct']}/{pers['total']} ({pers['accuracy']}%)"
            )
        per_mode = dim.get("by_predicted_mode", {})
        if per_mode:
            print("    answer accuracy by predicted mode (judge A only):")
            for m, b in per_mode.items():
                if b.get("total", 0) > 0:
                    print(f"      {m}: {b['correct']}/{b['total']} ({b['accuracy']}%)")

    print("\n  by category (per judge):")
    for cat, per_judge in report.by_category.items():
        print(f"    {cat}:")
        for jm, bucket in per_judge.items():
            print(f"      {jm}: {bucket['correct']}/{bucket['total']} ({bucket['accuracy']}%)")

    if args.output:
        print(f"\nFull report written to {args.output}")
