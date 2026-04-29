"""Local LongMemEval smoke — config-driven.

Stack discipline:
    DB / embedder / model choices come from ``configs/attestor.yaml`` via
    ``attestor.config.get_stack()``. The default canonical stack is
    PG+pgvector+Neo4j+Voyage with the four-role gpt-4.1 + claude-sonnet-4-6
    model lineup. Override the whole stack with ``--config <path>`` or
    individual values with the CLI flags below.

Why this exists:
    LongMemEval ``oracle`` variant is the cheapest LME run we can do for
    a sanity-check after retrieval-pipeline edits. The smoke applies a
    fresh schema, ingests N samples, runs answer + dual-LLM judge, and
    prints by-judge / by-category breakdowns.

Usage:
    set -a && source .env && set +a
    .venv/bin/python scripts/lme_smoke_local.py --n 2 --yes
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from dataclasses import replace as dc_replace
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from attestor.config import (  # noqa: E402
    DEFAULT_CONFIG,
    StackConfig,
    build_backend_config,
    configure_embedder,
    confirm_or_exit,
    load_stack,
)

SCHEMA_PATH = ROOT / "attestor" / "store" / "schema.sql"
RUN_LABEL = "smoke"


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="lme_smoke_local",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config",
        default=str(DEFAULT_CONFIG),
        help=f"Stack config YAML (default: {DEFAULT_CONFIG.relative_to(ROOT)})",
    )
    p.add_argument(
        "--embedder",
        choices=["voyage", "openai"],
        default=None,
        help="Override embedder.provider from the config file",
    )
    p.add_argument(
        "--no-distill",
        action="store_true",
        help="Disable distillation (default is on)",
    )
    p.add_argument(
        "--variant",
        default="oracle",
        help="LME variant (oracle|s|m); oracle is smallest / fastest",
    )
    p.add_argument(
        "--n",
        type=int,
        default=2,
        help="Number of samples to run (ignored when --sample-ids is given)",
    )
    p.add_argument(
        "--sample-ids",
        default="",
        help="Comma-separated sample IDs to run; overrides --n.",
    )
    p.add_argument(
        "--skip-schema",
        action="store_true",
        help="Skip schema reapply (faster repeat runs)",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Don't prompt to confirm the resolved stack",
    )
    p.add_argument("--verbose", action="store_true")
    p.add_argument(
        "--trace",
        action="store_true",
        help="Exhaustive per-stage trace (sets ATTESTOR_TRACE=1; writes "
             "JSONL to logs/lme_smoke_<ts>.jsonl)",
    )
    return p.parse_args()


def _apply_fresh_schema(pg_url: str, embedding_dim: int) -> None:
    """Drop + recreate v4 tables so each smoke starts clean."""
    import psycopg2

    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", str(embedding_dim))
    with psycopg2.connect(pg_url) as c, c.cursor() as cur:
        for tbl in (
            "deletion_audit",
            "user_quotas",
            "memories",
            "episodes",
            "sessions",
            "projects",
            "users",
        ):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
        c.commit()
    print(f"[{RUN_LABEL}] schema applied (embedding_dim={embedding_dim})")


def _wipe_neo4j(stack: StackConfig) -> None:
    from neo4j import GraphDatabase

    with GraphDatabase.driver(
        stack.neo4j.url, auth=(stack.neo4j.username, stack.neo4j.password)
    ) as drv, drv.session(database=stack.neo4j.database) as s:
        s.run("MATCH (n) DETACH DELETE n")


def _mem_factory(stack: StackConfig):
    """Return a zero-arg callable producing fresh AgentMemory instances —
    each LME sample gets its own SOLO user via tempdir."""
    from attestor.core import AgentMemory

    cfg = build_backend_config(stack)

    def _make():
        path = tempfile.mkdtemp(prefix="lme_smoke_")
        return AgentMemory(path, config=cfg)

    return _make


async def _run(args: argparse.Namespace, stack: StackConfig) -> None:
    from attestor.longmemeval import load_or_download, run_async

    # Pre-flight: the runtime LLM client picks its key env from the
    # resolved llm.provider in the stack. Surface a clear error here
    # rather than letting the first chat call fail with 401.
    expected_key_env = stack.llm.api_key_env
    if not os.environ.get(expected_key_env):
        sys.stderr.write(
            f"[{RUN_LABEL}] error: {expected_key_env} not set — required "
            f"for llm.provider={stack.llm.provider!r}. Either export the "
            f"env var or switch llm.provider in configs/attestor.yaml.\n"
        )
        raise SystemExit(2)

    print(f"[{RUN_LABEL}] loading LME variant={args.variant!r}…")
    samples = load_or_download(variant=args.variant)

    wanted_ids = {s.strip() for s in args.sample_ids.split(",") if s.strip()}
    if wanted_ids:
        sample_id_attr = (
            "question_id" if hasattr(samples[0], "question_id") else "id"
        )
        slice_ = [s for s in samples if getattr(s, sample_id_attr, None) in wanted_ids]
        missing = wanted_ids - {getattr(s, sample_id_attr) for s in slice_}
        if missing:
            sys.stderr.write(
                f"[{RUN_LABEL}] warning: sample_ids not found: {sorted(missing)}\n"
            )
        if not slice_:
            sys.stderr.write(
                f"[{RUN_LABEL}] error: --sample-ids matched zero samples\n"
            )
            raise SystemExit(2)
        print(
            f"[{RUN_LABEL}] dataset has {len(samples)} samples; "
            f"running {len(slice_)} matched by --sample-ids"
        )
    else:
        slice_ = samples[: args.n]
        print(
            f"[{RUN_LABEL}] dataset has {len(samples)} samples; running first {args.n}"
        )

    use_distillation = not args.no_distill

    report = await run_async(
        slice_,
        mem_factory=_mem_factory(stack),
        answer_model=stack.models.answerer,
        judge_models=[stack.models.judge, stack.models.verifier],
        distill_model=stack.models.distill,
        use_distillation=use_distillation,
        api_key=None,  # _get_client reads OPENAI_API_KEY / OPENROUTER_API_KEY
        budget=stack.budget,
        parallel=stack.parallel,
        verbose=args.verbose,
    )

    print()
    print("=" * 72)
    print(f"[{RUN_LABEL}] DONE — {report.total} samples")
    print(f"[{RUN_LABEL}] by_judge: {json.dumps(report.by_judge, indent=2)}")
    print(f"[{RUN_LABEL}] by_category: {json.dumps(report.by_category, indent=2)}")


def main() -> int:
    args = _parse_args()

    if args.trace:
        from datetime import datetime as _dt
        ts = _dt.now().strftime("%Y%m%d_%H%M%S")
        trace_path = ROOT / "logs" / f"lme_smoke_{ts}.jsonl"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        os.environ["ATTESTOR_TRACE"] = "1"
        os.environ["ATTESTOR_TRACE_FILE"] = str(trace_path)
        # Re-read env after setting it; module load happens later but the
        # smoke imports attestor.* below — make sure trace.py picks it up.
        import attestor.trace as _tr
        _tr.reset_for_test()
        print(f"[{RUN_LABEL}] trace ON — events stream to stderr; "
              f"jsonl → {trace_path}")

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    stack = load_stack(args.config)
    if args.embedder:
        stack = dc_replace(
            stack,
            embedder=dc_replace(
                stack.embedder,
                provider=args.embedder,
                model=(
                    "voyage-4" if args.embedder == "voyage" else "text-embedding-3-large"
                ),
            ),
        )

    confirm_or_exit(stack, run_label=RUN_LABEL, yes=args.yes)
    configure_embedder(stack)

    if not args.skip_schema:
        _apply_fresh_schema(stack.postgres.url, stack.embedder.dimensions)
        try:
            _wipe_neo4j(stack)
        except Exception as e:
            print(f"[{RUN_LABEL}] warning: neo4j wipe skipped — {e}")

    asyncio.run(_run(args, stack))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
