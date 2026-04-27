"""Local LongMemEval smoke — Ollama only, no cloud APIs.

Tiny driver:
  - Loads the LME 'oracle' variant (small + fast — 12 samples, ~1k tokens each).
  - Runs N samples (default 2) through ingest → answer → dual judge.
  - Answerer + judges = local Ollama via OpenAI-compatible API.
  - Embedder = local Ollama bge-m3 (auto-detected by attestor).
  - Postgres = attestor_v4_test on localhost.

Usage (from repo root):
    .venv/bin/python scripts/lme_smoke_local.py --n 2

Prereqs:
  - Ollama running with bge-m3 + qwen2.5:14b-instruct + mistral-small:24b pulled.
  - Postgres at localhost:5432, db `attestor_v4_test`, user/pass postgres/attestor.
  - Schema applied with embedding_dim=1024 (this script reapplies it on each run).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

# Force line-buffered stdout/stderr so progress shows in real time even
# when piped (`... | tee`, `... | tail -f`, CI capture). Equivalent to
# `python -u` but doesn't require remembering the flag.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover — pre-3.7 fallback path
    pass

# Make `attestor` importable when invoked as `python scripts/...`
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Force local Ollama BEFORE importing attestor.longmemeval — _get_client
# reads LME_LLM_BASE_URL at call time, but setting early is safest.
os.environ.setdefault("LME_LLM_BASE_URL", "http://localhost:11434/v1")

SCHEMA_PATH = ROOT / "attestor" / "store" / "schema.sql"

PG_URL = "postgresql://postgres:attestor@localhost:5432/attestor_v4_test"
ANSWER_MODEL = os.environ.get("ANSWER_MODEL", "qwen2.5:14b-instruct")
JUDGE_MODELS = os.environ.get(
    "JUDGE_MODELS", "qwen2.5:14b-instruct,mistral-small:24b",
).split(",")
EMBEDDING_DIM = 1024  # bge-m3


def _apply_fresh_schema() -> None:
    """Drop + recreate v4 tables so each smoke starts clean."""
    import psycopg2

    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", str(EMBEDDING_DIM))
    with psycopg2.connect(PG_URL) as c, c.cursor() as cur:
        for tbl in ("deletion_audit", "user_quotas", "memories", "episodes",
                    "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
        c.commit()
    print(f"[smoke] schema applied (embedding_dim={EMBEDDING_DIM})")


def _mem_factory():
    """Return a zero-arg callable producing fresh AgentMemory instances."""
    from attestor.core import AgentMemory

    def _make():
        # Each sample gets its own SOLO user via tempdir path (LME's
        # per-sample isolation guarantee).
        path = tempfile.mkdtemp(prefix="lme_smoke_")
        return AgentMemory(path, config={
            "mode": "solo",
            "backends": ["postgres"],
            "backend_configs": {"postgres": {
                "url": "postgresql://localhost:5432",
                "database": "attestor_v4_test",
                "auth": {"username": "postgres", "password": "attestor"},
                "v4": True,
                "skip_schema_init": True,
            }},
        })

    return _make


async def _run(n: int, variant: str, verbose: bool) -> None:
    from attestor.longmemeval import (
        load_or_download, run_async,
    )

    print(f"[smoke] loading LME variant={variant!r}…")
    samples = load_or_download(variant=variant)
    print(f"[smoke] dataset has {len(samples)} samples; running first {n}")

    samples = samples[:n]
    print(f"[smoke] answerer={ANSWER_MODEL!r} judges={JUDGE_MODELS!r}")
    print(f"[smoke] LME_LLM_BASE_URL={os.environ['LME_LLM_BASE_URL']!r}")

    report = await run_async(
        samples,
        mem_factory=_mem_factory(),
        answer_model=ANSWER_MODEL,
        judge_models=JUDGE_MODELS,
        api_key=None,           # _get_client picks up the local placeholder
        budget=2000,
        parallel=1,             # one at a time — local hardware
        verbose=verbose,
    )

    print()
    print("=" * 60)
    print(f"[smoke] DONE — {report.total} samples")
    print(f"[smoke] by_judge: {json.dumps(report.by_judge, indent=2)}")
    print(f"[smoke] by_category: {json.dumps(report.by_category, indent=2)}")


def main() -> int:
    parser = argparse.ArgumentParser(prog="lme_smoke_local")
    parser.add_argument("--n", type=int, default=2,
                        help="Number of samples (default 2)")
    parser.add_argument("--variant", default="oracle",
                        help="LME variant (oracle|s|m); oracle is smallest")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--skip-schema", action="store_true",
                        help="Skip schema reapply (faster repeat runs)")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.skip_schema:
        _apply_fresh_schema()

    asyncio.run(_run(args.n, args.variant, args.verbose))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
