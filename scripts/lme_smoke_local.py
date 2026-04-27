"""Local LongMemEval smoke — minimal, fully configurable, dual-LLM.

Default stack (override via env vars or CLI flags below):
  - Embedder: OpenAI ``text-embedding-3-large`` (truncated to 1024-D)
  - Answer:   ``openai/gpt-5.2``
  - Judges:   ``openai/gpt-5.2`` + ``anthropic/claude-sonnet-4.6``  (dual)
  - Distill:  ``openai/gpt-5.2``  (Mem0-style fact extraction enabled)

Why this stack:
  - Dual-LLM cross-family judging (one OpenAI, one Anthropic) avoids
    same-family collusion bias on the score.
  - text-embedding-3-large @ 1024 dims keeps schema-compat with the
    existing v4 attestor schema (``embedding_dim=1024``) — no migration.
  - All four model slots are env-configurable so swapping in cheaper /
    faster models for iteration is one-line.

Configuration (every default is overridable):

  Env vars                          CLI flag                 Default
  -------------------------------- ------------------------ -------------------
  ANSWER_MODEL                      --answer-model            openai/gpt-5.2
  JUDGE_MODELS  (CSV)               --judge-model (repeat)    openai/gpt-5.2,anthropic/claude-sonnet-4.6
  DISTILL_MODEL                     --distill-model           openai/gpt-5.2
  USE_DISTILLATION  (1/0)           --no-distill              1 (on)
  OPENAI_EMBEDDING_MODEL            --embedding-model         text-embedding-3-large
  OPENAI_EMBEDDING_DIMENSIONS       --embedding-dim           1024
  PG_URL                            --pg-url                  postgresql://postgres:attestor@localhost:5432/attestor_v4_test
  LME_VARIANT                       --variant                 oracle
  LME_N                             --n                       2
  LME_PARALLEL                      --parallel                2
  LME_BUDGET                        --budget                  4000

Required:
  OPENAI_API_KEY      — used for both the LLM calls and embeddings
                        (set OPENROUTER_API_KEY instead if you prefer
                        the OpenRouter routing layer).

Usage (from repo root):
    OPENAI_API_KEY=... .venv/bin/python scripts/lme_smoke_local.py --n 2

Quick iteration with a cheaper model:
    ANSWER_MODEL=openai/gpt-4.1-mini .venv/bin/python scripts/lme_smoke_local.py --n 2
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

# Force line-buffered stdout/stderr for live progress in pipes / tee / CI.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except AttributeError:  # pragma: no cover — pre-3.7 fallback
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SCHEMA_PATH = ROOT / "attestor" / "store" / "schema.sql"


# ─── Defaults (every one is overridable) ───
DEFAULTS = {
    "answer_model":    "openai/gpt-5.2",
    "judge_models":    "openai/gpt-5.2,anthropic/claude-sonnet-4.6",
    "distill_model":   "openai/gpt-5.2",
    "use_distill":     True,
    "embedding_model": "text-embedding-3-large",
    "embedding_dim":   1024,
    "pg_url":          "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
    "variant":         "oracle",
    "n":               2,
    "parallel":        2,
    "budget":          4000,
}


def _env_or(name: str, default):
    """Read env var; coerce to the type of the default (bool / int / str)."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "y", "on")
    if isinstance(default, int):
        return int(raw)
    return raw


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="lme_smoke_local",
                                description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--answer-model",
                   default=_env_or("ANSWER_MODEL", DEFAULTS["answer_model"]))
    p.add_argument("--judge-model", action="append", default=None,
                   help="Repeat for multiple judges. If omitted, uses JUDGE_MODELS env (CSV) or the default pair.")
    p.add_argument("--distill-model",
                   default=_env_or("DISTILL_MODEL", DEFAULTS["distill_model"]))
    p.add_argument("--no-distill", action="store_true",
                   help="Disable distillation (default is on)")
    p.add_argument("--embedding-model",
                   default=_env_or("OPENAI_EMBEDDING_MODEL", DEFAULTS["embedding_model"]))
    p.add_argument("--embedding-dim", type=int,
                   default=_env_or("OPENAI_EMBEDDING_DIMENSIONS", DEFAULTS["embedding_dim"]))
    p.add_argument("--pg-url",
                   default=_env_or("PG_URL", DEFAULTS["pg_url"]))
    p.add_argument("--variant",
                   default=_env_or("LME_VARIANT", DEFAULTS["variant"]),
                   help="LME variant (oracle|s|m); oracle is smallest / fastest")
    p.add_argument("--n", type=int,
                   default=_env_or("LME_N", DEFAULTS["n"]))
    p.add_argument("--parallel", type=int,
                   default=_env_or("LME_PARALLEL", DEFAULTS["parallel"]))
    p.add_argument("--budget", type=int,
                   default=_env_or("LME_BUDGET", DEFAULTS["budget"]))
    p.add_argument("--skip-schema", action="store_true",
                   help="Skip schema reapply (faster repeat runs)")
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    # Resolve judge models with the documented precedence:
    # CLI flag (repeated) > JUDGE_MODELS env (CSV) > default pair.
    if args.judge_model:
        judges = args.judge_model
    else:
        raw = os.environ.get("JUDGE_MODELS", DEFAULTS["judge_models"])
        judges = [j.strip() for j in raw.split(",") if j.strip()]
    args.judge_models = judges

    args.use_distillation = not args.no_distill and _env_or(
        "USE_DISTILLATION", DEFAULTS["use_distill"]
    )

    return args


# ─── Setup helpers ───

def _force_openai_embedder(model: str, dim: int) -> None:
    """Push attestor's embedding probe to OpenAI text-embedding-3-large.

    attestor/store/embeddings.py probes Ollama first by default; we
    disable that and pin model + dimensions so the smoke is deterministic
    regardless of what's running on localhost:11434.
    """
    os.environ["ATTESTOR_DISABLE_LOCAL_EMBED"] = "1"
    os.environ["OPENAI_EMBEDDING_MODEL"] = model
    os.environ["OPENAI_EMBEDDING_DIMENSIONS"] = str(dim)


def _apply_fresh_schema(pg_url: str, embedding_dim: int) -> None:
    """Drop + recreate v4 tables so each smoke starts clean."""
    import psycopg2

    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", str(embedding_dim))
    with psycopg2.connect(pg_url) as c, c.cursor() as cur:
        for tbl in ("deletion_audit", "user_quotas", "memories", "episodes",
                    "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
        c.commit()
    print(f"[smoke] schema applied (embedding_dim={embedding_dim})")


def _mem_factory(pg_url: str):
    """Return a zero-arg callable producing fresh AgentMemory instances."""
    from attestor.core import AgentMemory
    from urllib.parse import urlparse

    parsed = urlparse(pg_url)
    db = (parsed.path or "/").lstrip("/") or "attestor_v4_test"

    def _make():
        # Each sample gets its own SOLO user via tempdir path (LME's
        # per-sample isolation guarantee).
        path = tempfile.mkdtemp(prefix="lme_smoke_")
        return AgentMemory(path, config={
            "mode": "solo",
            "backends": ["postgres"],
            "backend_configs": {"postgres": {
                "url": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 5432}",
                "database": db,
                "auth": {
                    "username": parsed.username or "postgres",
                    "password": parsed.password or "attestor",
                },
                "v4": True,
                "skip_schema_init": True,
            }},
        })

    return _make


# ─── Main run ───

async def _run(args: argparse.Namespace) -> None:
    from attestor.longmemeval import load_or_download, run_async

    if not (os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")):
        sys.stderr.write(
            "[smoke] error: neither OPENAI_API_KEY nor OPENROUTER_API_KEY is set.\n"
            "        Both LLM calls and the OpenAI embedder need one of these.\n"
        )
        raise SystemExit(2)

    print(f"[smoke] loading LME variant={args.variant!r}…")
    samples = load_or_download(variant=args.variant)
    print(f"[smoke] dataset has {len(samples)} samples; running first {args.n}")

    print(f"[smoke] embedder  = {args.embedding_model!r} @ {args.embedding_dim}-D")
    print(f"[smoke] answerer  = {args.answer_model!r}")
    print(f"[smoke] judges    = {args.judge_models!r}  (dual-LLM)")
    if args.use_distillation:
        print(f"[smoke] distiller = {args.distill_model!r}  (distillation ON)")
    else:
        print("[smoke] distiller = (distillation OFF)")
    print(f"[smoke] budget    = {args.budget} tokens · parallel = {args.parallel}")

    report = await run_async(
        samples[: args.n],
        mem_factory=_mem_factory(args.pg_url),
        answer_model=args.answer_model,
        judge_models=args.judge_models,
        distill_model=args.distill_model,
        use_distillation=args.use_distillation,
        api_key=None,  # _get_client reads OPENAI_API_KEY / OPENROUTER_API_KEY
        budget=args.budget,
        parallel=args.parallel,
        verbose=args.verbose,
    )

    print()
    print("=" * 60)
    print(f"[smoke] DONE — {report.total} samples")
    print(f"[smoke] by_judge: {json.dumps(report.by_judge, indent=2)}")
    print(f"[smoke] by_category: {json.dumps(report.by_category, indent=2)}")


def main() -> int:
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    _force_openai_embedder(args.embedding_model, args.embedding_dim)

    if not args.skip_schema:
        _apply_fresh_schema(args.pg_url, args.embedding_dim)

    asyncio.run(_run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
