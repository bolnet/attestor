"""Bench-only configuration loader.

Mirrors the shape of ``attestor.config`` but for ``configs/bench.yaml``.
The two files are intentionally split:

  * ``configs/attestor.yaml`` — production stack (embedder, models,
    retrieval features, DBs, registries, clouds). Loaded via
    :func:`attestor.config.get_stack`.
  * ``configs/bench.yaml`` — bench-only knobs (dataset variants,
    category iteration order, target scores, output paths). Loaded
    here.

**Hard invariant:** the dotted-key sets of the two YAMLs must be
disjoint. :func:`get_bench` raises :class:`KeyOverlapError` if any key
appears in both files. The same rule is also enforced at commit time
by ``tests/test_config_no_duplicate_keys.py``.

Two layers of defense matter because a duplicate would otherwise
silently produce benchmark vs production drift — exactly the failure
mode the user pushed back against in the prior session.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from attestor.config import StackConfig, get_stack

# ──────────────────────────────────────────────────────────────────────
# Errors
# ──────────────────────────────────────────────────────────────────────


class KeyOverlapError(RuntimeError):
    """Raised when ``bench.yaml`` duplicates a key from ``attestor.yaml``.

    The two files must have disjoint dotted-key sets — if they overlap,
    we cannot tell which file "wins" without picking an arbitrary order,
    and bench/production drift becomes silent.
    """


# ──────────────────────────────────────────────────────────────────────
# Dataclasses
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class LMECfg:
    """LongMemEval bench harness knobs."""

    variant: str = "s"
    cache_dir: str = "~/.cache/attestor/lme"
    output_dir: str = "docs/bench"
    sample_limit: Optional[int] = None
    category: Optional[str] = None
    categories: List[str] = field(default_factory=list)
    variants_to_run: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class KnowledgeUpdatesCfg:
    """Synthetic Knowledge-Updates supersession suite knobs."""

    fixtures_path: str = "evals/knowledge_updates/fixtures.json"
    n_cases: int = 50
    target_score: float = 0.92
    categories: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class ReportCfg:
    """Reporting + trend-tracking knobs."""

    headline_slice: str = "abstention"
    trend_csv: str = "docs/bench/trend.csv"
    markdown_path: str = "docs/bench/LME-S.md"


@dataclass(frozen=True)
class BenchCfg:
    """Frozen bench config — combines stack (loaded via get_stack) +
    bench-only sections from configs/bench.yaml.

    Always access stack knobs via ``cfg.stack.*`` so it's obvious which
    file a value comes from.
    """

    stack: StackConfig
    lme: LMECfg
    knowledge_updates: KnowledgeUpdatesCfg
    report: ReportCfg


# ──────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────

DEFAULT_BENCH_CONFIG = "configs/bench.yaml"
DEFAULT_STACK_CONFIG = "configs/attestor.yaml"


# ──────────────────────────────────────────────────────────────────────
# Key-overlap enforcement
# ──────────────────────────────────────────────────────────────────────


def _flatten_keys(node: Any, prefix: str = "") -> set[str]:
    """Recursively flatten a YAML structure to a set of dotted keys.

    Lists are treated as leaves — their items are not flattened. This
    matches the user's rule: "if a key like ``models:`` appears in
    both files, that's a violation," regardless of whether the value
    is a scalar, list, or nested dict.
    """
    out: set[str] = set()
    if not isinstance(node, dict):
        return out
    for k, v in node.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            nested = _flatten_keys(v, key)
            if nested:
                out |= nested
            else:
                out.add(key)
        else:
            out.add(key)
    return out


def assert_disjoint_keys(
    attestor_yaml: dict, bench_yaml: dict,
) -> None:
    """Raise KeyOverlapError if the two YAMLs share any dotted key."""
    overlap = _flatten_keys(attestor_yaml) & _flatten_keys(bench_yaml)
    if overlap:
        raise KeyOverlapError(
            f"configs/attestor.yaml and configs/bench.yaml share "
            f"{len(overlap)} key(s): {sorted(overlap)}. Bench file is "
            f"strictly bench-only knobs — remove these from "
            f"configs/bench.yaml. Stack knobs live in attestor.yaml only."
        )


# ──────────────────────────────────────────────────────────────────────
# Loader
# ──────────────────────────────────────────────────────────────────────


def _parse_bench_yaml(path: Path) -> dict:
    """Read bench.yaml and return its raw dict. Empty file → empty dict."""
    raw = yaml.safe_load(path.read_text()) or {}
    if not isinstance(raw, dict):
        raise SystemExit(
            f"[attestor.bench_config] {path}: top-level must be a mapping"
        )
    return raw


def _build_bench_cfg(stack: StackConfig, bench_raw: dict) -> BenchCfg:
    """Map a raw bench dict + StackConfig into a frozen BenchCfg."""
    bench_section: Dict[str, Any] = bench_raw.get("bench", {}) or {}

    lme_raw = bench_section.get("lme", {}) or {}
    ku_raw = bench_section.get("knowledge_updates", {}) or {}
    report_raw = bench_section.get("report", {}) or {}

    return BenchCfg(
        stack=stack,
        lme=LMECfg(
            variant=str(lme_raw.get("variant", "s")),
            cache_dir=str(lme_raw.get("cache_dir", "~/.cache/attestor/lme")),
            output_dir=str(lme_raw.get("output_dir", "docs/bench")),
            sample_limit=lme_raw.get("sample_limit"),
            category=lme_raw.get("category"),
            categories=list(lme_raw.get("categories") or []),
            variants_to_run=list(lme_raw.get("variants_to_run") or []),
        ),
        knowledge_updates=KnowledgeUpdatesCfg(
            fixtures_path=str(ku_raw.get(
                "fixtures_path", "evals/knowledge_updates/fixtures.json",
            )),
            n_cases=int(ku_raw.get("n_cases", 50)),
            target_score=float(ku_raw.get("target_score", 0.92)),
            categories=list(ku_raw.get("categories") or []),
        ),
        report=ReportCfg(
            headline_slice=str(report_raw.get("headline_slice", "abstention")),
            trend_csv=str(report_raw.get("trend_csv", "docs/bench/trend.csv")),
            markdown_path=str(report_raw.get(
                "markdown_path", "docs/bench/LME-S.md",
            )),
        ),
    )


def load_bench(
    bench_path: Path | str | None = None,
    *,
    stack_path: Path | str | None = None,
) -> BenchCfg:
    """Read both YAMLs, enforce disjoint keys, and build a BenchCfg.

    Bypasses the cache. Most callers should use :func:`get_bench`
    instead.
    """
    bench_p = Path(
        bench_path
        or os.environ.get("ATTESTOR_BENCH_CONFIG")
        or DEFAULT_BENCH_CONFIG
    )
    stack_p = Path(
        stack_path
        or os.environ.get("ATTESTOR_CONFIG")
        or DEFAULT_STACK_CONFIG
    )

    if not bench_p.exists():
        raise SystemExit(f"[attestor.bench_config] missing: {bench_p}")
    if not stack_p.exists():
        raise SystemExit(f"[attestor.bench_config] missing: {stack_p}")

    bench_raw = _parse_bench_yaml(bench_p)
    stack_raw = yaml.safe_load(stack_p.read_text()) or {}

    # Enforce the hard invariant BEFORE building anything.
    assert_disjoint_keys(stack_raw, bench_raw)

    # Stack loader is responsible for env resolution + dataclass build.
    from attestor.config import load_stack
    stack = load_stack(stack_p)

    return _build_bench_cfg(stack, bench_raw)


_cached_bench: Optional[BenchCfg] = None
_CACHE_LOCK = threading.Lock()


def get_bench() -> BenchCfg:
    """Return the cached BenchCfg, loading on first call.

    Thread-safe via double-checked locking — same pattern as
    :func:`attestor.config.get_stack`.
    """
    global _cached_bench
    if _cached_bench is None:
        with _CACHE_LOCK:
            if _cached_bench is None:
                _cached_bench = load_bench()
    return _cached_bench


def reset_bench() -> None:
    """Drop the cache. Next get_bench() re-reads both YAMLs."""
    global _cached_bench
    with _CACHE_LOCK:
        _cached_bench = None


# ──────────────────────────────────────────────────────────────────────
# Banner — bench scripts MUST call this at startup so the operator
# sees both [stack] and [bench] blocks.
# ──────────────────────────────────────────────────────────────────────


def print_bench_banner(cfg: BenchCfg, *, run_label: str) -> None:
    """Print [stack] and [bench] blocks side-by-side at startup."""
    s = cfg.stack
    pg = s.postgres.url.split("@", 1)[-1] if "@" in s.postgres.url else s.postgres.url
    print(f"[{run_label}]")
    print(
        f"  [stack]   embedder={s.embedder.model} ({s.embedder.dimensions}d) "
        f"· answerer={s.models.answerer} · judge={s.models.judge} "
        f"· pg={pg} · neo4j={s.neo4j.url}",
    )
    print(
        f"  [bench]   variant={cfg.lme.variant} "
        f"· category={cfg.lme.category or 'ALL'} "
        f"· sample_limit={cfg.lme.sample_limit or 'full'} "
        f"· parallel={s.parallel} · budget={s.budget} "
        f"· output_dir={cfg.lme.output_dir}",
    )
