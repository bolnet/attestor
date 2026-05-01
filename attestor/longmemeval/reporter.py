"""LongMemEval result aggregation and reporting.

Per-sample and per-run dataclasses, summarizers, and provenance
helpers (git SHA, attestor version, file hashes). Every dataclass
here is JSON-serializable via ``dataclasses.asdict``.
"""

from __future__ import annotations

import hashlib
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Per-sample + per-run dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SampleReport:
    """Per-sample outcome of a LongMemEval run."""

    question_id: str
    category: str
    question: str
    gold: str
    answer: str
    judgments: dict  # judge_model -> JudgeResult (dict for JSON-serializability)
    answer_latency_ms: float
    ingest_turns: int
    ingest_memories: int
    retrieved_count: int
    # Dimension-B telemetry — per-sample multi-dimensional scoring.
    gold_session_ids: tuple[str, ...] = ()       # from sample.answer_session_ids
    retrieved_session_ids: tuple[str, ...] = ()  # from AnswerResult
    retrieval_hit: bool = False                  # any overlap between gold & retrieved
    retrieval_overlap: int = 0                   # count of overlapping sessions
    predicted_mode: str = ""                     # "fact" | "recommendation" | ""
    personalization: dict | None = None       # JudgeResult dict, only on RECOMMENDATION samples


@dataclass(frozen=True)
class RunProvenance:
    """Audit metadata written into every LMERunReport.

    Captures the six pieces needed for third-party verification:
      - git SHA of the attestor code that produced the run
      - exact argv executed
      - SHA256 of the dataset file
      - Attestor package version
      - UTC timestamps (also in LMERunReport, repeated for self-containment)
      - Host fingerprint (platform + python version — no PII)
    """

    git_sha: str
    git_dirty: bool
    attestor_version: str
    python_version: str
    platform: str
    argv: tuple[str, ...]
    dataset_path: str
    dataset_sha256: str
    dataset_sample_count: int
    started_at_utc: str
    completed_at_utc: str


@dataclass(frozen=True)
class LMERunReport:
    """Aggregated results of a run. JSON-serializable via ``asdict``."""

    total: int
    answer_model: str
    judge_models: tuple[str, ...]
    by_category: dict  # category -> {judge_model -> {correct, total, accuracy}}
    by_judge: dict     # judge_model -> {correct, total, accuracy}
    started_at: str
    completed_at: str
    samples: tuple[SampleReport, ...]
    provenance: RunProvenance | None = None
    run_config: dict = field(default_factory=dict)
    schema_version: str = "1.1"
    by_dimension: dict = field(default_factory=dict)
    # by_dimension shape (when non-empty):
    #   {
    #     "retrieval":          {"hits": int, "total": int, "precision": pct},
    #     "mode_distribution":  {"counts": {...}, "fact_pct": pct, "recommendation_pct": pct},
    #     "personalization":    {"correct": int, "total": int, "accuracy": pct},
    #     "by_predicted_mode":  {"fact": {correct,total,accuracy}, "recommendation": {...}, "unknown": {...}},
    #   }


# ---------------------------------------------------------------------------
# Summarizers
# ---------------------------------------------------------------------------


def _blank_counter() -> dict:
    return {"correct": 0, "total": 0}


def _accuracy(bucket: dict) -> dict:
    """Add a percentage accuracy field to a {correct,total} bucket."""
    total = bucket.get("total", 0)
    correct = bucket.get("correct", 0)
    pct = round(100.0 * correct / total, 2) if total else 0.0
    return {**bucket, "accuracy": pct}


def _summarize(
    sample_reports: list[SampleReport], judge_models: list[str]
) -> tuple[dict, dict]:
    """Return (by_category, by_judge) nested dicts with accuracy baked in."""
    by_category: dict = {}
    by_judge: dict = {jm: _blank_counter() for jm in judge_models}

    for sr in sample_reports:
        cat = sr.category
        by_category.setdefault(cat, {jm: _blank_counter() for jm in judge_models})
        for jm in judge_models:
            j = sr.judgments.get(jm)
            if not j:
                continue
            correct = bool(j["correct"])
            by_category[cat][jm]["total"] += 1
            by_judge[jm]["total"] += 1
            if correct:
                by_category[cat][jm]["correct"] += 1
                by_judge[jm]["correct"] += 1

    by_category_pct = {
        cat: {jm: _accuracy(bucket) for jm, bucket in per_judge.items()}
        for cat, per_judge in by_category.items()
    }
    by_judge_pct = {jm: _accuracy(b) for jm, b in by_judge.items()}
    return by_category_pct, by_judge_pct


def _summarize_dimensions(sample_reports: list[SampleReport]) -> dict:
    """Aggregate the dimension-B per-sample telemetry into report buckets.

    Buckets:
      - retrieval: hits / total / precision  (how often gold session was in top-K)
      - mode_distribution: counts of fact / recommendation / unknown
      - personalization: correct / total / accuracy  (only on rec-mode samples)
    """
    total = len(sample_reports)
    if total == 0:
        return {}

    # Retrieval precision (gold session in top-K retrieved)
    retr_hits = sum(1 for s in sample_reports if s.retrieval_hit)
    retr_total = sum(1 for s in sample_reports if s.gold_session_ids)
    retrieval = {
        "hits": retr_hits,
        "total": retr_total,
        "precision": round(100.0 * retr_hits / retr_total, 2) if retr_total else 0.0,
    }

    # Mode distribution
    mode_counts = {"fact": 0, "recommendation": 0, "unknown": 0}
    for s in sample_reports:
        m = s.predicted_mode or "unknown"
        if m not in mode_counts:
            mode_counts[m] = 0
        mode_counts[m] += 1
    mode_distribution = {
        "counts": mode_counts,
        "fact_pct": round(100.0 * mode_counts.get("fact", 0) / total, 2),
        "recommendation_pct": round(100.0 * mode_counts.get("recommendation", 0) / total, 2),
    }

    # Personalization (only meaningful on recommendation-mode samples)
    pers_samples = [s for s in sample_reports if s.personalization is not None]
    pers_correct = sum(1 for s in pers_samples if s.personalization.get("correct"))
    pers_total = len(pers_samples)
    personalization = {
        "correct": pers_correct,
        "total": pers_total,
        "accuracy": round(100.0 * pers_correct / pers_total, 2) if pers_total else 0.0,
    }

    # Per-mode answer accuracy slice — useful for understanding whether mode
    # selection alone moves the overall number.
    per_mode = {"fact": _blank_counter(), "recommendation": _blank_counter(), "unknown": _blank_counter()}
    for s in sample_reports:
        m = s.predicted_mode or "unknown"
        if m not in per_mode:
            per_mode[m] = _blank_counter()
        # Use the FIRST judge as the canonical correctness signal here.
        first_judge = next(iter(s.judgments.values()), None) if s.judgments else None
        if first_judge is None:
            continue
        per_mode[m]["total"] += 1
        if first_judge.get("correct"):
            per_mode[m]["correct"] += 1
    per_mode_pct = {m: _accuracy(b) for m, b in per_mode.items()}

    return {
        "retrieval": retrieval,
        "mode_distribution": mode_distribution,
        "personalization": personalization,
        "by_predicted_mode": per_mode_pct,
    }


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _git_sha() -> tuple[str, bool]:
    """Return (sha, dirty). ``sha='unknown'`` when attestor is not in a git tree.

    timeout=5 on each git call: a hung git process (e.g. waiting on an
    interactive credential prompt, or a wedged filesystem) MUST NOT block
    a benchmark run. ``TimeoutExpired`` is treated like ``FileNotFoundError``
    — return the missing-git fallback rather than re-raising.
    """
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=5,
        )
        return sha, bool(status.strip())
    except subprocess.TimeoutExpired:
        return "unknown", False
    except Exception:
        return "unknown", False


def _attestor_version() -> str:
    try:
        import importlib.metadata

        return importlib.metadata.version("attestor")
    except Exception:
        return "unknown"


def _sha256_file(path: Path | str) -> str:
    """SHA256 of a file, streamed — safe for multi-hundred-MB datasets."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
