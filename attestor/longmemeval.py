"""LongMemEval benchmark runner for Attestor.

LongMemEval (Wu et al., ICLR 2025) evaluates long-term memory of chat
assistants across five abilities: information extraction, multi-session
reasoning, temporal reasoning, knowledge updates, and abstention. The
cleaned 500-question release splits those into six ``question_type``
categories in the dataset.

This module is the Attestor-native runner. It mirrors ``attestor.locomo``
but targets the LongMemEval schema directly:

    Sample = {
        "question_id": str,
        "question_type": str,          # one of CATEGORY_NAMES keys
        "question": str,
        "question_date": str,          # "YYYY/MM/DD (DayOfWeek) HH:MM"
        "answer": str,                 # gold
        "answer_session_ids": list[str],
        "haystack_dates": list[str],   # per-session timestamps
        "haystack_session_ids": list[str],
        "haystack_sessions": list[list[{"role": str, "content": str}]],
    }

Dataset source (HuggingFace):
    https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned

Phase 1 scope (this revision): dataset schema, frozen ``LMESample``
dataclass, loader + downloader, and date parser. Later phases add
ingest / answer / judge / run.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HF_BASE_URL = (
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main"
)

# Dataset variants. Keep the raw filenames — they are stable on HuggingFace.
DATASET_VARIANTS: dict[str, str] = {
    "oracle": "longmemeval_oracle.json",
    "s": "longmemeval_s_cleaned.json",
    "m": "longmemeval_m_cleaned.json",
}

# Six question_type values present in longmemeval_s_cleaned.json (n=500).
# The kill-switch category (roadmap §10) is ``temporal-reasoning``.
CATEGORY_NAMES: dict[str, str] = {
    "single-session-user": "single_session_user",
    "single-session-assistant": "single_session_assistant",
    "single-session-preference": "single_session_preference",
    "multi-session": "multi_session",
    "temporal-reasoning": "temporal_reasoning",
    "knowledge-update": "knowledge_update",
}

TEMPORAL_CATEGORY = "temporal-reasoning"

# Date format used throughout the dataset: e.g. "2023/05/30 (Tue) 23:40".
_DATE_FMT = "%Y/%m/%d (%a) %H:%M"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LMETurn:
    """A single turn inside a haystack session."""

    role: str
    content: str


@dataclass(frozen=True)
class LMESample:
    """One LongMemEval question with its haystack context.

    Frozen so benchmark runs cannot accidentally mutate gold data.
    """

    question_id: str
    question_type: str
    question: str
    question_date: str
    answer: str
    answer_session_ids: Tuple[str, ...]
    haystack_dates: Tuple[str, ...]
    haystack_session_ids: Tuple[str, ...]
    haystack_sessions: Tuple[Tuple[LMETurn, ...], ...]

    @property
    def is_temporal(self) -> bool:
        return self.question_type == TEMPORAL_CATEGORY

    @property
    def total_haystack_turns(self) -> int:
        return sum(len(sess) for sess in self.haystack_sessions)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def parse_lme_date(date_str: str) -> Optional[datetime]:
    """Parse LongMemEval dates like ``'2023/05/30 (Tue) 23:40'``.

    Returns ``None`` if the string is empty or unparsable rather than
    raising — the benchmark tolerates missing dates on a per-row basis.
    """
    if not date_str:
        return None
    try:
        return datetime.strptime(date_str.strip(), _DATE_FMT)
    except ValueError:
        logger.warning("Unparsable LME date: %r", date_str)
        return None


# ---------------------------------------------------------------------------
# Dataset loading / downloading
# ---------------------------------------------------------------------------


def _resolve_variant(variant: str) -> str:
    if variant not in DATASET_VARIANTS:
        raise ValueError(
            f"Unknown LongMemEval variant {variant!r}. "
            f"Choose one of: {sorted(DATASET_VARIANTS)}"
        )
    return DATASET_VARIANTS[variant]


def download_longmemeval(dest: Path | str, variant: str = "s") -> Path:
    """Download a LongMemEval variant from HuggingFace if not already present.

    Args:
        dest: Directory to place the file in. Created if missing.
        variant: One of ``"oracle"``, ``"s"``, ``"m"``.

    Returns:
        Absolute path of the downloaded file.
    """
    filename = _resolve_variant(variant)
    dest_dir = Path(dest).expanduser().resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename

    if target.exists():
        logger.info("LongMemEval %s already present at %s", variant, target)
        return target

    url = f"{HF_BASE_URL}/{filename}"
    logger.info("Downloading LongMemEval %s from %s", variant, url)
    # bandit B310: urllib-urlopen — URL is a hard-coded HuggingFace release,
    # not user input, and the file is opened read-only. Safe for this use.
    with urllib.request.urlopen(url) as resp:  # noqa: S310
        target.write_bytes(resp.read())
    return target


def _coerce_turn(raw: Any) -> LMETurn:
    """Tolerant turn coercion — upstream dataset occasionally has stray keys."""
    if isinstance(raw, LMETurn):
        return raw
    if not isinstance(raw, dict):
        raise ValueError(f"LongMemEval turn is not a dict: {raw!r}")
    role = raw.get("role") or raw.get("speaker") or ""
    content = raw.get("content") or raw.get("text") or ""
    return LMETurn(role=str(role), content=str(content))


def _coerce_sample(raw: dict[str, Any]) -> LMESample:
    sessions = tuple(
        tuple(_coerce_turn(t) for t in session)
        for session in raw.get("haystack_sessions", [])
    )
    return LMESample(
        question_id=str(raw["question_id"]),
        question_type=str(raw["question_type"]),
        question=str(raw["question"]),
        question_date=str(raw.get("question_date", "")),
        answer=str(raw["answer"]),
        answer_session_ids=tuple(str(x) for x in raw.get("answer_session_ids", [])),
        haystack_dates=tuple(str(x) for x in raw.get("haystack_dates", [])),
        haystack_session_ids=tuple(
            str(x) for x in raw.get("haystack_session_ids", [])
        ),
        haystack_sessions=sessions,
    )


def load_longmemeval(
    path: Path | str, *, limit: Optional[int] = None
) -> list[LMESample]:
    """Load a LongMemEval JSON file and return a list of frozen samples.

    Args:
        path: Path to a ``longmemeval_*.json`` file.
        limit: Optional max number of samples to return (for CI speed).

    Raises:
        FileNotFoundError: if ``path`` does not exist.
        ValueError: if the file is not a list of samples.
    """
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"LongMemEval file not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(
            f"Expected a JSON list of samples in {p}, got {type(raw).__name__}"
        )
    samples = [_coerce_sample(s) for s in raw]
    if limit is not None:
        samples = samples[:limit]
    return samples


def load_or_download(
    cache_dir: Path | str | None = None, variant: str = "s"
) -> list[LMESample]:
    """Convenience: ensure the variant is on disk, then load it.

    Uses ``$XDG_CACHE_HOME/attestor/longmemeval`` (or ``~/.cache/attestor/longmemeval``)
    by default.
    """
    if cache_dir is None:
        xdg = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        cache_dir = Path(xdg) / "attestor" / "longmemeval"
    path = download_longmemeval(cache_dir, variant=variant)
    return load_longmemeval(path)
