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


# ---------------------------------------------------------------------------
# Namespace + content helpers
# ---------------------------------------------------------------------------


def namespace_for(sample: LMESample) -> str:
    """Per-sample namespace so haystacks from different samples do not collide."""
    return f"lme_{sample.question_id}"


def _iso_date(raw: str) -> str:
    """Return an ISO-8601 event_date (``YYYY-MM-DDTHH:MM``) or the raw string if unparsable."""
    dt = parse_lme_date(raw)
    if dt is None:
        return raw
    return dt.strftime("%Y-%m-%dT%H:%M")


def _short_date(raw: str) -> str:
    """Return a compact date tag (``YYYY-MM-DD``) for inline content prefixes."""
    dt = parse_lme_date(raw)
    if dt is None:
        return raw
    return dt.strftime("%Y-%m-%d")


def _format_turn_content(role: str, text: str, date_tag: str) -> str:
    """Belt-and-suspenders content: inline date tag so even backends that drop
    ``event_date`` still carry the date through vector + FTS paths.
    """
    display_role = "User" if role == "user" else "Assistant" if role == "assistant" else role or "Unknown"
    return f"[{date_tag}] {display_role}: {text}".strip()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class IngestStats:
    """Counts returned from ingest_history — useful for assertions and logs."""

    turns_seen: int
    memories_added: int
    sessions: int
    skipped_empty: int


def ingest_history(
    mem: Any,
    sample: LMESample,
    *,
    use_extraction: bool = False,
    extraction_model: str = "openai/gpt-4.1-mini",
    api_key: Optional[str] = None,
    verbose: bool = False,
) -> IngestStats:
    """Ingest a LongMemEval haystack into an Attestor ``AgentMemory``.

    Two strategies:

    1. ``use_extraction=False`` (raw): store each turn as a memory, prefixed
       with ``[YYYY-MM-DD] Role:`` and tagged with the ISO ``event_date``.
       This is the belt-and-suspenders option — any backend that strips
       ``event_date`` still has the date inline.
    2. ``use_extraction=True``: run Attestor's LLM extractor per session to
       distill atomic facts + relation triples, then store those.

    Args:
        mem: Instantiated ``attestor.core.AgentMemory``.
        sample: Frozen ``LMESample``.
        use_extraction: Extract atomic facts instead of ingesting raw turns.
        extraction_model: OpenRouter model id (only used when extracting).
        api_key: Optional override for the extractor's API key.
        verbose: Print per-session progress to stdout.

    Returns:
        ``IngestStats`` with turn / memory / session counts.
    """
    ns = namespace_for(sample)
    turns_seen = 0
    memories_added = 0
    skipped_empty = 0

    sessions = list(
        zip(sample.haystack_session_ids, sample.haystack_dates, sample.haystack_sessions)
    )

    if use_extraction:
        # Lazy import to keep the hot path (raw) free of extractor deps.
        from attestor.extraction.extractor import extract_from_session  # type: ignore

        for session_id, session_date, turns in sessions:
            # Map LongMemEval roles onto the speaker_a / speaker_b contract the
            # extractor expects.
            adapted_turns = [
                {
                    "speaker": "A" if t.role == "user" else "B",
                    "text": t.content,
                    "dia_id": f"{session_id}_t{idx}",
                }
                for idx, t in enumerate(turns)
                if t.content.strip()
            ]
            turns_seen += len(turns)
            skipped_empty += len(turns) - len(adapted_turns)
            if not adapted_turns:
                continue

            if verbose:
                print(f"    extracting {session_id} ({len(adapted_turns)} turns)")

            memories, _triples = extract_from_session(
                turns=adapted_turns,
                speaker_a="User",
                speaker_b="Assistant",
                session_date=_iso_date(session_date),
                model=extraction_model,
                api_key=api_key,
            )
            for m in memories:
                mem.add(
                    content=m.content,
                    tags=list(m.tags) + ["lme", session_id],
                    category=m.category,
                    entity=m.entity,
                    namespace=ns,
                    event_date=m.event_date or _iso_date(session_date),
                    confidence=m.confidence,
                    metadata={"session_id": session_id, "source": "lme_extracted"},
                )
                memories_added += 1
        return IngestStats(
            turns_seen=turns_seen,
            memories_added=memories_added,
            sessions=len(sessions),
            skipped_empty=skipped_empty,
        )

    # Raw path — option C: inline date tag in content AND event_date kwarg.
    for session_id, session_date, turns in sessions:
        iso = _iso_date(session_date)
        short = _short_date(session_date)
        if verbose:
            print(f"    raw ingest {session_id} ({len(turns)} turns) date={short}")
        for idx, turn in enumerate(turns):
            turns_seen += 1
            text = turn.content.strip()
            if not text:
                skipped_empty += 1
                continue
            mem.add(
                content=_format_turn_content(turn.role, text, short),
                tags=[turn.role or "unknown", session_id, "lme"],
                category="conversation",
                entity=None,
                namespace=ns,
                event_date=iso,
                metadata={
                    "session_id": session_id,
                    "role": turn.role,
                    "turn_idx": idx,
                    "source": "lme_raw",
                },
            )
            memories_added += 1

    return IngestStats(
        turns_seen=turns_seen,
        memories_added=memories_added,
        sessions=len(sessions),
        skipped_empty=skipped_empty,
    )
