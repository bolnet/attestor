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

import asyncio
import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

def _default_model() -> str:
    """Resolve the LME benchmark default model from
    ``configs/attestor.yaml``."""
    from attestor.config import get_stack
    return get_stack().models.benchmark_default


def _default_judges() -> tuple[str, ...]:
    """Resolve the dual-judge panel: the primary judge from the YAML
    plus the verifier (cross-family) — matches the canonical recommended
    ``Judge=gpt-4.1, Verifier=claude-sonnet-4-6`` pairing."""
    from attestor.config import get_stack
    s = get_stack()
    return (s.models.judge, s.models.verifier)


DEFAULT_MODEL = _default_model()
# Default dual-judge. Second judge anchors out answerer-judge collusion.
DEFAULT_JUDGES = _default_judges()
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_PARALLEL = 4

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
    distilled_facts: int = 0  # count of LLM-distilled facts when use_distillation=True
    skipped_by_distiller: int = 0  # turns the distiller marked SKIP


# ---------------------------------------------------------------------------
# Per-turn distillation (structured, universal schema)
# ---------------------------------------------------------------------------
#
# Memories are distilled into a structured record, not prose. Each record
# carries enough signal for the retrieval layer to boost-and-filter and for
# the answerer to disambiguate among multiple candidates without a second
# LLM call at query time. Fields are universal (they work for any memory-
# layer benchmark and any real agent use case), not LongMemEval-specific.

# Allowed vocabularies — the parser normalizes to these. Anything the LLM
# emits outside these sets is coerced to the sensible fallback.
_CLAIM_TYPES = (
    "fact",            # neutral factual statement (default)
    "preference",      # user's like/dislike/constraint/priority
    "recommendation",  # assistant explicit suggestion with a named target
    "event",           # dated or schedulable occurrence
    "opinion",         # speaker's subjective view
    "mentioned",       # low-salience reference, retrievable but not boosted
)

_SPEAKERS = ("user", "assistant", "unknown")
_EMPHASIS_LEVELS = ("explicit", "mentioned", "implied")


@dataclass(frozen=True)
class DistilledFact:
    """A single fact extracted from a turn, with retrieval-relevant metadata.

    Universal schema (not benchmark-specific):
      - ``content``: the distilled prose sentence itself.
      - ``speaker``: who authored the underlying claim ("user" / "assistant").
      - ``claim_type``: what KIND of statement this is — drives retrieval boost.
      - ``emphasis``: salience within the turn — "explicit" beats "mentioned"
        when the answerer must pick one candidate among several.
      - ``entities``: named entities the fact is about (proper nouns).
      - ``topics``: free-form topical tags (domain keywords).
    """

    content: str
    speaker: str = "unknown"
    claim_type: str = "fact"
    emphasis: str = "mentioned"
    entities: Tuple[str, ...] = ()
    topics: Tuple[str, ...] = ()


DISTILL_PROMPT = (
    "You are a precise memory distillation agent for a long-term memory "
    "system. Extract durable facts from ONE conversation turn and return "
    "them as a JSON array of structured records.\n\n"
    "Each record must have these fields (all required, though lists may "
    "be empty):\n"
    '  "content":   third-person sentence (see rules below)\n'
    '  "speaker":   "user" or "assistant" (who made the underlying claim)\n'
    '  "claim_type": one of: fact, preference, recommendation, event, '
    "opinion, mentioned\n"
    '  "emphasis":  "explicit" (speaker named it specifically / endorsed '
    'it), "mentioned" (referenced in passing), or "implied"\n'
    '  "entities":  array of named entities the fact is about\n'
    '  "topics":    array of short topical tags (lowercase nouns)\n\n'
    "CLAIM_TYPE guide (pick the most specific match):\n"
    "  preference     — user expresses a like / dislike / constraint / "
    "priority (e.g. 'The user prefers dark chocolate').\n"
    "  recommendation — assistant explicitly suggests / recommends / "
    "endorses a NAMED target (e.g. 'The assistant recommended Roscioli').\n"
    "  event          — dated or schedulable occurrence (e.g. 'The user "
    "visited MoMA on 2023-06-15').\n"
    "  opinion        — speaker's subjective view without a concrete "
    "commitment.\n"
    "  mentioned      — a target was named but NOT explicitly recommended "
    "or endorsed (e.g. 'La Pergola was also discussed'). Use this when a\n"
    "  restaurant/book/place is referenced but the speaker did not make a\n"
    "  clear endorsement.\n"
    "  fact           — neutral factual statement that isn't any of the "
    "above (default fallback).\n\n"
    "EMPHASIS guide:\n"
    "  explicit  — speaker named it directly AND made it a focal point\n"
    "              (recommended X; user said they LOVE X; event pinpointed).\n"
    "  mentioned — named but not the focal point of the turn.\n"
    "  implied   — derivable from the turn but not literally stated.\n\n"
    "CONTENT RULES — the 'content' string itself:\n"
    "  1. PRESERVE literally: all proper nouns, dates, numbers, quantities, "
    "places, model names, prices, durations, colors, sizes, materials. "
    "Copy them verbatim from the turn.\n"
    "  2. REWRITE in third person from an outside observer's POV:\n"
    "       - User turn: 'The user ...', 'The user prefers ...'\n"
    "       - Assistant turn: 'The assistant told the user that ...', "
    "'The assistant recommended ...'\n"
    "     Never use 'I', 'you', 'we', 'my', 'your'.\n"
    "  3. RESOLVE every pronoun to its antecedent.\n"
    "  4. RESOLVE every relative time reference to an absolute date using "
    "the session_date as the anchor; write YYYY-MM-DD.\n"
    "     - For weekday-relative phrases ('last Friday', 'this Sunday', "
    "'next Monday'), step day-by-day from the anchor to the named weekday; "
    "do not default to ±7 days.\n"
    "     - For 'the WEEKDAY before/after DATE' (e.g. 'the Friday before "
    "July 15, 2023'), resolve relative to DATE, not to session_date: pick "
    "the first WEEKDAY strictly before/after DATE.\n"
    "     - For 'the week before X' / 'the weekend before X', resolve to "
    "the 7-day window (or Sat–Sun pair) immediately preceding X; if a "
    "single day is needed, prefer the matching weekday or X-7.\n"
    "     - When a relative phrase is ambiguous (no anchor in turn or "
    "session_date), KEEP the original phrase verbatim instead of guessing.\n"
    "  5. ONE FACT PER RECORD. If a turn has multiple distinct facts, "
    "emit multiple records.\n"
    "  6. NEVER FABRICATE. Only restate information literally in the turn.\n"
    "  7. Keep assistant-stated facts (recommendations, named entities, "
    "attributes, instructions, dated events). Skip ONLY pure pleasantries "
    "('thanks', 'ok'), generic puzzle answers with no user context, and "
    "off-topic trivia. When in doubt, KEEP.\n\n"
    "OUTPUT FORMAT:\n"
    "  - If the turn has ≥1 keep-worthy fact: emit a JSON array "
    "(starts with '[' and ends with ']'). No prose before or after. No "
    "code fences.\n"
    "  - If the turn is pure filler: emit exactly SKIP (nothing else).\n\n"
    "Worked examples:\n\n"
    "TURN: assistant: \"I'd recommend Roscioli for romantic Italian in "
    "Rome — it's a classic.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant recommended Roscioli for romantic Italian '
    'dinner in Rome.", "speaker": "assistant", "claim_type": '
    '"recommendation", "emphasis": "explicit", "entities": ["Roscioli", '
    '"Rome"], "topics": ["restaurant", "italian", "romantic", "dinner"]}\n'
    "]\n\n"
    "TURN: assistant: \"The Plesiosaur in the image has a blue scaly body "
    "and four flippers.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant told the user that the Plesiosaur in '
    'the image has a blue scaly body.", "speaker": "assistant", '
    '"claim_type": "fact", "emphasis": "explicit", "entities": '
    '["Plesiosaur"], "topics": ["animal", "color", "image"]},\n'
    '  {"content": "The assistant told the user that the Plesiosaur in '
    'the image has four flippers.", "speaker": "assistant", '
    '"claim_type": "fact", "emphasis": "explicit", "entities": '
    '["Plesiosaur"], "topics": ["animal", "anatomy", "image"]}\n'
    "]\n\n"
    "TURN: user: \"I prefer dark chocolate over milk chocolate and I "
    "can't stand cilantro.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The user prefers dark chocolate over milk chocolate.", '
    '"speaker": "user", "claim_type": "preference", "emphasis": '
    '"explicit", "entities": ["dark chocolate", "milk chocolate"], '
    '"topics": ["food", "chocolate"]},\n'
    '  {"content": "The user cannot stand cilantro.", "speaker": "user", '
    '"claim_type": "preference", "emphasis": "explicit", "entities": '
    '["cilantro"], "topics": ["food", "dislike"]}\n'
    "]\n\n"
    "TURN: user: \"I visited the Rijksmuseum in Amsterdam on June 5, "
    "2023.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The user visited the Rijksmuseum in Amsterdam on '
    '2023-06-05.", "speaker": "user", "claim_type": "event", "emphasis": '
    '"explicit", "entities": ["Rijksmuseum", "Amsterdam"], "topics": '
    '["museum", "travel"]}\n'
    "]\n\n"
    "TURN: assistant: \"Some options in Orlando for milkshakes include "
    "Toothsome Chocolate Emporium, but the Sugar Factory at Icon Park is "
    "the one famous for giant goblet shakes.\"\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant identified the Sugar Factory at Icon '
    'Park as the Orlando spot famous for giant goblet milkshakes.", '
    '"speaker": "assistant", "claim_type": "recommendation", "emphasis": '
    '"explicit", "entities": ["Sugar Factory", "Icon Park", "Orlando"], '
    '"topics": ["dessert", "milkshake", "orlando"]},\n'
    '  {"content": "The assistant mentioned Toothsome Chocolate Emporium '
    'as another Orlando milkshake option.", "speaker": "assistant", '
    '"claim_type": "mentioned", "emphasis": "mentioned", "entities": '
    '["Toothsome Chocolate Emporium", "Orlando"], "topics": ["dessert", '
    '"orlando"]}\n'
    "]\n\n"
    "TURN: user: \"I had my charity 5K race last Sunday — beat my time.\"\n"
    "  (session_date = 2023-05-22, a Monday; last Sunday = 2023-05-21)\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The user ran a charity 5K race on 2023-05-21.", '
    '"speaker": "user", "claim_type": "event", "emphasis": "explicit", '
    '"entities": ["charity 5K race"], "topics": ["running", "charity"]}\n'
    "]\n\n"
    "TURN: assistant: \"Don’t forget your pottery workshop the Friday "
    "before your trip on July 15, 2023.\"\n"
    "  (Friday before 2023-07-15 (Sat) = 2023-07-14; resolve relative to "
    "the named DATE, not session_date)\n"
    "OUTPUT:\n"
    "[\n"
    '  {"content": "The assistant reminded the user about a pottery '
    'workshop on 2023-07-14, the Friday before the user’s trip on '
    '2023-07-15.", "speaker": "assistant", "claim_type": "event", '
    '"emphasis": "explicit", "entities": ["pottery workshop"], '
    '"topics": ["pottery", "workshop", "reminder"]}\n'
    "]\n\n"
    "TURN: assistant: \"Great, happy to help!\"\n"
    "OUTPUT:\n"
    "SKIP\n\n"
    "Turn context:\n"
    "  role = {role}\n"
    "  session_date = {session_date}\n\n"
    "Turn content:\n"
    "{content}\n\n"
    "Output: a JSON array of structured records, or SKIP."
)


_DISTILL_SENTINEL_SKIP = "SKIP"
_DISTILL_LEGACY_LINE_RE = re.compile(r"^\s*[-*•]\s*(.+?)\s*$")


def _normalize_claim_type(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in _CLAIM_TYPES else "fact"


def _normalize_speaker(value: Any, *, default: str = "unknown") -> str:
    v = str(value or "").strip().lower()
    if v in _SPEAKERS:
        return v
    if v in ("u",):
        return "user"
    if v in ("a", "ai", "bot"):
        return "assistant"
    return default if default in _SPEAKERS else "unknown"


def _normalize_emphasis(value: Any) -> str:
    v = str(value or "").strip().lower()
    return v if v in _EMPHASIS_LEVELS else "mentioned"


def _normalize_str_list(value: Any) -> Tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[;,]", value) if p.strip()]
        return tuple(parts)
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return tuple(parts)
    return ()


def _fact_from_record(record: Any, *, fallback_speaker: str = "unknown") -> Optional[DistilledFact]:
    if not isinstance(record, dict):
        return None
    content = str(record.get("content") or "").strip()
    if not content:
        return None
    return DistilledFact(
        content=content,
        speaker=_normalize_speaker(record.get("speaker"), default=fallback_speaker),
        claim_type=_normalize_claim_type(record.get("claim_type")),
        emphasis=_normalize_emphasis(record.get("emphasis")),
        entities=_normalize_str_list(record.get("entities")),
        topics=_normalize_str_list(record.get("topics")),
    )


def _extract_json_array(text: str) -> Optional[list]:
    """Try hard to pull a JSON array out of the LLM output.

    Handles: bare arrays, arrays wrapped in code fences, or arrays embedded
    inside preamble prose. Returns None if nothing parses.
    """
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", stripped)
        stripped = re.sub(r"\n?```\s*$", "", stripped).strip()
    # Greedy grab between first [ and last ] (tolerates preamble/epilogue).
    start = stripped.find("[")
    end = stripped.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = stripped[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


def _parse_distilled(
    raw: str, *, fallback_speaker: str = "unknown"
) -> List[DistilledFact]:
    """Parse the distiller's output into structured ``DistilledFact`` records.

    Preferred format: JSON array of records with the schema documented in
    ``DISTILL_PROMPT``. Falls back to legacy bullet-line prose so older
    outputs (or degraded LLM responses) still yield usable memories with
    sensible defaults (speaker inferred from the turn's role, claim_type
    ``fact``, emphasis ``mentioned``).

    Returns ``[]`` when the distiller said SKIP or produced nothing.
    """
    if not raw:
        return []
    text = raw.strip()
    if text.upper() == _DISTILL_SENTINEL_SKIP:
        return []

    # Strip a surrounding code fence before any other detection.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text).strip()
        if text.upper() == _DISTILL_SENTINEL_SKIP:
            return []

    # Preferred path: structured JSON array.
    records = _extract_json_array(text)
    if records is not None:
        facts: list[DistilledFact] = []
        for r in records:
            f = _fact_from_record(r, fallback_speaker=fallback_speaker)
            if f is not None:
                facts.append(f)
        return facts

    # Legacy fallback: bullet-line prose. Each line becomes a minimal fact
    # with speaker inferred from the caller and everything else defaulted.
    facts = []
    for line in text.splitlines():
        m = _DISTILL_LEGACY_LINE_RE.match(line)
        if not m:
            continue
        content = m.group(1).strip()
        if not content or content.upper() == _DISTILL_SENTINEL_SKIP:
            continue
        facts.append(
            DistilledFact(
                content=content,
                speaker=fallback_speaker
                if fallback_speaker in _SPEAKERS
                else "unknown",
                claim_type="fact",
                emphasis="mentioned",
            )
        )
    return facts


def distill_turn(
    *,
    role: str,
    content: str,
    session_date: str,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    max_tokens: int = 3000,
) -> List[DistilledFact]:
    """Run one turn through the distillation LLM; return structured facts.

    An empty list means the turn produced no durable memory worth keeping
    (pleasantries, generic puzzle answers, etc.). The caller should NOT
    store anything for those turns.

    Returns structured ``DistilledFact`` records. On LLM failure (timeout,
    402, etc.) falls back to a single minimal record carrying the raw turn
    so downstream retrieval still has something to latch onto.
    """
    text = (content or "").strip()
    if not text:
        return []
    if model is None:
        from attestor.config import get_stack
        model = get_stack().models.distill
    fallback_speaker = _normalize_speaker(role)
    # NOTE: str.replace (not str.format) — the prompt contains JSON worked
    # examples whose literal '{...}' would otherwise be misread as format
    # fields. Only three known placeholders are substituted.
    prompt = (
        DISTILL_PROMPT
        .replace("{role}", role or "unknown")
        .replace("{session_date}", session_date or "(unknown)")
        .replace("{content}", text)
    )
    try:
        client = _get_client(api_key)
        raw = _chat(client, model, prompt, max_tokens=max_tokens, role="distill")
    except Exception as e:  # noqa: BLE001 — distillation is best-effort
        logger.warning("distill_turn failed (%s); falling back to raw turn", e)
        return [
            DistilledFact(
                content=f"[{session_date}] {role}: {text}",
                speaker=fallback_speaker,
                claim_type="fact",
                emphasis="mentioned",
            )
        ]
    return _parse_distilled(raw, fallback_speaker=fallback_speaker)


def ingest_history(
    mem: Any,
    sample: LMESample,
    *,
    use_extraction: bool = False,
    extraction_model: Optional[str] = None,
    use_distillation: bool = False,
    distill_model: Optional[str] = None,
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
    if extraction_model is None or distill_model is None:
        from attestor.config import get_stack
        s = get_stack()
        if extraction_model is None:
            extraction_model = s.models.extraction
        if distill_model is None:
            distill_model = s.models.distill

    ns = namespace_for(sample)
    turns_seen = 0
    memories_added = 0
    skipped_empty = 0
    distilled_facts = 0
    skipped_by_distiller = 0

    sessions = list(
        zip(sample.haystack_session_ids, sample.haystack_dates, sample.haystack_sessions)
    )

    if use_distillation:
        # Per-turn LLM distillation — each turn → 0..N canonical facts.
        for session_id, session_date, turns in sessions:
            iso = _iso_date(session_date)
            short = _short_date(session_date)
            for turn_idx, turn in enumerate(turns):
                turns_seen += 1
                text = (turn.content or "").strip()
                if not text:
                    skipped_empty += 1
                    continue
                if verbose:
                    print(f"    distill {session_id}#{turn_idx} role={turn.role}")
                facts = distill_turn(
                    role=turn.role,
                    content=text,
                    session_date=short,
                    model=distill_model,
                    api_key=api_key,
                )
                if not facts:
                    skipped_by_distiller += 1
                    continue
                for fact_idx, fact in enumerate(facts):
                    distilled_facts += 1
                    # Still prefix with date for belt-and-suspenders — the
                    # distiller SHOULD have resolved dates, but if it drifted,
                    # the inline prefix is a safety net.
                    content_with_date = fact.content
                    if short and short not in fact.content:
                        content_with_date = f"[{short}] {fact.content}"
                    # v3-ablate-A: category/entity/tags identical to v2.
                    # Structured fields live ONLY in metadata.jsonb (inert
                    # to the retrieval pipeline). This isolates whether
                    # structured extraction by itself regresses anything.
                    mem.add(
                        content=content_with_date,
                        tags=[turn.role or "unknown", session_id, "lme", "distilled"],
                        category="fact",
                        entity=None,
                        namespace=ns,
                        event_date=iso,
                        metadata={
                            "session_id": session_id,
                            "role": turn.role,
                            "turn_idx": turn_idx,
                            "fact_idx": fact_idx,
                            "source": "lme_distilled",
                            "distill_model": distill_model,
                            # Structured fields — stored but not surfaced
                            # in retrieval or answer context for ablation-A.
                            "speaker": fact.speaker,
                            "claim_type": fact.claim_type,
                            "emphasis": fact.emphasis,
                            "entities": list(fact.entities),
                            "topics": list(fact.topics),
                        },
                    )
                    memories_added += 1
        return IngestStats(
            turns_seen=turns_seen,
            memories_added=memories_added,
            sessions=len(sessions),
            skipped_empty=skipped_empty,
            distilled_facts=distilled_facts,
            skipped_by_distiller=skipped_by_distiller,
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


# ---------------------------------------------------------------------------
# Answer + judge
# ---------------------------------------------------------------------------

ANSWER_PROMPT = (
    "You are answering a question based on an assistant's memory of a "
    "past chat history. The facts below are the user's memory — each is "
    "prefixed with [YYYY-MM-DD] for the session date when the turn was "
    "recorded.\n\n"
    "DECIDE THE QUESTION MODE FIRST (this governs everything else):\n"
    "  FACT mode — the question has a single correct answer the user "
    "or assistant previously stated. Hallmarks: when / where / who / "
    "what color / what (specific) / how many / how long / which "
    "(specific) / did I / have I / remind me what. Answer with the "
    "concrete value from the facts. If the facts don't support the "
    "answer, respond exactly: I don't know.\n\n"
    "  RECOMMENDATION mode — the question asks for tips, advice, "
    "ideas, suggestions, or a tailored proposal. Hallmarks: recommend / "
    "suggest / propose / advise / help me pick / any tips / any advice / "
    "any ideas / what should I / what would you / I'm looking for / I "
    "need help with. For these you MUST NOT respond \"I don't know\" — "
    "use the user's stored preferences, habits, constraints, and past "
    "experiences to produce a concrete, user-specific proposal. You may "
    "blend stored facts with sensible domain knowledge, but the "
    "response must be obviously tailored to THIS user (cite relevant "
    "stored facts in parentheses). Generic boilerplate is WRONG.\n\n"
    "EDGE CASES:\n"
    "  - \"Can you remind me what color …\" → FACT (recall)\n"
    "  - \"What's my favorite …\" → FACT (recall of a stated preference)\n"
    "  - \"Can you recommend a hotel for …\" → RECOMMENDATION\n"
    "  - \"Any tips for keeping my kitchen clean?\" → RECOMMENDATION\n\n"
    "DATE ARITHMETIC — applies to FACT mode questions about durations, "
    "gaps, or event dates:\n"
    "  1. Identify every date relevant to the question (event dates, "
    "the question date, reference dates in the question itself).\n"
    "  2. \"how many days/weeks/months/years between X and Y\" → "
    "     compute |date_Y − date_X| in the requested unit. Measure\n"
    "     BETWEEN the two events named in the question, not \"from now\".\n"
    "  3. \"how many days/months ago\" → anchor is the question_date, "
    "     not today.\n"
    "  4. \"when I did X\", \"when I made the cake\" — that clause IS\n"
    "     the anchor; do not substitute the question_date.\n"
    "  5. Months: full calendar months. 2022-10-22 → 2023-03-22 = 5.\n"
    "  6. Days: calendar count; verify month-by-month.\n\n"
    "Before your final answer, think step-by-step inside "
    "<reasoning>...</reasoning>. Include the mode decision (FACT vs "
    "RECOMMENDATION), the relevant facts you'll cite, and any "
    "arithmetic. Then output the final answer AFTER </reasoning> on "
    "its own — concise and concrete.\n\n"
    "WORKED EXAMPLES (showing both modes):\n\n"
    "Example A — FACT, temporal:\n"
    "  Question (asked on 2023-07-20): How many days between my visit\n"
    "  to the museum and the concert?\n"
    "  Facts:\n"
    "    - [2023-06-05] The user visited the Rijksmuseum in Amsterdam.\n"
    "    - [2023-06-17] The user attended a jazz concert at Bimhuis.\n"
    "  <reasoning>\n"
    "  Mode: FACT (how many days between X and Y).\n"
    "  Dates: 2023-06-05 (museum) and 2023-06-17 (concert).\n"
    "  Diff: 2023-06-17 − 2023-06-05 = 12 days.\n"
    "  </reasoning>\n"
    "  12 days.\n\n"
    "Example B — FACT, recall (single-session-assistant):\n"
    "  Question: What color was the Plesiosaur the assistant described?\n"
    "  Facts:\n"
    "    - [2023-04-10] The assistant told the user the Plesiosaur in\n"
    "      the image has a blue scaly body and four flippers.\n"
    "  <reasoning>\n"
    "  Mode: FACT (what color — one correct value).\n"
    "  Fact [2023-04-10] states the Plesiosaur had a blue scaly body.\n"
    "  </reasoning>\n"
    "  Blue.\n\n"
    "Example C — RECOMMENDATION, hotel:\n"
    "  Question (asked on 2024-03-10): Can you suggest a hotel for my\n"
    "  trip to Lisbon?\n"
    "  Facts:\n"
    "    - [2022-04-12] The user stayed at Casa das Janelas com Vista\n"
    "      in Lisbon and enjoyed it.\n"
    "    - [2023-09-03] The user said they prefer boutique hotels over\n"
    "      chains and love rooftop views of the water.\n"
    "  <reasoning>\n"
    "  Mode: RECOMMENDATION. User preferences: boutique + rooftop water\n"
    "  views. Past win: Casa das Janelas com Vista. Combine with\n"
    "  domain knowledge to propose concrete boutique options with water\n"
    "  views.\n"
    "  </reasoning>\n"
    "  Based on your preference for boutique hotels with rooftop water\n"
    "  views (per 2023-09-03):\n"
    "  - Memmo Alfama — boutique, rooftop plunge pool over the Tagus.\n"
    "  - Santiago de Alfama — boutique, Alfama rooftop, 32 rooms.\n"
    "  - Casa das Janelas com Vista — you already enjoyed this in\n"
    "    2022-04-12 and it still fits your stated preferences.\n\n"
    "Example D — RECOMMENDATION, kitchen tips:\n"
    "  Question: Any tips for keeping my kitchen organized?\n"
    "  Facts:\n"
    "    - [2024-01-14] The user bought a magnetic knife strip.\n"
    "    - [2024-02-02] The user said they dislike countertop clutter.\n"
    "    - [2023-11-10] The user has granite countertops.\n"
    "  <reasoning>\n"
    "  Mode: RECOMMENDATION. Build on existing tools (magnetic knife\n"
    "  strip), respect granite (no vinegar/lemon), target declutter.\n"
    "  </reasoning>\n"
    "  Build on what you already have:\n"
    "  - Move cutting boards vertical beside the magnetic knife strip\n"
    "    you installed (per 2024-01-14) to keep counters clear.\n"
    "  - Use pH-neutral cleaner on granite; avoid vinegar or lemon.\n"
    "  - Add a two-tier pull-out under the sink to corral bottles —\n"
    "    directly addresses your dislike of countertop clutter\n"
    "    (per 2024-02-02).\n\n"
    "Now answer the real question.\n\n"
    "Question (asked on {question_date}):\n{question}\n\n"
    "Facts:\n{context}\n\n"
    "Output:\n"
    "<reasoning>...your mode decision + cited facts + arithmetic...</reasoning>\n"
    "<final answer only>"
)


# ------------------------------------------------------------------
# The following three symbols are retained as deprecated no-op
# placeholders so external callers / tests import cleanly. The
# unified ANSWER_PROMPT now handles both modes; no separate
# PERSONALIZATION_PROMPT or classifier is needed.
# ------------------------------------------------------------------

PERSONALIZATION_PROMPT = ANSWER_PROMPT  # unified into ANSWER_PROMPT


def is_recommendation_question(question: str, **_: Any) -> bool:
    """Deprecated: the unified ANSWER_PROMPT decides mode internally.

    Kept only so existing imports don't break. Always returns False now;
    the LLM handles mode selection inside a single prompt.
    """
    return False


def classify_question(question: str, **_: Any) -> int:
    """Deprecated: no separate classifier — see unified ANSWER_PROMPT."""
    return 0


VERIFY_PROMPT = (
    "Double-check the AI's answer below against the facts. You are a "
    "second-pass verifier. Your job is to catch date-arithmetic mistakes, "
    "misread temporal anchors, and questions where the AI abstained "
    "despite sufficient evidence.\n\n"
    "Question (asked on {question_date}):\n{question}\n\n"
    "Facts:\n{context}\n\n"
    "AI's first answer:\n{first_answer}\n\n"
    "Rules:\n"
    "  1. If the first answer is correct, repeat it verbatim (same number, "
    "same date, same phrase). Do NOT rephrase or elaborate.\n"
    "  2. If the first answer has an arithmetic error, recompute and "
    "output the corrected final answer.\n"
    "  3. If the first answer abstained (\"I don't know\") but the facts "
    "contain a specific answer, replace the abstention with the correct "
    "answer.\n"
    "  4. If the facts truly do not support an answer, keep \"I don't know\".\n\n"
    "Output ONLY the final answer on one line. No reasoning, no prefix, "
    "no explanation."
)

JUDGE_PROMPT = (
    "You are judging whether an AI's answer is CORRECT or WRONG given the "
    "gold answer. Be generous — if the AI's answer semantically matches or "
    "contains the gold answer, mark CORRECT. For dates, accept equivalent "
    "formats.\n\n"
    "Category-specific rubric (question category: {category}):\n"
    "  - temporal-reasoning: the answer must include the correct date or "
    "period; if the AI paraphrases a relative phrase (\"last year\") instead "
    "of resolving it, mark WRONG.\n"
    "  - knowledge-update: the answer must reflect the LATEST state, not an "
    "older superseded value; WRONG if stale.\n"
    "  - abstention: if the gold answer is an abstention (e.g. \"I don't "
    "know\", \"not mentioned\"), accept any reasonable abstention; hallucinated "
    "facts are WRONG.\n"
    "  - Other categories: match gold answer on substance, not wording.\n\n"
    "Question: {question}\n"
    "Gold answer: {expected}\n"
    "AI answer: {generated}\n\n"
    "Return JSON with keys \"reasoning\" (one sentence) and \"label\" "
    "(CORRECT or WRONG)."
)


def _get_client(api_key: Optional[str] = None) -> Any:
    """Instantiate an OpenAI-compatible client.

    Resolution order (top wins):
      1. ``LME_LLM_BASE_URL`` env — explicit ad-hoc override; mostly
         used to point at local Ollama (``http://localhost:11434/v1``).
         When the URL points at localhost, the API key is optional.
      2. ``configs/attestor.yaml`` ``stack.llm.provider`` — picks the
         base URL and the env var name for the API key. Two providers
         wired today: ``openrouter`` (default; multi-vendor model names
         like ``anthropic/claude-sonnet-4-6``) and ``openai`` (no
         OpenRouter markup, OpenAI models only).
      3. Hardcoded fallback to OpenRouter — only if YAML is missing.

    The chosen provider's API key env var is required unless the base
    URL is local. A clear error names the env var that was missing
    rather than the generic "OPENROUTER_API_KEY".
    """
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover — import-time error path
        raise RuntimeError(
            "openai package required for benchmarks. Install with "
            "`poetry add --group dev openai`."
        ) from e

    # 2 — resolve provider via stack
    try:
        from attestor.config import get_stack
        llm_cfg = get_stack().llm
    except Exception:
        # Stack load can fail in stripped-checkout / no-YAML test scenarios;
        # fall back to the legacy OpenRouter shape so existing callers
        # don't crash before they even get to the API call.
        llm_cfg = None

    # 1 — explicit env override wins
    env_base_url = os.environ.get("LME_LLM_BASE_URL")
    if env_base_url:
        base_url = env_base_url
        # When user set the URL explicitly, default the key env to the
        # configured provider's; the request itself will surface a 401
        # if the key is mismatched.
        key_env = (llm_cfg.api_key_env if llm_cfg else "OPENROUTER_API_KEY")
    elif llm_cfg is not None:
        base_url = llm_cfg.base_url
        key_env = llm_cfg.api_key_env
    else:
        base_url = OPENROUTER_BASE_URL
        key_env = "OPENROUTER_API_KEY"

    is_local = "localhost" in base_url or "127.0.0.1" in base_url

    key = api_key or os.environ.get(key_env)
    if not key:
        if is_local:
            key = "ollama"  # placeholder; Ollama ignores the key
        else:
            provider_name = (llm_cfg.provider if llm_cfg else "openrouter")
            raise RuntimeError(
                f"{key_env} not set — required for LongMemEval "
                f"answer/judge against {base_url} "
                f"(llm.provider={provider_name!r}). Either export "
                f"{key_env}, switch llm.provider in configs/attestor.yaml, "
                f"or set LME_LLM_BASE_URL=http://localhost:11434/v1 to "
                f"run against local Ollama instead."
            )
    return OpenAI(base_url=base_url, api_key=key)


def _chat(
    client: Any,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 300,
    reasoning_effort: Optional[str] = None,
    role: Optional[str] = None,
) -> str:
    """One-shot chat completion; returns content text.

    When ``role`` is provided, look up per-role overrides for
    ``reasoning_effort`` and ``max_tokens`` from the YAML stack
    (``models.reasoning_effort[role]`` and ``models.max_tokens[role]``).
    Explicit kwargs win over YAML; YAML wins over the legacy default.

    ``reasoning_effort`` is a gpt-5.x param. Models that don't support
    it ignore it silently via OpenRouter's API surface; no need to
    filter client-side.
    """
    if role is not None:
        from attestor.config import chat_kwargs_for_role
        role_kwargs = chat_kwargs_for_role(role, fallback_max_tokens=max_tokens)
        # Explicit caller args override YAML
        if max_tokens != 300:  # legacy sentinel — caller passed an explicit value
            role_kwargs["max_tokens"] = max_tokens
        else:
            max_tokens = role_kwargs["max_tokens"]
        if reasoning_effort is None and "reasoning_effort" in role_kwargs:
            reasoning_effort = role_kwargs["reasoning_effort"]

    create_kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort:
        create_kwargs["reasoning_effort"] = reasoning_effort

    from attestor.llm_trace import traced_create
    response = traced_create(client, role=role or "lme.chat", **create_kwargs)
    return response.choices[0].message.content or ""


def _format_recall_context(results: List[Any], max_facts: int = 40) -> str:
    """Join top retrieval hits into a plain newline-delimited context block.

    v3-ablate-A note: structured tags (speaker / claim_type / emphasis)
    live in ``metadata.jsonb`` but are NOT surfaced in the answerer's
    context. This matches the v2 format exactly so ablation-A isolates
    the distillation change alone.
    """
    lines: list[str] = []
    for r in results[:max_facts]:
        mem_obj = getattr(r, "memory", None) or r
        content = getattr(mem_obj, "content", str(mem_obj))
        lines.append(f"- {content}")
    return "\n".join(lines)


@dataclass(frozen=True)
class AnswerResult:
    """Output of ``answer_question`` — answer text plus retrieval diagnostics."""

    answer: str
    retrieved_count: int
    used_fact_count: int
    latency_ms: float
    reasoning: str = ""  # the <reasoning> block if the answerer produced one
    verified: bool = False  # True when a verification pass ran
    raw_first_answer: str = ""  # set when verification overrode the first answer
    # Dimension-B telemetry — filled by answer_question for later scoring.
    retrieved_session_ids: Tuple[str, ...] = ()
    predicted_mode: str = ""  # "fact" | "recommendation" | "" (unknown)
    context: str = ""          # the formatted recall context the answerer saw


_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.DOTALL | re.IGNORECASE)

# Mode-tag extraction from the reasoning block. Recognizes both
# "Mode: FACT" / "Mode: RECOMMENDATION" and the less-structured
# "FACT mode" / "RECOMMENDATION mode" patterns the prompt invites.
_MODE_RE = re.compile(
    r"\b(?:mode\s*[:=]?\s*)?(fact|recommendation)(?:\s+mode)?\b",
    re.IGNORECASE,
)


def _parse_predicted_mode(reasoning: str) -> str:
    """Extract the model's chosen mode from its reasoning block.

    Returns "fact", "recommendation", or "" (unknown). Prefers the FIRST
    clear mode token since the answerer declares mode up front per the
    prompt.
    """
    if not reasoning:
        return ""
    m = _MODE_RE.search(reasoning)
    if not m:
        return ""
    return m.group(1).lower()


def _extract_retrieved_session_ids(results: list) -> Tuple[str, ...]:
    """Pull session_ids from retrieval results, tolerating ducktyped shapes.

    Memories written by our ingest carry ``metadata["session_id"]``. Older
    test fixtures may lack the metadata dict. Missing values are dropped,
    not substituted.
    """
    out: list[str] = []
    for r in results:
        mem_obj = getattr(r, "memory", None) or r
        meta = getattr(mem_obj, "metadata", None) or {}
        sid = None
        if isinstance(meta, dict):
            sid = meta.get("session_id")
        if sid:
            out.append(str(sid))
    return tuple(out)


def _strip_reasoning(raw: str) -> Tuple[str, str]:
    """Split an answerer response into (reasoning, final_answer).

    Supports the <reasoning>...</reasoning> then final-answer contract.
    If no tags are present, treats the whole string as the final answer.
    """
    if not raw:
        return "", ""
    m = _REASONING_RE.search(raw)
    if not m:
        return "", raw.strip()
    reasoning = m.group(1).strip()
    after = raw[m.end():].strip()
    # Some models wrap the final answer in ticks / extra prose — take the
    # first non-empty line after the reasoning block.
    for line in after.splitlines():
        line = line.strip().strip("`").strip()
        if line:
            return reasoning, line
    return reasoning, after


def answer_question(
    mem: Any,
    sample: LMESample,
    *,
    budget: int = 4000,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_facts: int = 40,
    max_tokens: int = 1200,
    verify: bool = False,
    verify_model: Optional[str] = None,
) -> AnswerResult:
    """Recall + synthesize an answer for a LongMemEval sample.

    Args:
        mem: ``AgentMemory`` instance, already populated via ``ingest_history``.
        sample: ``LMESample`` being answered.
        budget: Retrieval token budget for ``mem.recall``.
        model: OpenRouter model id for synthesis.
        api_key: Optional override for ``OPENROUTER_API_KEY``.
        max_facts: Cap on facts injected into the prompt (guards prompt size).
        max_tokens: Answerer ``max_tokens``. Bumped to 600 to accommodate the
            chain-of-thought <reasoning> block.
        verify: When True, run a second-pass verification that re-checks
            the first answer against the same facts. Catches arithmetic
            errors and over-abstention.
        verify_model: OpenRouter model id for the verifier. Defaults to
            ``model`` (same model self-verifying).

    Returns:
        ``AnswerResult`` with final answer + retrieval counts + latency +
        optional reasoning trace + verification flag.
    """
    import time

    ns = namespace_for(sample)
    t0 = time.monotonic()
    results = mem.recall(sample.question, budget=budget, namespace=ns) or []
    results = sorted(results, key=lambda r: getattr(r, "score", 0.0), reverse=True)

    retrieved_session_ids = _extract_retrieved_session_ids(results[:max_facts])

    if not results:
        return AnswerResult(
            answer="I don't know.",
            retrieved_count=0,
            used_fact_count=0,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
            retrieved_session_ids=retrieved_session_ids,
        )

    context = _format_recall_context(results, max_facts=max_facts)
    question_date = sample.question_date or "(unknown)"

    # Unified prompt: the model picks FACT vs RECOMMENDATION mode inside the
    # prompt (no separate classifier). Fact mode is strict; recommendation
    # mode weaves stored user preferences into a tailored proposal. See
    # ANSWER_PROMPT for the mode-decision rubric + worked examples.
    prompt = ANSWER_PROMPT.format(
        question=sample.question,
        question_date=question_date,
        context=context,
    )
    client = _get_client(api_key)
    raw = _chat(client, model, prompt, max_tokens=max_tokens, role="answerer").strip()
    reasoning, first_answer = _strip_reasoning(raw)

    final_answer = first_answer
    verified = False
    raw_first = ""
    if verify:
        verify_prompt = VERIFY_PROMPT.format(
            question=sample.question,
            question_date=question_date,
            context=context,
            first_answer=first_answer,
        )
        verified_text = _chat(
            client, verify_model or model, verify_prompt,
            max_tokens=150, role="verifier",
        ).strip()
        # Only accept the verifier's output if it is a non-empty single line
        # that differs from the first answer. Preserve the verified=True flag
        # either way so telemetry records that the pass ran.
        verified = True
        cleaned = verified_text.splitlines()[0].strip() if verified_text else ""
        if cleaned:
            if cleaned != first_answer:
                raw_first = first_answer
                final_answer = cleaned
            else:
                final_answer = cleaned

    return AnswerResult(
        answer=final_answer,
        retrieved_count=len(results),
        used_fact_count=min(len(results), max_facts),
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
        reasoning=reasoning,
        verified=verified,
        raw_first_answer=raw_first,
        retrieved_session_ids=retrieved_session_ids,
        predicted_mode=_parse_predicted_mode(reasoning),
        context=context,
    )


# Robust label extraction — works on clean JSON, malformed JSON, or plain text.
_LABEL_FALLBACK_RE = re.compile(r"\b(CORRECT|WRONG)\b", re.IGNORECASE)


def _parse_judge_response(raw: str) -> Tuple[str, str]:
    """Parse a judge response into ``(label, reasoning)``.

    Strategy:
      1. Try strict JSON parse.
      2. Extract JSON blob between the first ``{`` and last ``}`` and retry.
      3. Fall back to regex over the raw text; prefer the LAST label mention
         so trailing verdicts override in-reasoning quotations.

    Defaults to ``("WRONG", raw)`` if nothing matches — bias is conservative
    so bad judge output never inflates accuracy.
    """
    if not raw or not raw.strip():
        return "WRONG", ""
    text = raw.strip()

    # Strip markdown code fences.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", text)
        text = re.sub(r"\n?```\s*$", "", text)

    candidates = [text]
    lb, rb = text.find("{"), text.rfind("}")
    if 0 <= lb < rb:
        candidates.append(text[lb : rb + 1])

    for blob in candidates:
        try:
            obj = json.loads(blob)
        except (json.JSONDecodeError, TypeError):
            continue
        if isinstance(obj, dict):
            label = str(obj.get("label", "")).strip().upper()
            reasoning = str(obj.get("reasoning", "")).strip()
            if label in {"CORRECT", "WRONG"}:
                return label, reasoning

    matches = _LABEL_FALLBACK_RE.findall(text)
    if matches:
        return matches[-1].upper(), text
    return "WRONG", text


@dataclass(frozen=True)
class JudgeResult:
    """Output of ``judge_answer`` — normalized label + reasoning + raw."""

    label: str  # "CORRECT" | "WRONG"
    correct: bool
    reasoning: str
    raw: str
    judge_model: str


PERSONALIZATION_JUDGE_PROMPT = (
    "You are scoring the PERSONALIZATION QUALITY of an AI's answer to a "
    "recommendation question. The user asked for advice / tips / "
    "suggestions, and the AI gave a tailored answer. Your job is to "
    "decide whether the answer is sufficiently personalized to THIS "
    "user's stored memory.\n\n"
    "Inputs:\n"
    "  Question: {question}\n"
    "  Stored user facts available to the AI:\n{context}\n"
    "  AI answer: {generated}\n"
    "  Reference 'preferred response' specification: {expected}\n\n"
    "Score CRITERIA — both must be true for label CORRECT:\n"
    "  (a) The AI cites or references at least one stored user fact "
    "      (preference, habit, past experience, constraint, named entity)\n"
    "      that is relevant to the question.\n"
    "  (b) The AI's recommendations are concrete (named items, specific "
    "      methods) and align with the reference specification's intent. "
    "      Generic boilerplate ('focus on a few key habits') without "
    "      user-specific tailoring is WRONG.\n\n"
    "Edge cases:\n"
    "  - If the AI abstains ('I don't know') for a recommendation "
    "    question, that is WRONG.\n"
    "  - If the stored facts don't support strong tailoring AND the AI "
    "    politely admits limited context while still giving a useful "
    "    starting list, that is CORRECT.\n"
    "  - Citing a stored fact that contradicts the question's intent is "
    "    WRONG.\n\n"
    "Return JSON with keys \"reasoning\" (one sentence) and \"label\" "
    "(CORRECT or WRONG)."
)


def judge_personalization(
    question: str,
    expected: str,
    generated: str,
    context: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = 300,
) -> JudgeResult:
    """LLM judge for personalization quality on RECOMMENDATION-mode samples.

    Same shape as ``judge_answer`` so reporting paths can treat them
    uniformly. Robust JSON parsing — bad output defaults to WRONG so
    bad judge output never inflates the personalization score.
    """
    prompt = PERSONALIZATION_JUDGE_PROMPT.format(
        question=question,
        expected=expected,
        generated=generated,
        context=context,
    )
    client = _get_client(api_key)
    raw = _chat(client, model, prompt, max_tokens=max_tokens, role="judge")
    label, reasoning = _parse_judge_response(raw)
    return JudgeResult(
        label=label,
        correct=label == "CORRECT",
        reasoning=reasoning,
        raw=raw,
        judge_model=f"{model}__personalization",
    )


def judge_answer(
    question: str,
    expected: str,
    generated: str,
    category: str,
    *,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_tokens: int = 300,
) -> JudgeResult:
    """Use an LLM to score an AI answer against the gold answer.

    Robust against JSON drift — always returns a well-formed ``JudgeResult``.
    """
    prompt = JUDGE_PROMPT.format(
        category=category,
        question=question,
        expected=expected,
        generated=generated,
    )
    client = _get_client(api_key)
    raw = _chat(client, model, prompt, max_tokens=max_tokens, role="judge")
    label, reasoning = _parse_judge_response(raw)
    return JudgeResult(
        label=label,
        correct=label == "CORRECT",
        reasoning=reasoning,
        raw=raw,
        judge_model=model,
    )


# ---------------------------------------------------------------------------
# Runner + reporting
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
    gold_session_ids: Tuple[str, ...] = ()       # from sample.answer_session_ids
    retrieved_session_ids: Tuple[str, ...] = ()  # from AnswerResult
    retrieval_hit: bool = False                  # any overlap between gold & retrieved
    retrieval_overlap: int = 0                   # count of overlapping sessions
    predicted_mode: str = ""                     # "fact" | "recommendation" | ""
    personalization: Optional[dict] = None       # JudgeResult dict, only on RECOMMENDATION samples


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
    argv: Tuple[str, ...]
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
    judge_models: Tuple[str, ...]
    by_category: dict  # category -> {judge_model -> {correct, total, accuracy}}
    by_judge: dict     # judge_model -> {correct, total, accuracy}
    started_at: str
    completed_at: str
    samples: Tuple[SampleReport, ...]
    provenance: Optional[RunProvenance] = None
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


def _blank_counter() -> dict:
    return {"correct": 0, "total": 0}


def _judgement_to_dict(j: JudgeResult) -> dict:
    return {
        "label": j.label,
        "correct": j.correct,
        "reasoning": j.reasoning,
        "judge_model": j.judge_model,
    }


def _accuracy(bucket: dict) -> dict:
    """Add a percentage accuracy field to a {correct,total} bucket."""
    total = bucket.get("total", 0)
    correct = bucket.get("correct", 0)
    pct = round(100.0 * correct / total, 2) if total else 0.0
    return {**bucket, "accuracy": pct}


def _summarize(
    sample_reports: List[SampleReport], judge_models: List[str]
) -> Tuple[dict, dict]:
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


def _summarize_dimensions(sample_reports: List[SampleReport]) -> dict:
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


def _git_sha() -> Tuple[str, bool]:
    """Return (sha, dirty). ``sha='unknown'`` when attestor is not in a git tree."""
    try:
        sha = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "-C", str(Path(__file__).resolve().parent), "status", "--porcelain"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return sha, bool(status.strip())
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


def _safe_judge_dict(
    jm: str, result: "JudgeResult | BaseException"
) -> dict:
    """Normalize one judge outcome — JudgeResult or an exception — into a dict.

    Errors do NOT inflate accuracy: they are recorded as WRONG with a reason.
    """
    if isinstance(result, BaseException):
        return {
            "label": "WRONG",
            "correct": False,
            "reasoning": f"judge_error: {type(result).__name__}: {result}",
            "judge_model": jm,
        }
    return _judgement_to_dict(result)


async def _process_sample(
    sample: LMESample,
    *,
    mem_factory: Callable[[], Any],
    answer_model: str,
    judge_models: List[str],
    api_key: Optional[str],
    budget: int,
    use_extraction: bool,
    max_facts: int,
    use_distillation: bool = False,
    distill_model: Optional[str] = None,
    verify: bool = False,
    verify_model: Optional[str] = None,
) -> SampleReport:
    """Ingest → answer → judge one sample on its own AgentMemory instance.

    Isolation: each call creates its own ``AgentMemory`` via ``mem_factory``
    and closes it at the end. Per-sample namespace (``lme_<qid>``) keeps
    haystacks disjoint at the document layer; per-instance connections
    keep the Postgres driver's thread-unsafe connection objects disjoint
    at the transport layer. Both together give us correctness under
    per-sample concurrency.

    Judge calls inside a sample run in parallel via ``asyncio.gather`` —
    independent network round-trips, no shared state.

    A failing judge is recorded as WRONG but does NOT fail the sample.
    A failing ingest/answer re-raises so the top-level gather can count it.
    """
    mem = await asyncio.to_thread(mem_factory)
    try:
        stats: IngestStats = await asyncio.to_thread(
            ingest_history,
            mem,
            sample,
            use_extraction=use_extraction,
            use_distillation=use_distillation,
            distill_model=distill_model,
            api_key=api_key,
            verbose=False,
        )
        ans: AnswerResult = await asyncio.to_thread(
            answer_question,
            mem,
            sample,
            budget=budget,
            model=answer_model,
            api_key=api_key,
            max_facts=max_facts,
            verify=verify,
            verify_model=verify_model,
        )

        judge_coros = [
            asyncio.to_thread(
                judge_answer,
                sample.question,
                sample.answer,
                ans.answer,
                sample.question_type,
                model=jm,
                api_key=api_key,
            )
            for jm in judge_models
        ]
        judge_results = await asyncio.gather(*judge_coros, return_exceptions=True)
        judgments = {
            jm: _safe_judge_dict(jm, res)
            for jm, res in zip(judge_models, judge_results)
        }

        # Dimension B — multi-dimensional scoring computed inline so the
        # per-sample report is self-describing and post-hoc analysis doesn't
        # need to re-run anything.
        gold_sessions = tuple(sample.answer_session_ids)
        retrieved_sessions = ans.retrieved_session_ids
        gold_set = set(gold_sessions)
        overlap = sum(1 for s in retrieved_sessions if s in gold_set)
        retrieval_hit = overlap > 0
        predicted_mode = ans.predicted_mode

        # Personalization judge — only on samples the answerer claims are
        # RECOMMENDATION mode. One judge call per sample (not per
        # judge_model) — cheap; uses the first configured judge model.
        personalization_dict: Optional[dict] = None
        if predicted_mode == "recommendation" and judge_models:
            judge_for_pers = judge_models[0]
            try:
                pj = await asyncio.to_thread(
                    judge_personalization,
                    sample.question,
                    sample.answer,
                    ans.answer,
                    ans.context,  # reuse the formatted context the answerer saw
                    model=judge_for_pers,
                    api_key=api_key,
                )
                personalization_dict = _judgement_to_dict(pj)
            except Exception as e:  # noqa: BLE001 — never sink the sample
                logger.warning(
                    "personalization judge failed for %s: %s",
                    sample.question_id, e,
                )
                personalization_dict = {
                    "label": "WRONG",
                    "correct": False,
                    "reasoning": f"personalization_judge_error: {type(e).__name__}: {e}",
                    "judge_model": f"{judge_for_pers}__personalization",
                }

        return SampleReport(
            question_id=sample.question_id,
            category=sample.question_type,
            question=sample.question,
            gold=sample.answer,
            answer=ans.answer,
            judgments=judgments,
            answer_latency_ms=ans.latency_ms,
            ingest_turns=stats.turns_seen,
            ingest_memories=stats.memories_added,
            retrieved_count=ans.retrieved_count,
            gold_session_ids=gold_sessions,
            retrieved_session_ids=retrieved_sessions,
            retrieval_hit=retrieval_hit,
            retrieval_overlap=overlap,
            predicted_mode=predicted_mode,
            personalization=personalization_dict,
        )
    finally:
        close = getattr(mem, "close", None)
        if callable(close):
            try:
                await asyncio.to_thread(close)
            except Exception as e:  # noqa: BLE001 — never mask the real error
                logger.warning("mem.close() failed: %s", e)


async def run_async(
    samples: List[LMESample],
    *,
    mem_factory: Callable[[], Any],
    answer_model: str = DEFAULT_MODEL,
    judge_models: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    budget: int = 4000,
    use_extraction: bool = False,
    use_distillation: bool = False,
    distill_model: Optional[str] = None,
    max_facts: int = 40,
    parallel: int = DEFAULT_PARALLEL,
    verify: bool = False,
    verify_model: Optional[str] = None,
    verbose: bool = False,
    output_path: Optional[Path | str] = None,
    dataset_path: Optional[Path | str] = None,
    progress_callback: Optional[Callable[[int, int, SampleReport], None]] = None,
) -> LMERunReport:
    """Parallel LongMemEval orchestrator — ingest → answer → judge per sample.

    Args:
        samples: ``LMESample`` list to score.
        mem_factory: Zero-arg callable that returns a fresh ``AgentMemory``.
            Called once PER SAMPLE so each task has isolated backend state.
        answer_model: OpenRouter model id for the answerer.
        judge_models: List of OpenRouter model ids. Multiple judges are
            called in parallel per sample and scored independently.
        api_key: Optional OPENROUTER_API_KEY override.
        budget: Recall token budget per question.
        use_extraction: Run the LLM extractor during ingest.
        max_facts: Cap on facts injected into the answerer prompt.
        parallel: Max concurrent samples. Default 4. Increase at your
            own rate-limit risk.
        verbose: Print per-sample verdicts to stdout as they arrive.
        output_path: Optional path to write the final JSON report.
        progress_callback: Optional ``(completed, total, sample_report)``
            hook fired as each sample finishes — useful for custom UIs.

    Returns:
        ``LMERunReport`` ordered the same way as ``samples`` (stable output
        despite concurrent execution).
    """
    from dataclasses import asdict

    judge_models = list(judge_models or list(DEFAULT_JUDGES))
    if distill_model is None or verify_model is None:
        from attestor.config import get_stack
        s = get_stack()
        if distill_model is None:
            distill_model = s.models.distill
        if verify_model is None and verify:
            verify_model = s.models.verifier
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

    semaphore = asyncio.Semaphore(max(1, parallel))
    ordered_reports: List[Optional[SampleReport]] = [None] * len(samples)

    async def _guarded(idx: int, sample: LMESample) -> None:
        async with semaphore:
            try:
                report = await _process_sample(
                    sample,
                    mem_factory=mem_factory,
                    answer_model=answer_model,
                    judge_models=judge_models,
                    api_key=api_key,
                    budget=budget,
                    use_extraction=use_extraction,
                    use_distillation=use_distillation,
                    distill_model=distill_model,
                    max_facts=max_facts,
                    verify=verify,
                    verify_model=verify_model,
                )
            except Exception as e:  # noqa: BLE001 — one bad sample must not sink the run
                logger.exception(
                    "sample %s failed; recording as all-WRONG",
                    sample.question_id,
                )
                report = SampleReport(
                    question_id=sample.question_id,
                    category=sample.question_type,
                    question=sample.question,
                    gold=sample.answer,
                    answer=f"pipeline_error: {type(e).__name__}: {e}",
                    judgments={
                        jm: {
                            "label": "WRONG",
                            "correct": False,
                            "reasoning": f"pipeline_error: {e}",
                            "judge_model": jm,
                        }
                        for jm in judge_models
                    },
                    answer_latency_ms=0.0,
                    ingest_turns=0,
                    ingest_memories=0,
                    retrieved_count=0,
                )
            ordered_reports[idx] = report
            if verbose:
                verdicts = ", ".join(
                    f"{jm.split('/')[-1]}={report.judgments[jm]['label']}"
                    for jm in judge_models
                )
                done = sum(1 for r in ordered_reports if r is not None)
                print(
                    f"[{done}/{len(samples)}] {sample.question_id} "
                    f"[{sample.question_type}] → {verdicts}",
                    flush=True,
                )
            if progress_callback is not None:
                done = sum(1 for r in ordered_reports if r is not None)
                progress_callback(done, len(samples), report)

    await asyncio.gather(
        *(_guarded(i, s) for i, s in enumerate(samples)), return_exceptions=False
    )

    sample_reports: List[SampleReport] = [r for r in ordered_reports if r is not None]
    completed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    by_category, by_judge = _summarize(sample_reports, judge_models)

    # Inter-judge agreement — only meaningful with ≥2 judges.
    agreement: dict = {}
    if len(judge_models) >= 2:
        for a, b in itertools.combinations(judge_models, 2):
            both_correct = sum(
                1 for r in sample_reports
                if r.judgments.get(a, {}).get("correct")
                and r.judgments.get(b, {}).get("correct")
            )
            both_wrong = sum(
                1 for r in sample_reports
                if not r.judgments.get(a, {}).get("correct", True)
                and not r.judgments.get(b, {}).get("correct", True)
            )
            agree = both_correct + both_wrong
            total = len(sample_reports)
            agreement[f"{a}__vs__{b}"] = {
                "both_correct": both_correct,
                "both_wrong": both_wrong,
                "agreement_pct": round(100.0 * agree / total, 2) if total else 0.0,
            }

    by_judge_enriched = dict(by_judge)
    if agreement:
        by_judge_enriched["_inter_judge_agreement"] = agreement

    # Provenance — captured once per run for third-party verification.
    git_sha, git_dirty = _git_sha()
    ds_path_str = str(Path(dataset_path).expanduser().resolve()) if dataset_path else ""
    ds_sha = _sha256_file(ds_path_str) if ds_path_str and Path(ds_path_str).exists() else ""
    provenance = RunProvenance(
        git_sha=git_sha,
        git_dirty=git_dirty,
        attestor_version=_attestor_version(),
        python_version=sys.version.split()[0],
        platform=sys.platform,
        argv=tuple(sys.argv),
        dataset_path=ds_path_str,
        dataset_sha256=ds_sha,
        dataset_sample_count=len(samples),
        started_at_utc=started,
        completed_at_utc=completed,
    )

    # Echo the runtime config so the output is self-describing.
    run_config = {
        "answer_model": answer_model,
        "judge_models": list(judge_models),
        "budget": budget,
        "use_extraction": use_extraction,
        "use_distillation": use_distillation,
        "distill_model": distill_model if use_distillation else None,
        "max_facts": max_facts,
        "parallel": parallel,
        "verify": verify,
        "verify_model": verify_model if verify else None,
    }

    by_dimension = _summarize_dimensions(sample_reports)

    report = LMERunReport(
        total=len(sample_reports),
        answer_model=answer_model,
        judge_models=tuple(judge_models),
        by_category=by_category,
        by_judge=by_judge_enriched,
        started_at=started,
        completed_at=completed,
        samples=tuple(sample_reports),
        provenance=provenance,
        run_config=run_config,
        by_dimension=by_dimension,
    )

    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(asdict(report), indent=2, sort_keys=False)
        out.write_text(payload)
        sidecar = out.with_suffix(out.suffix + ".sha256")
        sidecar.write_text(f"{_sha256_str(payload)}  {out.name}\n")
        logger.info("LongMemEval report written: %s (sha256 %s)", out, sidecar.name)

    return report


def run(
    samples: List[LMESample],
    *,
    mem_factory: Optional[Callable[[], Any]] = None,
    mem: Any = None,
    answer_model: str = DEFAULT_MODEL,
    judge_models: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    budget: int = 4000,
    use_extraction: bool = False,
    use_distillation: bool = False,
    distill_model: Optional[str] = None,
    max_facts: int = 40,
    parallel: int = DEFAULT_PARALLEL,
    verify: bool = False,
    verify_model: Optional[str] = None,
    verbose: bool = False,
    output_path: Optional[Path | str] = None,
    dataset_path: Optional[Path | str] = None,
    progress_callback: Optional[Callable[[int, int, SampleReport], None]] = None,
) -> LMERunReport:
    """Synchronous entry point — thin wrapper over ``run_async``.

    Accepts either ``mem_factory`` (recommended — true per-sample isolation)
    or ``mem`` (single shared instance; parallel is forced to 1 for safety).
    """
    if mem_factory is None and mem is None:
        raise ValueError("run() requires mem_factory or mem")

    if mem_factory is None:
        # Legacy single-instance path: force serial to keep psycopg2 happy.
        shared_mem = mem
        def _single() -> Any:
            return shared_mem
        mem_factory = _single
        parallel = 1

    return asyncio.run(
        run_async(
            samples,
            mem_factory=mem_factory,
            answer_model=answer_model,
            judge_models=judge_models,
            api_key=api_key,
            budget=budget,
            use_extraction=use_extraction,
            use_distillation=use_distillation,
            distill_model=distill_model,
            max_facts=max_facts,
            parallel=parallel,
            verify=verify,
            verify_model=verify_model,
            verbose=verbose,
            output_path=output_path,
            dataset_path=dataset_path,
            progress_callback=progress_callback,
        )
    )
