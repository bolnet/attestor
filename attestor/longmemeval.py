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
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4.1-mini"
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

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


# ---------------------------------------------------------------------------
# Answer + judge
# ---------------------------------------------------------------------------

ANSWER_PROMPT = (
    "You are answering a question based on an assistant's memory of a past "
    "chat history. Use ONLY the facts listed below. Each fact is prefixed "
    "with [YYYY-MM-DD] indicating the session date when the turn was recorded.\n\n"
    "Ground temporal language against those session dates:\n"
    "  - Relative phrases (\"last week\", \"yesterday\", \"two months ago\") must be\n"
    "    resolved to absolute dates using the session date of the turn that\n"
    "    contains them.\n"
    "  - When a question asks \"when did X happen\", answer with the EVENT date,\n"
    "    not the session date unless they coincide.\n"
    "  - Prefer concrete absolute dates; never paraphrase the relative phrase.\n\n"
    "If the facts do not support an answer, respond with exactly: "
    "I don't know.\n\n"
    "Question (asked on {question_date}):\n{question}\n\n"
    "Facts:\n{context}\n\n"
    "Answer concisely. Do not repeat the question."
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
    """Instantiate an OpenAI client pointed at OpenRouter for benchmark runs."""
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover — import-time error path
        raise RuntimeError(
            "openai package required for benchmarks. Install with "
            "`poetry add --group dev openai`."
        ) from e

    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set — required for LongMemEval answer/judge."
        )
    return OpenAI(base_url=OPENROUTER_BASE_URL, api_key=key)


def _chat(client: Any, model: str, prompt: str, *, max_tokens: int = 300) -> str:
    """One-shot chat completion; returns content text."""
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content or ""


def _format_recall_context(results: List[Any], max_facts: int = 40) -> str:
    """Join top retrieval hits into a plain newline-delimited context block."""
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


def answer_question(
    mem: Any,
    sample: LMESample,
    *,
    budget: int = 4000,
    model: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
    max_facts: int = 40,
    max_tokens: int = 150,
) -> AnswerResult:
    """Recall + synthesize an answer for a LongMemEval sample.

    Args:
        mem: ``AgentMemory`` instance, already populated via ``ingest_history``.
        sample: ``LMESample`` being answered.
        budget: Retrieval token budget for ``mem.recall``.
        model: OpenRouter model id for synthesis.
        api_key: Optional override for ``OPENROUTER_API_KEY``.
        max_facts: Cap on facts injected into the prompt (guards prompt size).
        max_tokens: Answerer ``max_tokens``.

    Returns:
        ``AnswerResult`` with answer text + retrieval counts + latency.
    """
    import time

    ns = namespace_for(sample)
    t0 = time.monotonic()
    results = mem.recall(sample.question, budget=budget, namespace=ns) or []
    # Highest-score first (recall already scores; tolerate arbitrary order).
    results = sorted(results, key=lambda r: getattr(r, "score", 0.0), reverse=True)

    if not results:
        return AnswerResult(
            answer="I don't know.",
            retrieved_count=0,
            used_fact_count=0,
            latency_ms=round((time.monotonic() - t0) * 1000, 2),
        )

    context = _format_recall_context(results, max_facts=max_facts)
    prompt = ANSWER_PROMPT.format(
        question=sample.question,
        question_date=sample.question_date or "(unknown)",
        context=context,
    )
    client = _get_client(api_key)
    answer = _chat(client, model, prompt, max_tokens=max_tokens).strip()
    return AnswerResult(
        answer=answer,
        retrieved_count=len(results),
        used_fact_count=min(len(results), max_facts),
        latency_ms=round((time.monotonic() - t0) * 1000, 2),
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
    raw = _chat(client, model, prompt, max_tokens=max_tokens)
    label, reasoning = _parse_judge_response(raw)
    return JudgeResult(
        label=label,
        correct=label == "CORRECT",
        reasoning=reasoning,
        raw=raw,
        judge_model=model,
    )
