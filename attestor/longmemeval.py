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
import itertools
import json
import logging
import os
import re
import urllib.request
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "openai/gpt-4.1-mini"
# Default dual-judge. Second judge anchors out answerer-judge collusion.
DEFAULT_JUDGES = ("openai/gpt-4.1-mini", "anthropic/claude-haiku-4.5")
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
    max_facts: int = 40,
    parallel: int = DEFAULT_PARALLEL,
    verbose: bool = False,
    output_path: Optional[Path | str] = None,
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
    started = datetime.utcnow().isoformat(timespec="seconds")

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
                    max_facts=max_facts,
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
    completed = datetime.utcnow().isoformat(timespec="seconds")
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

    report = LMERunReport(
        total=len(sample_reports),
        answer_model=answer_model,
        judge_models=tuple(judge_models),
        by_category=by_category,
        by_judge=by_judge_enriched,
        started_at=started,
        completed_at=completed,
        samples=tuple(sample_reports),
    )

    if output_path:
        out = Path(output_path).expanduser().resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(asdict(report), indent=2))
        logger.info("LongMemEval report written: %s", out)

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
    max_facts: int = 40,
    parallel: int = DEFAULT_PARALLEL,
    verbose: bool = False,
    output_path: Optional[Path | str] = None,
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
            max_facts=max_facts,
            parallel=parallel,
            verbose=verbose,
            output_path=output_path,
            progress_callback=progress_callback,
        )
    )
