"""LongMemEval dataset and sample types.

Schema, frozen dataclasses, dataset URL constants, load/download
helpers, and the per-turn distillation pipeline. Split out of the
legacy 2466-line ``attestor.longmemeval`` module — every symbol kept
verbatim so callers see byte-identical behavior.
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
from typing import Any

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
    answer_session_ids: tuple[str, ...]
    haystack_dates: tuple[str, ...]
    haystack_session_ids: tuple[str, ...]
    haystack_sessions: tuple[tuple[LMETurn, ...], ...]

    @property
    def is_temporal(self) -> bool:
        return self.question_type == TEMPORAL_CATEGORY

    @property
    def total_haystack_turns(self) -> int:
        return sum(len(sess) for sess in self.haystack_sessions)


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------


def parse_lme_date(date_str: str) -> datetime | None:
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
    # timeout=30: SDK defaults are None which can hang indefinitely on a
    # stalled connection. See timeout-cascade incident (2026-04-30).
    with urllib.request.urlopen(url, timeout=30) as resp:  # noqa: S310
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
    path: Path | str, *, limit: int | None = None
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
# Ingest stats
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
    entities: tuple[str, ...] = ()
    topics: tuple[str, ...] = ()


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


def _normalize_str_list(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        parts = [p.strip() for p in re.split(r"[;,]", value) if p.strip()]
        return tuple(parts)
    if isinstance(value, (list, tuple)):
        parts = [str(x).strip() for x in value if str(x).strip()]
        return tuple(parts)
    return ()


def _fact_from_record(record: Any, *, fallback_speaker: str = "unknown") -> DistilledFact | None:
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


def _extract_json_array(text: str) -> list | None:
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
) -> list[DistilledFact]:
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
    model: str | None = None,
    api_key: str | None = None,
    max_tokens: int = 3000,
) -> list[DistilledFact]:
    """Run one turn through the distillation LLM; return structured facts.

    An empty list means the turn produced no durable memory worth keeping
    (pleasantries, generic puzzle answers, etc.). The caller should NOT
    store anything for those turns.

    Returns structured ``DistilledFact`` records. On LLM failure (timeout,
    402, etc.) falls back to a single minimal record carrying the raw turn
    so downstream retrieval still has something to latch onto.
    """
    # Lazy import — ``DISTILL_PROMPT`` lives in ``prompts.py``. The LLM
    # client helpers (``_chat`` / ``_get_client`` / ``_get_client_for_model``)
    # live in this module further down, so no extra import is needed here.
    from attestor.longmemeval.prompts import DISTILL_PROMPT

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
        if api_key is not None:
            client = _get_client(api_key)
            clean_model = model
        else:
            client, clean_model = _get_client_for_model(model)
        raw = _chat(client, clean_model, prompt, max_tokens=max_tokens, role="distill")
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


# ---------------------------------------------------------------------------
# LLM client helpers
# ---------------------------------------------------------------------------
#
# Shared by ``judge.py``, ``runner.py``, and ``distill_turn`` above. They
# live here (not in ``runner.py``) so all three call sites can import the
# same identity without a runner↔caller cycle and so ``runner.py`` stays
# under the per-module size cap.


def _get_client_for_model(model: str) -> tuple[Any, str]:
    """Pool-aware client lookup. Returns ``(client, clean_model)`` where
    ``clean_model`` has the provider prefix stripped — pass ``clean_model``
    to the SDK. Falls back to the default provider when ``model`` has
    no ``provider/`` prefix.
    """
    from attestor.llm_trace import get_client_for_model
    return get_client_for_model(model)


def _get_client(api_key: str | None = None) -> Any:
    """Instantiate an OpenAI-compatible client (default provider).

    Back-compat shim retained for callers that don't have an explicit
    model in scope. Model-aware call sites should prefer
    ``_get_client_for_model(model)`` so the pool can route per-model.

    Resolution order (top wins):
      1. ``LME_LLM_BASE_URL`` env — explicit ad-hoc override; mostly
         used to point at local Ollama (``http://localhost:11434/v1``).
         When the URL points at localhost, the API key is optional.
      2. ``api_key`` argument — when caller passes one, build a client
         directly so the explicit override beats the pool's lookup.
      3. Pool default — delegate to ``attestor.llm_trace`` and ask for
         the default provider's client.

    There is no hardcoded fallback. ``configs/attestor.yaml`` is the
    source of truth for the default provider; if it can't be loaded
    we raise loudly rather than silently routing somewhere unintended.
    """
    from attestor.llm_trace import make_client
    from attestor.config import get_stack

    llm_cfg = get_stack().llm  # raises loudly if YAML unloadable — by design.

    # 1 — explicit env override wins (preserved verbatim — local-Ollama path).
    env_base_url = os.environ.get("LME_LLM_BASE_URL")
    if env_base_url:
        base_url = env_base_url
        key_env = llm_cfg.api_key_env
        is_local = "localhost" in base_url or "127.0.0.1" in base_url
        key = api_key or os.environ.get(key_env)
        if not key:
            if is_local:
                key = "ollama"  # placeholder; Ollama ignores the key
            else:
                raise RuntimeError(
                    f"{key_env} not set — required for LongMemEval "
                    f"answer/judge against {base_url} "
                    f"(llm.provider={llm_cfg.provider!r}). Either export "
                    f"{key_env}, switch llm.provider in configs/attestor.yaml, "
                    f"or set LME_LLM_BASE_URL=http://localhost:11434/v1 to "
                    f"run against local Ollama instead."
                )
        return make_client(base_url=base_url, api_key=key)

    # 2 — explicit api_key override: build directly so the caller's key
    # truly wins over any cached pool client.
    if api_key is not None:
        return make_client(base_url=llm_cfg.base_url, api_key=api_key)

    # 3 — delegate to the pool for the default provider. If YAML is
    # missing or malformed the pool raises; we let it propagate.
    from attestor.llm_trace import _get_pool
    pool = _get_pool()
    return pool.client_for(pool.default_strategy().name)


def _chat(
    client: Any,
    model: str,
    prompt: str,
    *,
    max_tokens: int = 300,
    reasoning_effort: str | None = None,
    role: str | None = None,
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

    create_kwargs: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    if reasoning_effort:
        create_kwargs["reasoning_effort"] = reasoning_effort

    from attestor.llm_trace import traced_create
    response = traced_create(client, role=role or "lme.chat", **create_kwargs)
    return response.choices[0].message.content or ""


def _format_recall_context(results: list[Any], max_facts: int = 40) -> str:
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


# ---------------------------------------------------------------------------
# Answer wire-format + parsers
# ---------------------------------------------------------------------------


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
    retrieved_session_ids: tuple[str, ...] = ()
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


def _extract_retrieved_session_ids(results: list) -> tuple[str, ...]:
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


def _strip_reasoning(raw: str) -> tuple[str, str]:
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
