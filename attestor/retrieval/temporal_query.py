"""Temporal query expansion (Phase 5.1, roadmap §C.1).

Extracts a ``TimeWindow`` from a natural-language query so the
retrieval pipeline can hard-filter by ``t_valid`` overlap before doing
semantic search. Wu et al. Finding 3: +7-11% on the LongMemEval
``temporal-reasoning`` category from running this single LLM call at
query time.

Examples:
  "What did I say last Tuesday?"  → start=last_tuesday_00:00, end=last_tuesday_23:59
  "Before I had kids"             → start=null, end=<earliest known kids date>
  "Recent recommendations"         → start=now-30d, end=now
  "What's my favorite color?"     → has_time_constraint=False (returns None)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

logger = logging.getLogger("attestor.retrieval.temporal_query")


TIME_EXTRACTION_PROMPT = """\
Extract a time window from this query, anchored to the reference time.
Return JSON only.

Reference time (now): {now}
Query: "{query}"

Output schema:
{{
  "has_time_constraint": <true|false>,
  "start": "<ISO datetime or null>",
  "end":   "<ISO datetime or null>",
  "interpretation": "<one short sentence>"
}}

Examples:
- "What did I say last Tuesday?" -> start=last_tuesday_00:00, end=last_tuesday_23:59
- "Before I had kids" -> has_time_constraint=true, start=null, end=<earliest known kids date>
- "What's my favorite color?" -> has_time_constraint=false
- "Recent recommendations" -> start=now-30d, end=now

Output:
"""


@dataclass(frozen=True)
class TimeWindow:
    """A time range over which to filter retrieval.

    Either bound may be None (open-ended). ``has_time_constraint`` is
    implicit: a TimeWindow object exists ⇒ the caller said yes.

    `interpretation` is a short human-readable description from the LLM,
    useful for debugging traces and audit logs.
    """

    start: Optional[datetime]
    end: Optional[datetime]
    interpretation: str = ""

    def __post_init__(self) -> None:
        # Allow strings on input — coerce to datetime
        if isinstance(self.start, str):
            object.__setattr__(self, "start", _parse_iso(self.start))
        if isinstance(self.end, str):
            object.__setattr__(self, "end", _parse_iso(self.end))
        if self.start is not None and self.end is not None:
            if self.start > self.end:
                raise ValueError(
                    f"TimeWindow start ({self.start}) > end ({self.end})"
                )

    @property
    def is_open_ended(self) -> bool:
        return self.start is None or self.end is None

    @property
    def is_unbounded(self) -> bool:
        return self.start is None and self.end is None


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    """Best-effort ISO 8601 → datetime. Returns None on failure or 'null'."""
    if s is None:
        return None
    if not isinstance(s, str):
        return None
    s = s.strip()
    if not s or s.lower() in {"null", "none"}:
        return None
    # Strip trailing Z (Python datetime.fromisoformat accepts +00:00)
    s2 = s.replace("Z", "+00:00") if s.endswith("Z") else s
    try:
        dt = datetime.fromisoformat(s2)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ──────────────────────────────────────────────────────────────────────────
# Expander
# ──────────────────────────────────────────────────────────────────────────


def _strip_markdown_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()
    return text


class TemporalQueryExpander:
    """Run an LLM to extract a TimeWindow from a query.

    Designed to be cheap: small model, low max_tokens, JSON-only output.
    Returns None when:
      - the query has no time constraint
      - the LLM call fails
      - the response is unparseable

    Returning None is safe: the orchestrator just skips the temporal
    pre-filter and behaves like Phase 4.
    """

    def __init__(
        self,
        client: Optional[Any] = None,
        *,
        model: str = "openai/gpt-4o-mini",
        max_tokens: int = 256,
    ) -> None:
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def expand(
        self, query: str, now: Optional[datetime] = None,
    ) -> Optional[TimeWindow]:
        """Return a TimeWindow if the query has a time constraint, else None."""
        if not query or not query.strip():
            return None
        ref_now = now or datetime.now(timezone.utc)
        prompt = TIME_EXTRACTION_PROMPT.format(
            now=ref_now.isoformat(), query=query,
        )
        client = self._client or _default_client()
        if client is None:
            return None
        try:
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = response.choices[0].message.content or ""
        except Exception as e:
            logger.debug("temporal LLM call failed: %s", e)
            return None
        return _parse_window(raw)


def _parse_window(raw: str) -> Optional[TimeWindow]:
    """Parse a TIME_EXTRACTION_PROMPT response into a TimeWindow or None."""
    text = _strip_markdown_fences(raw)
    if not text:
        return None
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as e:
        logger.debug("temporal JSON parse failed: %s; raw=%r", e, raw[:200])
        return None
    if not isinstance(parsed, dict):
        return None
    if not parsed.get("has_time_constraint"):
        return None
    start = _parse_iso(parsed.get("start"))
    end = _parse_iso(parsed.get("end"))
    # No bounds at all → not actually constrained
    if start is None and end is None:
        return None
    interp = parsed.get("interpretation") or ""
    if not isinstance(interp, str):
        interp = ""
    try:
        return TimeWindow(start=start, end=end, interpretation=interp)
    except ValueError as e:
        # start > end, etc. — treat as unconstrained
        logger.debug("temporal window invalid: %s", e)
        return None


def _default_client() -> Optional[Any]:
    """Build the same OpenAI/OpenRouter client the extractor uses.

    Returns None if no key set — the expander then no-ops gracefully.
    """
    try:
        from openai import OpenAI
    except ImportError:
        return None
    or_key = os.environ.get("OPENROUTER_API_KEY")
    if or_key:
        return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=or_key)
    oa_key = os.environ.get("OPENAI_API_KEY")
    if oa_key:
        return OpenAI(api_key=oa_key)
    return None
