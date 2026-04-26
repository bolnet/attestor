"""Regression case schema + YAML loader (Phase 9.1.1).

Each case is a self-contained recall scenario: a list of conversation
rounds to ingest, a query to ask, and assertions about the recalled
ContextPack. Pure data — no I/O beyond YAML parsing, no model calls.

YAML schema (one item per case):

    - id: str            # unique, becomes test name
      description: str   # human-readable summary
      category: str      # preference|temporal|abstention|multi_session|...
      ingest:            # list of rounds (one user/assistant exchange per round)
        - user: "..."
          assistant: "..."
          ts: "2026-01-15T10:00:00Z"   # optional; round wall-clock time
      query: "..."
      must_contain: ["..."]            # case-insensitive substring of any pack entry
      must_not_contain: ["..."]
      abstain_required: false          # if true, pack MUST be empty (no useful memories)
      abstain_ok: false                # if true, an empty pack also passes
      as_of: "2026-01-15T10:00:00Z"    # optional bi-temporal replay
      time_window_start: "..."         # optional time-window query
      time_window_end:   "..."

Validation is strict — unknown top-level keys raise. Unknown round keys
also raise. The point is that a typo in qa.yaml fails loud, not silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


_ROUND_KEYS = {"user", "assistant", "ts"}
_CASE_KEYS = {
    "id", "description", "category",
    "ingest", "query",
    "must_contain", "must_not_contain",
    "abstain_required", "abstain_ok",
    "as_of", "time_window_start", "time_window_end",
}


@dataclass(frozen=True)
class Round:
    """One conversation round to ingest before the query."""
    user: str
    assistant: str
    ts: Optional[str] = None


@dataclass(frozen=True)
class RegressionCase:
    """A single regression scenario.

    Frozen so the loader produces an immutable catalog the runner can
    safely share across workers.
    """
    id: str
    description: str
    category: str
    ingest: Tuple[Round, ...]
    query: str
    must_contain: Tuple[str, ...] = ()
    must_not_contain: Tuple[str, ...] = ()
    abstain_required: bool = False
    abstain_ok: bool = False
    as_of: Optional[str] = None
    time_window_start: Optional[str] = None
    time_window_end: Optional[str] = None

    def __post_init__(self) -> None:  # type: ignore[override]
        if not self.id:
            raise ValueError("case.id is required")
        if not self.query:
            raise ValueError(f"case {self.id!r}: query is required")
        if self.abstain_required and self.must_contain:
            raise ValueError(
                f"case {self.id!r}: abstain_required is incompatible with "
                "must_contain — if you need recalled facts the case is not "
                "an abstention case"
            )
        if (self.time_window_start is None) != (self.time_window_end is None):
            raise ValueError(
                f"case {self.id!r}: time_window_start and time_window_end "
                "must be specified together or not at all"
            )


# ── Loader ────────────────────────────────────────────────────────────────


def _parse_round(raw: Any, case_id: str, idx: int) -> Round:
    if not isinstance(raw, dict):
        raise ValueError(
            f"case {case_id!r} ingest[{idx}]: expected mapping, got "
            f"{type(raw).__name__}"
        )
    extra = set(raw) - _ROUND_KEYS
    if extra:
        raise ValueError(
            f"case {case_id!r} ingest[{idx}]: unknown keys {sorted(extra)}"
        )
    if "user" not in raw or "assistant" not in raw:
        raise ValueError(
            f"case {case_id!r} ingest[{idx}]: both 'user' and 'assistant' "
            "are required"
        )
    return Round(
        user=str(raw["user"]),
        assistant=str(raw["assistant"]),
        ts=raw.get("ts"),
    )


def _parse_case(raw: Any) -> RegressionCase:
    if not isinstance(raw, dict):
        raise ValueError(
            f"regression case must be a mapping, got {type(raw).__name__}"
        )
    extra = set(raw) - _CASE_KEYS
    if extra:
        raise ValueError(
            f"regression case has unknown keys {sorted(extra)}"
        )
    case_id = str(raw.get("id", "")).strip()
    rounds_raw = raw.get("ingest") or []
    if not isinstance(rounds_raw, list):
        raise ValueError(
            f"case {case_id!r}: ingest must be a list, got "
            f"{type(rounds_raw).__name__}"
        )
    rounds = tuple(
        _parse_round(r, case_id, i) for i, r in enumerate(rounds_raw)
    )
    return RegressionCase(
        id=case_id,
        description=str(raw.get("description", "")),
        category=str(raw.get("category", "general")),
        ingest=rounds,
        query=str(raw.get("query", "")),
        must_contain=tuple(str(s) for s in raw.get("must_contain", []) or []),
        must_not_contain=tuple(
            str(s) for s in raw.get("must_not_contain", []) or []
        ),
        abstain_required=bool(raw.get("abstain_required", False)),
        abstain_ok=bool(raw.get("abstain_ok", False)),
        as_of=raw.get("as_of"),
        time_window_start=raw.get("time_window_start"),
        time_window_end=raw.get("time_window_end"),
    )


def load_cases(path: Path | str) -> List[RegressionCase]:
    """Load and validate a qa.yaml file. Raises on any structural issue."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text())
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(
            f"{p}: top-level must be a list of cases, got "
            f"{type(raw).__name__}"
        )
    cases = [_parse_case(item) for item in raw]
    # Reject duplicate ids — they make test output ambiguous
    seen: Dict[str, int] = {}
    for i, c in enumerate(cases):
        if c.id in seen:
            raise ValueError(
                f"{p}: duplicate case id {c.id!r} at index {i} "
                f"(first seen at {seen[c.id]})"
            )
        seen[c.id] = i
    return cases
