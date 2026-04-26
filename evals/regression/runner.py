"""Regression runner (Phase 9.1.3).

Drives a list of RegressionCases through a memory backend and produces a
RegressionReport. The runner is duck-typed against a tiny protocol so it
can target either a real ``AgentMemory`` (live integration) or a fake
in-memory store (unit tests).

Memory protocol expected by ``run_regression``:

    class MemoryProtocol(Protocol):
        def ingest_round(self, user_turn, assistant_turn, **kwargs): ...
        def recall_as_pack(self, query, **kwargs): ...
        def reset(self) -> None: ...    # optional; for between-case isolation

If ``reset`` is absent, the caller is responsible for isolation (e.g.,
fresh schema per case, or namespace partitioning).
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from evals.regression.cases import RegressionCase, Round
from evals.regression.scorer import (
    CaseResult, RegressionReport, score_case,
)

logger = logging.getLogger("attestor.regression")


# ── Helpers ───────────────────────────────────────────────────────────────


def _parse_iso(s: Optional[str]) -> Optional[datetime]:
    if s is None:
        return None
    # Accept "Z" suffix as UTC (yaml -> str -> datetime)
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def _build_turns(case_id: str, round_: Round, idx: int) -> tuple[Any, Any]:
    """Build the (user_turn, assistant_turn) pair for ingest_round."""
    from attestor.conversation.turns import ConversationTurn

    ts = _parse_iso(round_.ts) or datetime.now(timezone.utc)
    thread_id = f"reg-{case_id}"
    user = ConversationTurn(
        thread_id=thread_id, speaker="user", role="user",
        content=round_.user, ts=ts,
    )
    asst = ConversationTurn(
        thread_id=thread_id, speaker="assistant", role="assistant",
        content=round_.assistant, ts=ts,
    )
    return user, asst


def _recall_kwargs(case: RegressionCase) -> dict:
    kw: dict = {}
    if case.as_of:
        kw["as_of"] = _parse_iso(case.as_of)
    if case.time_window_start and case.time_window_end:
        kw["time_window"] = (
            _parse_iso(case.time_window_start),
            _parse_iso(case.time_window_end),
        )
    return kw


# ── Public API ────────────────────────────────────────────────────────────


def run_case(
    case: RegressionCase,
    mem: Any,
    *,
    ingest_kwargs: Optional[dict] = None,
) -> CaseResult:
    """Execute one case end-to-end and score it.

    Errors during ingest or recall are caught and recorded as a failed
    CaseResult — never raised — so a single broken case can't kill the
    rest of the suite.
    """
    ingest_kwargs = ingest_kwargs or {}
    try:
        for i, round_ in enumerate(case.ingest):
            try:
                user, asst = _build_turns(case.id, round_, i)
            except Exception as exc:
                logger.exception(
                    "case %s round %d: turn build failed", case.id, i,
                )
                return CaseResult(
                    case_id=case.id, passed=False,
                    reasons=(f"round {i} turn build error: {exc}",),
                )
            mem.ingest_round(user, asst, **ingest_kwargs)

        pack = mem.recall_as_pack(case.query, **_recall_kwargs(case))
    except Exception as exc:
        logger.exception("case %s: runtime error", case.id)
        return CaseResult(
            case_id=case.id, passed=False,
            reasons=(f"runtime error: {type(exc).__name__}: {exc}",),
        )

    return score_case(case, pack)


def run_regression(
    cases: Sequence[RegressionCase],
    mem: Any,
    *,
    isolate: Optional[Any] = None,
    ingest_kwargs: Optional[dict] = None,
) -> RegressionReport:
    """Run every case in ``cases`` and return an aggregated report.

    ``isolate`` is a no-arg callable invoked between cases to reset state
    (e.g., ``lambda: mem.purge_user("local")``). When None, the runner
    relies on the ingest layer to namespace by ``thread_id`` — which
    keeps episodes separate but DOES allow user-scope memories from one
    case to bleed into recall of the next. For deterministic catalogs
    that only test SESSION-scoped recall, that's fine; for cross-case
    isolation, pass an isolate callback.
    """
    results = []
    for case in cases:
        if isolate is not None:
            try:
                isolate()
            except Exception:
                logger.exception(
                    "isolate() before case %s failed; continuing", case.id,
                )
        results.append(run_case(case, mem, ingest_kwargs=ingest_kwargs))
    return RegressionReport(results=tuple(results))


# ── Convenience: load + run from a YAML path ──────────────────────────────


def run_yaml(
    yaml_path: str,
    mem: Any,
    *,
    isolate: Optional[Any] = None,
    ingest_kwargs: Optional[dict] = None,
) -> RegressionReport:
    """Load a qa.yaml and run every case in it."""
    from evals.regression.cases import load_cases
    cases = load_cases(yaml_path)
    return run_regression(cases, mem, isolate=isolate,
                          ingest_kwargs=ingest_kwargs)
