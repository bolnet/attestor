# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``ForgetUser`` saga activities — one per backend, audit first.

Activity side = I/O side. Each activity wraps one lane from
``attestor.compliance.forget_lanes`` (plain SQL / backend calls, also
usable in-process) and returns a ``ForgetLaneOutcome`` with counts only.

Failure contract: auth / permission / malformed-id errors are permanent
(``non_retryable``); anything else (Postgres, Pinecone, Neo4j down) is
retried by the workflow's ``RetryPolicy`` with unlimited attempts, so a
graph outage keeps the graph lane retrying while the others complete.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from temporalio import activity

from attestor.compliance import forget_lanes as lanes
from attestor.durable.activities._errors import raise_for
from attestor.durable.models import (
    FORGET_DOC_ACTIVITY,
    FORGET_GRAPH_ACTIVITY,
    FORGET_LANE_AUDIT,
    FORGET_LANE_DOC,
    FORGET_LANE_GRAPH,
    FORGET_LANE_STATE,
    FORGET_LANE_VECTOR,
    FORGET_STATE_ACTIVITY,
    FORGET_VECTOR_ACTIVITY,
    LANE_STATUS_OK,
    WRITE_FORGET_AUDIT_ACTIVITY,
    ForgetLaneOutcome,
    ForgetRequest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("attestor.durable.activities.forget")

ERROR_TYPE_TRANSIENT = "ForgetFailed"
ERROR_TYPE_PERMANENT = "ForgetPermanent"


def _ok(lane: str, deleted: int = 0, *, edges: int = 0, skipped: bool = False) -> ForgetLaneOutcome:
    return ForgetLaneOutcome(
        lane=lane, status=LANE_STATUS_OK, deleted=deleted, edges_deleted=edges, skipped=skipped,
    )


class ForgetActivities:
    """Activity set bound to a lazily-built ``AgentMemory``."""

    def __init__(self, memory_provider: Callable[[], Any]) -> None:
        self._provider = memory_provider

    def _get(self) -> Any:
        # Always ask the provider: it caches the healthy instance and
        # rebuilds one whose backends failed to initialise (worker.py).
        return self._provider()

    def _fail(self, exc: BaseException, req: ForgetRequest, lane: str) -> None:
        raise_for(
            exc, subject=f"user={req.user_id} audit={req.audit_id}", step=lane,
            transient_type=ERROR_TYPE_TRANSIENT, permanent_type=ERROR_TYPE_PERMANENT,
            logger=logger,
        )

    @activity.defn(name=WRITE_FORGET_AUDIT_ACTIVITY)
    def write_forget_audit(self, req: ForgetRequest) -> ForgetLaneOutcome:
        """Pre-count and append the audit row — MUST succeed before any delete."""
        try:
            mem = self._get()
            counts = lanes.count_user_rows(mem, req.user_id)
            lanes.write_forget_audit(
                mem, audit_id=req.audit_id, user_id=req.user_id,
                doc_rows=counts.doc_rows, state_rows=counts.state_rows,
                initiated_by=req.initiated_by,
            )
        except Exception as exc:
            self._fail(exc, req, FORGET_LANE_AUDIT)
        return _ok(FORGET_LANE_AUDIT)

    @activity.defn(name=FORGET_DOC_ACTIVITY)
    def forget_doc(self, req: ForgetRequest) -> ForgetLaneOutcome:
        try:
            deleted = lanes.forget_doc(self._get(), req.user_id)
        except Exception as exc:
            self._fail(exc, req, FORGET_LANE_DOC)
        return _ok(FORGET_LANE_DOC, deleted)

    @activity.defn(name=FORGET_VECTOR_ACTIVITY)
    def forget_vector(self, req: ForgetRequest) -> ForgetLaneOutcome:
        try:
            out = lanes.forget_vector(self._get(), req.user_id)
        except Exception as exc:
            self._fail(exc, req, FORGET_LANE_VECTOR)
        return _ok(FORGET_LANE_VECTOR, out.deleted, skipped=out.skipped)

    @activity.defn(name=FORGET_GRAPH_ACTIVITY)
    def forget_graph(self, req: ForgetRequest) -> ForgetLaneOutcome:
        try:
            out = lanes.forget_graph(self._get(), req.user_id)
        except Exception as exc:
            self._fail(exc, req, FORGET_LANE_GRAPH)
        return _ok(FORGET_LANE_GRAPH, out.nodes, edges=out.edges, skipped=out.skipped)

    @activity.defn(name=FORGET_STATE_ACTIVITY)
    def forget_state(self, req: ForgetRequest) -> ForgetLaneOutcome:
        try:
            out = lanes.forget_state(self._get(), req.user_id)
        except Exception as exc:
            self._fail(exc, req, FORGET_LANE_STATE)
        return _ok(FORGET_LANE_STATE, out.deleted, skipped=out.skipped)
