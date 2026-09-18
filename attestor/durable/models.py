# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Frozen models that cross the Temporal workflow boundary.

This is the ONLY Attestor module a workflow may import (under
``workflow.unsafe.imports_passed_through()``). It must therefore stay
pure: stdlib only — no ``attestor.*``, no drivers, no ``temporalio``.
Everything here is serialised by ``pydantic_data_converter``.

Workflow ids are deterministic so retries dedupe:
``consolidate-{episode_id}``, ``derive-{memory_id}``,
``rebuild-derived-{YYYYMMDDTHHMMSSZ}``, ``forget-{user_id}-{audit_id}``,
``retention-sweep-{YYYY-MM-DD}``, ``session-sweep-{YYYY-MM-DDTHH:MM}`` (plan §3).

Payload rule (plan §2, "Postgres stays the source of truth"): workflow
inputs and activity args are persisted in Temporal's event history for
the namespace's retention window — a second store that neither the
retention sweep nor a forget saga governs. So ``EpisodeRef`` carries
IDENTIFIERS ONLY; the activity re-reads the conversation text from the
claimed Postgres row. ``EPISODE_CONTENT_FIELDS`` names what must never
appear here (guarded by ``tests/test_durable_models.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import replace as _replace
from datetime import date, datetime, timezone
from typing import Any

# ── Names shared by client, worker, workflow, and activity ──────────────
CONSOLIDATE_WORKFLOW = "ConsolidateEpisode"
CONSOLIDATE_EPISODE_ACTIVITY = "consolidate_episode"
CONSOLIDATE_WORKFLOW_ID_PREFIX = "consolidate-"

DERIVE_WORKFLOW = "DeriveMemory"
REBUILD_WORKFLOW = "RebuildDerived"
EMBED_AND_UPSERT_ACTIVITY = "embed_and_upsert"
GRAPH_EXTRACT_ACTIVITY = "graph_extract"
LIST_MEMORY_IDS_ACTIVITY = "list_memory_ids"
DERIVE_WORKFLOW_ID_PREFIX = "derive-"
REBUILD_WORKFLOW_ID_PREFIX = "rebuild-derived-"
REBUILD_ID_TIMESTAMP_FORMAT = "%Y%m%dT%H%M%SZ"

DERIVE_STEP_VECTOR = "vector"
DERIVE_STEP_GRAPH = "graph"

# Queue-row columns that hold raw conversation text. They stay in Postgres.
EPISODE_CONTENT_FIELDS: tuple[str, ...] = ("user_turn_text", "assistant_turn_text")
# Memory columns that hold content / derived data. They stay in Postgres too.
MEMORY_CONTENT_FIELDS: tuple[str, ...] = ("content", "embedding", "metadata", "tags")

# RebuildDerived sizing. One child workflow per id; a page of ids per list
# activity call; ``max_pages_per_run`` bounds one run's event history
# before continue-as-new (≈ 5 events per child).
DEFAULT_REBUILD_PAGE_SIZE = 100
DEFAULT_REBUILD_WINDOW_SIZE = 10
DEFAULT_REBUILD_PAGES_PER_RUN = 10
MAX_REPORTED_FAILED_IDS = 100


def consolidate_workflow_id(episode_id: str) -> str:
    """Deterministic workflow id for one episode (idempotent starts)."""
    return f"{CONSOLIDATE_WORKFLOW_ID_PREFIX}{episode_id}"


def derive_workflow_id(memory_id: str) -> str:
    """Deterministic workflow id for one memory — add() repair and rebuild dedupe on it."""
    return f"{DERIVE_WORKFLOW_ID_PREFIX}{memory_id}"


def rebuild_workflow_id(started_at: datetime) -> str:
    """Operator-initiated rebuild id, stamped with the (UTC) start time.

    Naive datetimes are taken as UTC; the workflow itself never reads a clock.
    """
    utc = started_at if started_at.tzinfo is None else started_at.astimezone(timezone.utc)
    return f"{REBUILD_WORKFLOW_ID_PREFIX}{utc.strftime(REBUILD_ID_TIMESTAMP_FORMAT)}"


@dataclass(frozen=True)
class ActivityPolicy:
    """Timeout + retry knobs for the consolidation activity.

    Defaults follow the plan's retry facts: exponential backoff from 2 s
    capped at 5 min, unlimited attempts (``maximum_attempts=0``) —
    permanent errors are marked non-retryable by the activity instead.
    """

    start_to_close_seconds: float = 600.0
    initial_interval_seconds: float = 2.0
    backoff_coefficient: float = 2.0
    maximum_interval_seconds: float = 300.0
    maximum_attempts: int = 0


@dataclass(frozen=True)
class EpisodeRef:
    """Identifiers of a claimed ``QueuedEpisode`` — the Temporal payload.

    The queue row claim stays in Postgres (zero schema change). The
    activity re-reads the row by ``(episode_id, user_id)`` inside the
    worker, so no conversation text ever enters workflow history.
    Timestamps and scope ids are kept for observability (Temporal UI
    search / grouping); they are not content.
    """

    episode_id: str
    user_id: str
    session_id: str
    thread_id: str
    user_ts: datetime
    assistant_ts: datetime
    project_id: str | None = None
    agent_id: str | None = None

    @classmethod
    def from_queued(cls, ep: Any) -> EpisodeRef:
        """Build from a ``QueuedEpisode`` (duck-typed to keep this module pure).

        Deliberately does NOT copy ``EPISODE_CONTENT_FIELDS``.
        """
        return cls(
            episode_id=str(ep.id),
            user_id=str(ep.user_id),
            session_id=str(ep.session_id),
            thread_id=str(ep.thread_id),
            user_ts=ep.user_ts,
            assistant_ts=ep.assistant_ts,
            project_id=ep.project_id,
            agent_id=ep.agent_id,
        )


@dataclass(frozen=True)
class ConsolidateRequest:
    """Input of the ``ConsolidateEpisode`` workflow."""

    episode: EpisodeRef
    policy: ActivityPolicy = field(default_factory=ActivityPolicy)


@dataclass(frozen=True)
class ConsolidationOutcome:
    """Result of the ``ConsolidateEpisode`` workflow / activity."""

    episode_id: str
    ok: bool
    error: str | None = None
    written_memory_ids: tuple[str, ...] = ()
    user_fact_count: int = 0
    agent_fact_count: int = 0


@dataclass(frozen=True)
class DurableDispatch:
    """What ``SleepTimeConsolidator.dispatch_durable`` did with one batch."""

    workflow_ids: tuple[str, ...] = ()
    released_episode_ids: tuple[str, ...] = ()


# ── DeriveMemory (Phase 2) ───────────────────────────────────────────────

@dataclass(frozen=True)
class DeriveRef:
    """Identifier of one memory whose derived state (vector + graph) to rebuild.

    Ids only (``MEMORY_CONTENT_FIELDS`` never appear here): the activity
    re-reads content by id from the document store. ``namespace`` is for
    observability in the Temporal UI; the activity uses the row's own.

    Tenant scope (same rule as ``EpisodeRef.user_id``): ``user_id`` is the
    row's OWNER — the worker sets the Postgres RLS session user to it
    before the re-read and again before every derived write, and refuses
    a row owned by anyone else. ``agent_id`` is the writer agent, applied
    as the requester for the ``visibility='private'`` guard.
    """

    memory_id: str
    namespace: str | None = None
    user_id: str | None = None
    agent_id: str | None = None

    @classmethod
    def from_memory(cls, memory: Any) -> DeriveRef:
        """Build from a ``Memory`` (duck-typed to keep this module pure)."""
        user_id = getattr(memory, "user_id", None)
        agent_id = getattr(memory, "agent_id", None)
        return cls(
            memory_id=str(memory.id),
            namespace=memory.namespace,
            user_id=str(user_id) if user_id else None,
            agent_id=str(agent_id) if agent_id else None,
        )


@dataclass(frozen=True)
class DeriveRequest:
    """Input of the ``DeriveMemory`` workflow."""

    memory: DeriveRef
    policy: ActivityPolicy = field(default_factory=ActivityPolicy)


@dataclass(frozen=True)
class DeriveStepOutcome:
    """Result of one lane (``DERIVE_STEP_VECTOR`` / ``DERIVE_STEP_GRAPH``)."""

    memory_id: str
    step: str
    ok: bool
    error: str | None = None
    entity_count: int = 0
    relation_count: int = 0


@dataclass(frozen=True)
class DeriveOutcome:
    """Result of the ``DeriveMemory`` workflow — both lanes, ids + counts only."""

    memory_id: str
    vector: DeriveStepOutcome
    graph: DeriveStepOutcome

    @property
    def ok(self) -> bool:
        return self.vector.ok and self.graph.ok

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            f"{step.step}: {step.error}" for step in (self.vector, self.graph) if not step.ok
        )


# ── RebuildDerived (Phase 2) ─────────────────────────────────────────────

@dataclass(frozen=True)
class ListMemoryIdsRequest:
    """One keyset page request over Postgres memory ids (ordered by id).

    ``user_id`` is the tenant scope. ``None`` means "the worker's own
    single-tenant scope" (SOLO default user, or a v3 store with no
    tenancy); on a multi-tenant store the activity refuses it.
    """

    since: datetime | None = None
    namespace: str | None = None
    user_id: str | None = None
    after_id: str | None = None
    limit: int = DEFAULT_REBUILD_PAGE_SIZE


@dataclass(frozen=True)
class MemoryIdPage:
    """A page of ids; ``next_after_id`` is ``None`` on the last page.

    ``user_id`` is the scope the activity actually ran as (an explicit
    request scope echoed back, or the resolved SOLO default user) so
    every child ``DeriveMemory`` carries an explicit owner.
    """

    ids: tuple[str, ...] = ()
    next_after_id: str | None = None
    user_id: str | None = None


@dataclass(frozen=True)
class RebuildProgress:
    """Counters carried across continue-as-new runs (immutable; use ``with_*``)."""

    runs: int = 0
    pages: int = 0
    listed: int = 0
    ok: int = 0
    failed: int = 0
    skipped: int = 0

    def with_run(self) -> RebuildProgress:
        return _replace(self, runs=self.runs + 1)

    def with_page(self, count: int) -> RebuildProgress:
        return _replace(self, pages=self.pages + 1, listed=self.listed + count)

    def with_child_ok(self) -> RebuildProgress:
        return _replace(self, ok=self.ok + 1)

    def with_child_failed(self) -> RebuildProgress:
        return _replace(self, failed=self.failed + 1)

    def with_child_skipped(self) -> RebuildProgress:
        return _replace(self, skipped=self.skipped + 1)


@dataclass(frozen=True)
class RebuildRequest:
    """Input of ``RebuildDerived``; ``after_id`` + ``progress`` are the resume cursor.

    One rebuild run = one tenant (``user_id``). A multi-tenant store with
    no ``user_id`` fails loudly in the list activity — there is no
    cross-tenant rebuild; operators loop over users.
    """

    since: datetime | None = None
    namespace: str | None = None
    user_id: str | None = None
    page_size: int = DEFAULT_REBUILD_PAGE_SIZE
    window_size: int = DEFAULT_REBUILD_WINDOW_SIZE
    max_pages_per_run: int = DEFAULT_REBUILD_PAGES_PER_RUN
    policy: ActivityPolicy = field(default_factory=ActivityPolicy)
    after_id: str | None = None
    progress: RebuildProgress = field(default_factory=RebuildProgress)
    failed_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class RebuildOutcome:
    """Result of ``RebuildDerived`` (ids + counts only)."""

    progress: RebuildProgress
    done: bool = True
    failed_ids: tuple[str, ...] = ()


# ── Phase 3: governance jobs ─────────────────────────────────────────────
#
# ForgetUser saga, RetentionSweep, SessionSweep. Same payload rule: ids,
# labels and counts only — never memory content, never turn text.

FORGET_WORKFLOW = "ForgetUser"
WRITE_FORGET_AUDIT_ACTIVITY = "write_forget_audit"
FORGET_DOC_ACTIVITY = "forget_doc"
FORGET_VECTOR_ACTIVITY = "forget_vector"
FORGET_GRAPH_ACTIVITY = "forget_graph"
FORGET_STATE_ACTIVITY = "forget_state"
FORGET_WORKFLOW_ID_PREFIX = "forget-"

FORGET_LANE_AUDIT = "audit"
FORGET_LANE_DOC = "document"
FORGET_LANE_VECTOR = "vector"
FORGET_LANE_GRAPH = "graph"
FORGET_LANE_STATE = "state"
# Backend lanes, in the order the saga issues them (after the audit row).
FORGET_LANES: tuple[str, ...] = (
    FORGET_LANE_DOC, FORGET_LANE_VECTOR, FORGET_LANE_GRAPH, FORGET_LANE_STATE,
)

LANE_STATUS_PENDING = "pending"
LANE_STATUS_OK = "ok"
LANE_STATUS_FAILED = "failed"

RETENTION_WORKFLOW = "RetentionSweep"
LIST_DUE_POLICIES_ACTIVITY = "list_due_policies"
APPLY_POLICY_ACTIVITY = "apply_policy"
RETENTION_WORKFLOW_ID_PREFIX = "retention-sweep"
RETENTION_SWEEP_INITIATOR = "durable:retention-sweep"

SESSION_WORKFLOW = "SessionSweep"
SWEEP_IDLE_ACTIVITY = "sweep_idle"
SWEEP_ENDED_ACTIVITY = "sweep_ended"
SWEEP_ARCHIVED_ACTIVITY = "sweep_archived"
SESSION_WORKFLOW_ID_PREFIX = "session-sweep"
SESSION_SWEEP_ID_TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M"

SESSION_TRANSITION_IDLE = "idle"
SESSION_TRANSITION_ENDED = "ended"
SESSION_TRANSITION_ARCHIVED = "archived"

# Temporal Schedule ids (one per scheduled governance workflow).
RETENTION_SCHEDULE_ID = "attestor-retention-sweep"
SESSION_SCHEDULE_ID = "attestor-session-sweep"

# Session lifecycle thresholds (minutes). active → idle → ended → archived.
DEFAULT_SESSION_IDLE_AFTER_MINUTES = 30
DEFAULT_SESSION_ENDED_AFTER_MINUTES = 24 * 60
DEFAULT_SESSION_ARCHIVED_AFTER_MINUTES = 30 * 24 * 60
DEFAULT_SESSION_SWEEP_BATCH = 1000


def forget_workflow_id(user_id: str, audit_id: str) -> str:
    """``forget-{user_id}-{audit_id}`` — the audit id is minted by the caller
    so the saga's first activity can insert it idempotently."""
    return f"{FORGET_WORKFLOW_ID_PREFIX}{user_id}-{audit_id}"


def retention_sweep_workflow_id(day: date) -> str:
    """``retention-sweep-{YYYY-MM-DD}`` (the Schedule appends its own tick)."""
    return f"{RETENTION_WORKFLOW_ID_PREFIX}-{day.isoformat()}"


def session_sweep_workflow_id(at: datetime) -> str:
    """``session-sweep-{YYYY-MM-DDTHH:MM}`` in UTC (naive → UTC)."""
    utc = at if at.tzinfo is None else at.astimezone(timezone.utc)
    return f"{SESSION_WORKFLOW_ID_PREFIX}-{utc.strftime(SESSION_SWEEP_ID_TIMESTAMP_FORMAT)}"


@dataclass(frozen=True)
class ForgetRequest:
    """Input of ``ForgetUser``: who to forget, under which audit id."""

    user_id: str
    audit_id: str
    initiated_by: str | None = None
    policy: ActivityPolicy = field(default_factory=ActivityPolicy)


@dataclass(frozen=True)
class ForgetLaneOutcome:
    """One backend lane of the saga (counts only).

    ``skipped`` marks a backend that is absent or exposes no
    ``delete_by_user`` — still ``ok`` (nothing to delete there) but
    visible to an auditor.
    """

    lane: str
    status: str = LANE_STATUS_PENDING
    deleted: int = 0
    edges_deleted: int = 0
    skipped: bool = False
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == LANE_STATUS_OK


@dataclass(frozen=True)
class ForgetUserOutcome:
    """Per-backend result of the saga; also the ``progress`` query shape."""

    user_id: str
    audit_id: str
    audit: ForgetLaneOutcome
    document: ForgetLaneOutcome
    vector: ForgetLaneOutcome
    graph: ForgetLaneOutcome
    state: ForgetLaneOutcome

    @classmethod
    def pending(cls, user_id: str, audit_id: str) -> ForgetUserOutcome:
        return cls(
            user_id=user_id, audit_id=audit_id,
            audit=ForgetLaneOutcome(lane=FORGET_LANE_AUDIT),
            document=ForgetLaneOutcome(lane=FORGET_LANE_DOC),
            vector=ForgetLaneOutcome(lane=FORGET_LANE_VECTOR),
            graph=ForgetLaneOutcome(lane=FORGET_LANE_GRAPH),
            state=ForgetLaneOutcome(lane=FORGET_LANE_STATE),
        )

    @property
    def lanes(self) -> tuple[ForgetLaneOutcome, ...]:
        return (self.audit, self.document, self.vector, self.graph, self.state)

    @property
    def ok(self) -> bool:
        return all(lane.ok for lane in self.lanes)

    @property
    def failed_lanes(self) -> tuple[str, ...]:
        return tuple(lane.lane for lane in self.lanes if lane.status == LANE_STATUS_FAILED)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            f"{lane.lane}: {lane.error}" for lane in self.lanes
            if lane.status == LANE_STATUS_FAILED
        )

    @property
    def total_deleted(self) -> int:
        backends = (self.document, self.vector, self.graph, self.state)
        return sum(lane.deleted + lane.edges_deleted for lane in backends)

    def with_lane(self, outcome: ForgetLaneOutcome) -> ForgetUserOutcome:
        """New outcome with ``outcome.lane`` replaced (immutable update)."""
        field_by_lane = {
            FORGET_LANE_AUDIT: "audit", FORGET_LANE_DOC: "document",
            FORGET_LANE_VECTOR: "vector", FORGET_LANE_GRAPH: "graph",
            FORGET_LANE_STATE: "state",
        }
        name = field_by_lane.get(outcome.lane)
        if name is None:
            raise ValueError(f"unknown forget lane {outcome.lane!r}")
        return _replace(self, **{name: outcome})


@dataclass(frozen=True)
class RetentionSweepRequest:
    """Input of ``RetentionSweep`` (the Schedule passes the defaults)."""

    dry_run: bool = False
    initiated_by: str | None = RETENTION_SWEEP_INITIATOR
    policy: ActivityPolicy = field(default_factory=ActivityPolicy)


@dataclass(frozen=True)
class DuePolicy:
    """A retention policy the sweep will apply (policy metadata, not memory content)."""

    policy_id: str
    name: str
    action: str


@dataclass(frozen=True)
class DuePolicies:
    policies: tuple[DuePolicy, ...] = ()


@dataclass(frozen=True)
class ApplyPolicyRequest:
    policy_id: str
    dry_run: bool = False
    initiated_by: str | None = None


@dataclass(frozen=True)
class PolicyApplyOutcome:
    policy_id: str
    ok: bool
    action: str = ""
    matched: int = 0
    applied: int = 0
    vector_purged: int = 0
    error: str | None = None


@dataclass(frozen=True)
class RetentionSweepOutcome:
    policies_evaluated: int
    memories_archived: int
    memories_deleted: int
    dry_run: bool
    by_policy: tuple[PolicyApplyOutcome, ...] = ()

    @property
    def ok(self) -> bool:
        return all(p.ok for p in self.by_policy)

    @property
    def failed_policy_ids(self) -> tuple[str, ...]:
        return tuple(p.policy_id for p in self.by_policy if not p.ok)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(f"{p.policy_id}: {p.error}" for p in self.by_policy if not p.ok)


@dataclass(frozen=True)
class SessionSweepRequest:
    """Input of ``SessionSweep``: lifecycle thresholds in minutes."""

    idle_after_minutes: int = DEFAULT_SESSION_IDLE_AFTER_MINUTES
    ended_after_minutes: int = DEFAULT_SESSION_ENDED_AFTER_MINUTES
    archived_after_minutes: int = DEFAULT_SESSION_ARCHIVED_AFTER_MINUTES
    batch_limit: int = DEFAULT_SESSION_SWEEP_BATCH
    policy: ActivityPolicy = field(default_factory=ActivityPolicy)


@dataclass(frozen=True)
class SessionSweepStep:
    transition: str
    count: int = 0
    ok: bool = True
    error: str | None = None


@dataclass(frozen=True)
class SessionSweepOutcome:
    idle: SessionSweepStep
    ended: SessionSweepStep
    archived: SessionSweepStep

    @property
    def steps(self) -> tuple[SessionSweepStep, ...]:
        return (self.idle, self.ended, self.archived)

    @property
    def ok(self) -> bool:
        return all(s.ok for s in self.steps)

    @property
    def total(self) -> int:
        return sum(s.count for s in self.steps)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(f"{s.transition}: {s.error}" for s in self.steps if not s.ok)
