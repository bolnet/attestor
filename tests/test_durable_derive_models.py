"""Phase 2 boundary models — DeriveMemory / RebuildDerived payloads.

Same rule as ``EpisodeRef``: workflow inputs and activity args are stored
in Temporal event history, so they carry IDENTIFIERS ONLY. Memory content
is re-read from Postgres inside the activity.
"""

from __future__ import annotations

import dataclasses
from datetime import datetime, timezone

import pytest

from attestor.durable.models import (
    DERIVE_STEP_GRAPH,
    DERIVE_STEP_VECTOR,
    MEMORY_CONTENT_FIELDS,
    ActivityPolicy,
    DeriveOutcome,
    DeriveRef,
    DeriveRequest,
    DeriveStepOutcome,
    ListMemoryIdsRequest,
    MemoryIdPage,
    RebuildOutcome,
    RebuildProgress,
    RebuildRequest,
    derive_workflow_id,
    rebuild_workflow_id,
)
from attestor.models import Memory


def _step(step: str, ok: bool = True, error: str | None = None) -> DeriveStepOutcome:
    return DeriveStepOutcome(memory_id="m-1", step=step, ok=ok, error=error)


@pytest.mark.unit
def test_derive_ref_is_frozen_and_carries_ids_only():
    ref = DeriveRef(memory_id="m-1", namespace="ns")
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.memory_id = "x"  # type: ignore[misc]
    names = {f.name for f in dataclasses.fields(DeriveRef)}
    # ids only: the row id, its namespace, its OWNER (RLS scope) and the
    # writer agent (private-visibility requester). Never content.
    assert names == {"memory_id", "namespace", "user_id", "agent_id"}
    assert not names & set(MEMORY_CONTENT_FIELDS)
    assert "content" in MEMORY_CONTENT_FIELDS
    assert DeriveRef(memory_id="m").user_id is None
    assert DeriveRef(memory_id="m").agent_id is None


@pytest.mark.unit
def test_derive_ref_from_memory_drops_content_and_keeps_owner_ids():
    mem = Memory(id="abc", content="secret conversation text", namespace="lme_1",
                 metadata={"_context_prefix": "private prefix"},
                 user_id="u-1", agent_id="planner")
    ref = DeriveRef.from_memory(mem)
    assert ref == DeriveRef(memory_id="abc", namespace="lme_1", user_id="u-1", agent_id="planner")
    assert "secret" not in repr(ref)
    assert "private" not in repr(ref)


@pytest.mark.unit
def test_rebuild_and_list_payloads_carry_tenant_scope():
    assert RebuildRequest().user_id is None
    assert RebuildRequest(user_id="u-1").user_id == "u-1"
    assert ListMemoryIdsRequest().user_id is None
    assert ListMemoryIdsRequest(user_id="u-1").user_id == "u-1"
    # The list activity reports the scope it resolved (SOLO default user)
    # so every child DeriveMemory carries an explicit owner.
    assert MemoryIdPage().user_id is None
    assert MemoryIdPage(ids=("a",), user_id="solo-1").user_id == "solo-1"


@pytest.mark.unit
def test_derive_request_defaults_to_plan_retry_policy():
    req = DeriveRequest(memory=DeriveRef(memory_id="m-1"))
    assert req.policy == ActivityPolicy()
    assert req.policy.initial_interval_seconds == 2.0
    assert req.policy.backoff_coefficient == 2.0
    assert req.policy.maximum_interval_seconds == 300.0
    assert req.policy.maximum_attempts == 0  # unlimited


@pytest.mark.unit
def test_derive_outcome_ok_requires_both_lanes():
    both = DeriveOutcome(
        memory_id="m-1", vector=_step(DERIVE_STEP_VECTOR), graph=_step(DERIVE_STEP_GRAPH),
    )
    assert both.ok
    assert both.errors == ()
    partial = DeriveOutcome(
        memory_id="m-1",
        vector=_step(DERIVE_STEP_VECTOR, ok=False, error="dimension mismatch"),
        graph=_step(DERIVE_STEP_GRAPH),
    )
    assert not partial.ok
    assert partial.errors == ("vector: dimension mismatch",)


@pytest.mark.unit
def test_workflow_ids_are_deterministic_and_prefixed():
    assert derive_workflow_id("m-1") == "derive-m-1"
    assert derive_workflow_id("m-1") == derive_workflow_id("m-1")
    started = datetime(2026, 9, 3, 14, 5, 9, tzinfo=timezone.utc)
    assert rebuild_workflow_id(started) == "rebuild-derived-20260903T140509Z"
    # Naive input is treated as UTC rather than guessed from the host zone.
    assert rebuild_workflow_id(started.replace(tzinfo=None)) == rebuild_workflow_id(started)


@pytest.mark.unit
def test_rebuild_request_defaults_and_cursor_fields():
    req = RebuildRequest()
    assert req.since is None
    assert req.namespace is None
    assert req.after_id is None
    assert req.page_size > 0
    assert req.window_size > 0
    assert req.max_pages_per_run > 0
    assert req.progress == RebuildProgress()
    assert req.policy == ActivityPolicy()


@pytest.mark.unit
def test_rebuild_progress_is_immutable_and_accumulates():
    p0 = RebuildProgress()
    p1 = p0.with_page(3)
    assert p0 == RebuildProgress()
    assert (p1.pages, p1.listed) == (1, 3)
    p2 = p1.with_child_ok().with_child_failed().with_child_skipped()
    assert (p2.ok, p2.failed, p2.skipped) == (1, 1, 1)
    assert (p1.ok, p1.failed, p1.skipped) == (0, 0, 0)


@pytest.mark.unit
def test_list_request_and_page_defaults():
    req = ListMemoryIdsRequest()
    assert req.since is None
    assert req.namespace is None
    assert req.after_id is None
    assert req.limit > 0
    page = MemoryIdPage()
    assert page.ids == ()
    assert page.next_after_id is None
    out = RebuildOutcome(progress=RebuildProgress())
    assert out.done is True
    assert out.failed_ids == ()
