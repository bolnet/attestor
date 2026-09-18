"""Durable boundary models — pure, frozen, sandbox-safe.

``attestor.durable.models`` is the only Attestor module a workflow may
touch, so it must import nothing but the stdlib.
"""

from __future__ import annotations

import ast
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import pytest

MODELS_PATH = (
    Path(__file__).resolve().parents[1] / "attestor" / "durable" / "models.py"
)


def _ts() -> datetime:
    return datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)


@pytest.mark.unit
def test_models_module_imports_only_stdlib():
    tree = ast.parse(MODELS_PATH.read_text())
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    banned = {"attestor", "temporalio", "psycopg2", "pinecone", "openai", "neo4j"}
    forbidden = imported & banned
    assert not forbidden, f"durable.models must stay pure; found {sorted(forbidden)}"


@pytest.mark.unit
def test_episode_ref_is_frozen_and_round_trips_from_queued_episode():
    from attestor.durable.models import EpisodeRef

    class FakeQueued:  # duck-typed QueuedEpisode
        id = "ep-1"
        user_id = "u-1"
        session_id = "s-1"
        thread_id = "t-1"
        user_turn_text = "hi"
        assistant_turn_text = "hello"
        user_ts = _ts()
        assistant_ts = _ts()
        project_id = "p-1"
        agent_id = "planner"

    ref = EpisodeRef.from_queued(FakeQueued())
    assert dataclasses.is_dataclass(ref)
    assert ref.episode_id == "ep-1"
    assert ref.user_id == "u-1"
    assert ref.project_id == "p-1"
    assert ref.agent_id == "planner"
    assert ref.user_ts == _ts()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ref.episode_id = "x"  # type: ignore[misc]


@pytest.mark.unit
def test_episode_ref_carries_ids_only_never_turn_text():
    """Workflow inputs land in Temporal history (a second, un-governed
    store). The boundary payload must therefore be identifiers only —
    raw conversation text is re-read from Postgres inside the activity."""
    from attestor.durable.models import EPISODE_CONTENT_FIELDS, EpisodeRef

    names = {f.name for f in dataclasses.fields(EpisodeRef)}
    assert EPISODE_CONTENT_FIELDS == ("user_turn_text", "assistant_turn_text")
    assert not names & set(EPISODE_CONTENT_FIELDS), names
    assert not any("text" in n or "content" in n for n in names), names


@pytest.mark.unit
def test_episode_ref_from_queued_ignores_turn_text():
    from attestor.durable.models import EpisodeRef

    class FakeQueued:
        id = "ep-2"
        user_id = "u-2"
        session_id = "s"
        thread_id = "t"
        user_turn_text = "my SSN is 123-45-6789"
        assistant_turn_text = "noted"
        user_ts = _ts()
        assistant_ts = _ts()
        project_id = None
        agent_id = None

    ref = EpisodeRef.from_queued(FakeQueued())
    assert "123-45-6789" not in repr(ref)
    assert "noted" not in repr(dataclasses.asdict(ref))


@pytest.mark.unit
def test_consolidate_request_carries_default_policy():
    from attestor.durable.models import (
        ActivityPolicy,
        ConsolidateRequest,
        EpisodeRef,
    )

    ref = EpisodeRef(
        episode_id="e", user_id="u", session_id="s", thread_id="t",
        user_ts=_ts(), assistant_ts=_ts(),
    )
    req = ConsolidateRequest(episode=ref)
    assert req.policy == ActivityPolicy()
    assert req.policy.maximum_attempts == 0  # unlimited; permanent errors are non-retryable
    assert req.policy.start_to_close_seconds > 0
    assert req.policy.backoff_coefficient >= 1.0


@pytest.mark.unit
def test_consolidation_outcome_defaults_and_ok():
    from attestor.durable.models import ConsolidationOutcome

    ok = ConsolidationOutcome(episode_id="e", ok=True)
    assert ok.error is None
    assert ok.written_memory_ids == ()
    bad = ConsolidationOutcome(episode_id="e", ok=False, error="boom")
    assert not bad.ok


@pytest.mark.unit
def test_workflow_id_helpers_are_deterministic():
    from attestor.durable.models import (
        CONSOLIDATE_EPISODE_ACTIVITY,
        CONSOLIDATE_WORKFLOW,
        consolidate_workflow_id,
    )

    assert consolidate_workflow_id("abc") == "consolidate-abc"
    assert consolidate_workflow_id("abc") == consolidate_workflow_id("abc")
    assert CONSOLIDATE_WORKFLOW == "ConsolidateEpisode"
    assert CONSOLIDATE_EPISODE_ACTIVITY == "consolidate_episode"
