"""AgentMemory.forget_user() — durable saga when enabled, in-process otherwise.

No Postgres / Temporal: the in-process path is patched at the
``compliance.retention.forget_user`` seam, the durable start at
``attestor.core.forget_service.start_forget``.
"""

from __future__ import annotations

from typing import Any

import pytest

from attestor.compliance.retention import ForgetUserResult
from attestor.config import DurableCfg
from attestor.core import forget_service
from attestor.core.agent_memory import AgentMemory

pytestmark = pytest.mark.unit


class _Mem(AgentMemory):
    """Skip the store bootstrap; only ``forget_user`` is exercised."""

    def __init__(self) -> None:  # noqa: D401 - deliberately not calling super
        self._v4 = True

    def _require_v4(self) -> None:
        return None


@pytest.fixture
def seams(monkeypatch):
    from attestor.compliance import retention

    holder: dict[str, Any] = {"cfg": None, "in_process": [], "starts": [], "raise": None}

    def fake_forget(mem, user_id, *, dry_run=False, initiated_by=None):
        holder["in_process"].append((user_id, dry_run, initiated_by))
        return ForgetUserResult(user_id=user_id, doc_rows_deleted=1, vector_rows_deleted=1,
                                graph_nodes_deleted=1, graph_edges_deleted=1,
                                state_rows_deleted=1, elapsed_ms=1.0, audit_id="a-in",
                                dry_run=dry_run)

    def fake_start(cfg, request):
        holder["starts"].append((cfg, request))
        if holder["raise"]:
            raise holder["raise"]
        return f"forget-{request.user_id}-{request.audit_id}"

    monkeypatch.setattr(retention, "forget_user", fake_forget)
    monkeypatch.setattr(forget_service, "resolve_durable_cfg", lambda: holder["cfg"])
    monkeypatch.setattr(forget_service, "start_forget", fake_start)
    return holder


def test_disabled_runs_in_process_unchanged(seams):
    seams["cfg"] = DurableCfg(enabled=False)
    out = _Mem().forget_user("u-1", initiated_by="ops")
    assert seams["in_process"] == [("u-1", False, "ops")]
    assert seams["starts"] == []
    assert out["audit_id"] == "a-in"
    assert out["doc_rows_deleted"] == 1
    assert out["durable"] is False


def test_no_config_runs_in_process(seams):
    seams["cfg"] = None
    _Mem().forget_user("u-1")
    assert len(seams["in_process"]) == 1
    assert seams["starts"] == []


def test_enabled_starts_saga_and_returns_workflow_id(seams):
    seams["cfg"] = DurableCfg(enabled=True, task_queue="tq")
    out = _Mem().forget_user("u-2", initiated_by="ops")
    assert seams["in_process"] == []
    assert len(seams["starts"]) == 1
    cfg, request = seams["starts"][0]
    assert cfg.task_queue == "tq"
    assert request.user_id == "u-2"
    assert request.initiated_by == "ops"
    assert request.audit_id
    assert len(request.audit_id) >= 8
    assert out["workflow_id"] == f"forget-u-2-{request.audit_id}"
    assert out["audit_id"] == request.audit_id
    assert out["durable"] is True
    assert out["dry_run"] is False
    assert out["user_id"] == "u-2"
    assert "attestor durable status" in out["follow"]


def test_dry_run_never_goes_durable(seams):
    seams["cfg"] = DurableCfg(enabled=True)
    out = _Mem().forget_user("u-3", dry_run=True)
    assert seams["starts"] == []
    assert seams["in_process"] == [("u-3", True, None)]
    assert out["dry_run"] is True


def test_enabled_but_unreachable_fails_loudly_without_deleting(seams):
    from attestor.durable.config import DurableUnavailableError

    seams["cfg"] = DurableCfg(enabled=True)
    seams["raise"] = DurableUnavailableError("Temporal server unreachable at x:1")
    with pytest.raises(DurableUnavailableError, match="unreachable"):
        _Mem().forget_user("u-4")
    assert seams["in_process"] == []  # no silent in-process fallback for a GDPR delete


def test_audit_ids_are_unique_per_call(seams):
    seams["cfg"] = DurableCfg(enabled=True)
    a = _Mem().forget_user("u-5")["audit_id"]
    b = _Mem().forget_user("u-5")["audit_id"]
    assert a != b
