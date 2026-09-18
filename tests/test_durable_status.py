"""attestor.durable.status — probe + describe_workflow."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

pytest.importorskip("temporalio", reason="temporalio not installed (attestor[durable])")

from attestor.config import DurableCfg  # noqa: E402


@pytest.mark.unit
async def test_probe_reports_unreachable_without_raising(monkeypatch):
    from attestor.durable import client as durable_client
    from attestor.durable.config import DurableUnavailableError
    from attestor.durable.status import probe

    async def fail(cfg):
        raise DurableUnavailableError("connection refused")

    monkeypatch.setattr(durable_client, "connect", fail)
    out = await probe(DurableCfg(enabled=True, address="h:1", namespace="ns"))
    assert out.reachable is False
    assert out.address == "h:1"
    assert out.namespace == "ns"
    assert "connection refused" in (out.error or "")


@pytest.mark.unit
async def test_probe_reports_reachable(monkeypatch):
    from attestor.durable import client as durable_client
    from attestor.durable.status import probe

    async def ok(cfg):
        return object()

    monkeypatch.setattr(durable_client, "connect", ok)
    out = await probe(DurableCfg(enabled=True))
    assert out.reachable is True
    assert out.error is None


class _Desc:
    def __init__(self, status) -> None:
        self.status = status
        self.run_id = "run-1"
        self.start_time = datetime(2026, 9, 3, tzinfo=timezone.utc)
        self.close_time = None


class _Handle:
    def __init__(self, desc=None, error=None) -> None:
        self._desc, self._error = desc, error

    async def describe(self):
        if self._error:
            raise self._error
        return self._desc


class _Client:
    def __init__(self, handle: _Handle) -> None:
        self._handle = handle
        self.ids: list[str] = []

    def get_workflow_handle(self, workflow_id: str, **kw):
        self.ids.append(workflow_id)
        return self._handle


@pytest.mark.unit
async def test_describe_workflow_maps_status(monkeypatch):
    from temporalio.client import WorkflowExecutionStatus

    from attestor.durable import client as durable_client
    from attestor.durable.status import describe_workflow

    fake = _Client(_Handle(desc=_Desc(WorkflowExecutionStatus.RUNNING)))

    async def connect(cfg):
        return fake

    monkeypatch.setattr(durable_client, "connect", connect)
    st = await describe_workflow(DurableCfg(enabled=True), "consolidate-ep-1")
    assert fake.ids == ["consolidate-ep-1"]
    assert st.workflow_id == "consolidate-ep-1"
    assert st.status == "RUNNING"
    assert st.run_id == "run-1"
    assert st.close_time is None


@pytest.mark.unit
async def test_describe_workflow_unknown_status_and_not_found(monkeypatch):
    from temporalio.service import RPCError, RPCStatusCode

    from attestor.durable import client as durable_client
    from attestor.durable.config import DurableError
    from attestor.durable.status import describe_workflow

    async def connect_unknown(cfg):
        return _Client(_Handle(desc=_Desc(None)))

    monkeypatch.setattr(durable_client, "connect", connect_unknown)
    st = await describe_workflow(DurableCfg(enabled=True), "wf")
    assert st.status == "UNKNOWN"

    async def connect_missing(cfg):
        return _Client(_Handle(error=RPCError("nope", RPCStatusCode.NOT_FOUND, b"")))

    monkeypatch.setattr(durable_client, "connect", connect_missing)
    with pytest.raises(DurableError, match="not found"):
        await describe_workflow(DurableCfg(enabled=True), "wf-missing")
