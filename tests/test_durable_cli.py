"""CLI surface: ``attestor worker`` and ``attestor durable status``."""

from __future__ import annotations

import pytest

from attestor.cli import main
from attestor.config import DurableCfg


@pytest.fixture
def durable_cfg(monkeypatch):
    """Pin the durable config the CLI resolves, independent of YAML."""
    from attestor.cli.commands import durable as cmd

    holder: dict = {"cfg": DurableCfg(enabled=False)}

    def fake_resolve(cfg=None, *, env=None, overrides=None):
        return cmd.apply_overrides(holder["cfg"], overrides or {})

    monkeypatch.setattr(cmd, "resolve_durable", fake_resolve)
    return holder


@pytest.mark.unit
def test_worker_exits_loudly_when_disabled(durable_cfg, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["worker"])
    assert exc.value.code != 0
    assert "durable.enabled" in capsys.readouterr().err


@pytest.mark.unit
def test_worker_exits_loudly_when_unreachable(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable as cmd
    from attestor.durable.config import DurableUnavailableError

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_run_worker(cfg, *, store_path, activities=None):
        raise DurableUnavailableError(f"cannot reach {cfg.address}")

    monkeypatch.setattr(cmd, "run_worker", fake_run_worker)
    with pytest.raises(SystemExit) as exc:
        main(["worker", "--address", "temporal.example:7233"])
    assert exc.value.code != 0
    assert "temporal.example:7233" in capsys.readouterr().err


@pytest.mark.unit
def test_worker_passes_overrides_to_run_worker(durable_cfg, monkeypatch, tmp_path):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable as cmd

    durable_cfg["cfg"] = DurableCfg(enabled=True)
    seen: dict = {}

    async def fake_run_worker(cfg, *, store_path, activities=None):
        seen["cfg"] = cfg
        seen["store_path"] = store_path

    monkeypatch.setattr(cmd, "run_worker", fake_run_worker)
    main(["worker", "--path", str(tmp_path), "--task-queue", "tq-x", "--namespace", "ns-x"])
    assert seen["cfg"].task_queue == "tq-x"
    assert seen["cfg"].namespace == "ns-x"
    assert seen["store_path"] == str(tmp_path)


@pytest.mark.unit
def test_durable_status_reports_disabled(durable_cfg, capsys):
    main(["durable", "status"])
    out = capsys.readouterr().out
    assert "disabled" in out
    assert "localhost:7233" in out


@pytest.mark.unit
def test_durable_status_reports_unreachable_with_nonzero_exit(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable as cmd
    from attestor.durable.status import DurableProbe

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_probe(cfg):
        return DurableProbe(
            address=cfg.address, namespace=cfg.namespace, reachable=False, error="refused",
        )

    monkeypatch.setattr(cmd, "probe", fake_probe)
    with pytest.raises(SystemExit) as exc:
        main(["durable", "status"])
    assert exc.value.code != 0
    assert "refused" in capsys.readouterr().out


@pytest.mark.unit
def test_durable_status_describes_workflow(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable as cmd
    from attestor.durable.status import DurableProbe, WorkflowStatus

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_probe(cfg):
        return DurableProbe(address=cfg.address, namespace=cfg.namespace, reachable=True)

    async def fake_describe(cfg, workflow_id):
        return WorkflowStatus(workflow_id=workflow_id, status="COMPLETED", run_id="r-1")

    monkeypatch.setattr(cmd, "probe", fake_probe)
    monkeypatch.setattr(cmd, "describe_workflow", fake_describe)
    main(["durable", "status", "consolidate-ep-1"])
    out = capsys.readouterr().out
    assert "consolidate-ep-1" in out
    assert "COMPLETED" in out


@pytest.mark.unit
def test_durable_without_subcommand_prints_usage_and_exits(durable_cfg):
    with pytest.raises(SystemExit):
        main(["durable"])


# ── attestor durable rebuild ─────────────────────────────────────────────

@pytest.mark.unit
def test_durable_rebuild_exits_loudly_when_disabled(durable_cfg, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["durable", "rebuild"])
    assert exc.value.code != 0
    assert "durable.enabled" in capsys.readouterr().err


@pytest.mark.unit
def test_durable_rebuild_rejects_bad_since(durable_cfg, capsys):
    durable_cfg["cfg"] = DurableCfg(enabled=True)
    with pytest.raises(SystemExit) as exc:
        main(["durable", "rebuild", "--since", "yesterday"])
    assert exc.value.code != 0
    assert "--since" in capsys.readouterr().err


@pytest.mark.unit
def test_durable_rebuild_starts_workflow_with_filters(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from datetime import datetime, timezone

    from attestor.cli.commands import durable as cmd

    durable_cfg["cfg"] = DurableCfg(enabled=True)
    seen: dict = {}

    async def fake_start_rebuild(cfg, request, *, workflow_id):
        seen["cfg"] = cfg
        seen["request"] = request
        seen["workflow_id"] = workflow_id
        return workflow_id

    monkeypatch.setattr(cmd, "start_rebuild", fake_start_rebuild)
    main([
        "durable", "rebuild", "--since", "2026-01-02T03:04:05", "--namespace", "lme_7",
        "--user", "u-1", "--temporal-namespace", "tns", "--page-size", "50", "--window", "4",
    ])
    out = capsys.readouterr().out
    req = seen["request"]
    assert req.since == datetime(2026, 1, 2, 3, 4, 5, tzinfo=timezone.utc)  # naive → UTC
    assert req.namespace == "lme_7"
    assert req.user_id == "u-1"  # tenant scope: the worker runs every derive as this owner
    assert "u-1" in out
    assert req.page_size == 50
    assert req.window_size == 4
    assert seen["cfg"].namespace == "tns"  # Temporal namespace override, not memory namespace
    assert seen["workflow_id"].startswith("rebuild-derived-")
    assert seen["workflow_id"] in out
    assert "attestor durable status" in out


@pytest.mark.unit
def test_durable_rebuild_wait_prints_outcome(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable as cmd
    from attestor.durable.models import RebuildOutcome, RebuildProgress

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_start_rebuild(cfg, request, *, workflow_id):
        return workflow_id

    async def fake_rebuild_result(cfg, workflow_id):
        return RebuildOutcome(progress=RebuildProgress(pages=2, listed=9, ok=8, failed=1, runs=1),
                              failed_ids=("m-3",))

    monkeypatch.setattr(cmd, "start_rebuild", fake_start_rebuild)
    monkeypatch.setattr(cmd, "rebuild_result", fake_rebuild_result)
    main(["durable", "rebuild", "--wait"])
    out = capsys.readouterr().out
    assert "user        (solo default user)" in out
    assert "listed      9" in out
    assert "ok          8" in out
    assert "failed      1" in out
    assert "m-3" in out


@pytest.mark.unit
def test_durable_rebuild_wait_reports_failed_workflow_loudly(durable_cfg, monkeypatch, capsys):
    """A rebuild refused by the worker (multi-tenant store, no --user) must
    surface as a clean non-zero exit, not a traceback."""
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable as cmd
    from attestor.durable.config import DurableError

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_start_rebuild(cfg, request, *, workflow_id):
        return workflow_id

    async def fake_rebuild_result(cfg, workflow_id):
        raise DurableError(f"workflow {workflow_id} failed: rebuild needs --user")

    monkeypatch.setattr(cmd, "start_rebuild", fake_start_rebuild)
    monkeypatch.setattr(cmd, "rebuild_result", fake_rebuild_result)
    with pytest.raises(SystemExit) as exc:
        main(["durable", "rebuild", "--wait"])
    assert exc.value.code != 0
    assert "--user" in capsys.readouterr().err


@pytest.mark.unit
def test_durable_rebuild_reports_unreachable(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable as cmd
    from attestor.durable.config import DurableUnavailableError

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_start_rebuild(cfg, request, *, workflow_id):
        raise DurableUnavailableError("Temporal server unreachable at localhost:7233")

    monkeypatch.setattr(cmd, "start_rebuild", fake_start_rebuild)
    with pytest.raises(SystemExit) as exc:
        main(["durable", "rebuild"])
    assert exc.value.code != 0
    assert "unreachable" in capsys.readouterr().err
