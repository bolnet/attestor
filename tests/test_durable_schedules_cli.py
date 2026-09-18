"""CLI surface: ``attestor durable schedules apply|list``."""

from __future__ import annotations

import pytest

from attestor.cli import main
from attestor.config import DurableCfg

pytestmark = pytest.mark.unit


@pytest.fixture
def durable_cfg(monkeypatch):
    from attestor.cli.commands import durable as cmd

    holder: dict = {"cfg": DurableCfg(enabled=False)}

    def fake_resolve(cfg=None, *, env=None):
        return holder["cfg"]

    monkeypatch.setattr(cmd, "resolve_durable", fake_resolve)
    return holder


def test_schedules_without_subcommand_prints_usage(durable_cfg, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["durable", "schedules"])
    assert exc.value.code != 0


def test_schedules_apply_exits_loudly_when_disabled(durable_cfg, capsys):
    with pytest.raises(SystemExit) as exc:
        main(["durable", "schedules", "apply"])
    assert exc.value.code != 0
    assert "durable.enabled" in capsys.readouterr().err


def test_schedules_apply_prints_each_result(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable_schedules as cmd
    from attestor.durable.schedules import APPLY_CREATED, APPLY_UPDATED, ScheduleApplyResult

    durable_cfg["cfg"] = DurableCfg(enabled=True)
    seen: dict = {}

    async def fake_apply(cfg):
        seen["cfg"] = cfg
        return (
            ScheduleApplyResult(schedule_id="attestor-retention-sweep", cron="0 3 * * *",
                                workflow="RetentionSweep", result=APPLY_CREATED),
            ScheduleApplyResult(schedule_id="attestor-session-sweep", cron="*/10 * * * *",
                                workflow="SessionSweep", result=APPLY_UPDATED),
        )

    monkeypatch.setattr(cmd, "apply_schedules", fake_apply)
    main(["durable", "schedules", "apply", "--task-queue", "tq-z"])
    out = capsys.readouterr().out
    assert seen["cfg"].task_queue == "tq-z"
    assert "attestor-retention-sweep" in out
    assert "created" in out
    assert "attestor-session-sweep" in out
    assert "updated" in out
    assert "0 3 * * *" in out


def test_schedules_list_prints_presence(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable_schedules as cmd
    from attestor.durable.schedules import ScheduleView

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_list(cfg):
        return (
            ScheduleView(schedule_id="attestor-retention-sweep", workflow="RetentionSweep",
                         cron="0 3 * * *", present=True, paused=False, num_actions=3),
            ScheduleView(schedule_id="attestor-session-sweep", workflow="SessionSweep",
                         cron="*/10 * * * *", present=False),
        )

    monkeypatch.setattr(cmd, "list_schedules", fake_list)
    main(["durable", "schedules", "list"])
    out = capsys.readouterr().out
    assert "attestor-retention-sweep" in out
    assert "present" in out
    assert "attestor-session-sweep" in out
    assert "missing" in out
    assert "attestor durable schedules apply" in out  # hint for the missing one


def test_schedules_apply_reports_unreachable(durable_cfg, monkeypatch, capsys):
    pytest.importorskip("temporalio")
    from attestor.cli.commands import durable_schedules as cmd
    from attestor.durable.config import DurableUnavailableError

    durable_cfg["cfg"] = DurableCfg(enabled=True)

    async def fake_apply(cfg):
        raise DurableUnavailableError("Temporal server unreachable at x:1")

    monkeypatch.setattr(cmd, "apply_schedules", fake_apply)
    with pytest.raises(SystemExit) as exc:
        main(["durable", "schedules", "apply"])
    assert exc.value.code != 0
    assert "unreachable" in capsys.readouterr().err
