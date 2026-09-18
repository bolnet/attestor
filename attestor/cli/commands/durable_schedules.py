# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``attestor durable schedules apply|list`` — Temporal Schedules for the sweeps.

Thin CLI over ``attestor.durable.schedules``; the async wrappers are
module attributes so tests patch them without a server.
"""

from __future__ import annotations

import asyncio
import sys
from typing import TYPE_CHECKING

from attestor.durable.config import DurableError

if TYPE_CHECKING:
    import argparse

    from attestor.config import DurableCfg
    from attestor.durable.schedules import ScheduleApplyResult, ScheduleView

EXIT_USAGE = 2
EXIT_FAILURE = 1
SCHEDULES_SUBCOMMANDS = ("apply", "list")
USAGE = "Usage: attestor durable schedules apply | attestor durable schedules list"
APPLY_HINT = "attestor durable schedules apply"


async def apply_schedules(cfg: DurableCfg) -> tuple[ScheduleApplyResult, ...]:
    from attestor.durable import client as durable_client
    from attestor.durable import schedules

    return await schedules.apply(await durable_client.connect(cfg), cfg)


async def list_schedules(cfg: DurableCfg) -> tuple[ScheduleView, ...]:
    from attestor.durable import client as durable_client
    from attestor.durable import schedules

    return await schedules.list_declared(await durable_client.connect(cfg), cfg)


def _fail(msg: str, *, code: int = EXIT_FAILURE) -> None:
    print(f"[attestor.durable] {msg}", file=sys.stderr)
    sys.exit(code)


def _print_apply(results: tuple[ScheduleApplyResult, ...]) -> None:
    for r in results:
        print(f"  {r.schedule_id:<28} {r.result:<8} cron={r.cron!r:<16} → {r.workflow}")


def _print_list(views: tuple[ScheduleView, ...]) -> None:
    missing = False
    for v in views:
        state = "present" if v.present else "missing"
        if v.present:
            extra = (
                f"paused={'true' if v.paused else 'false'} actions={v.num_actions} "
                f"next={v.next_action_at or '-'}"
            )
        else:
            extra = ""
            missing = True
        print(f"  {v.schedule_id:<28} {state:<8} cron={v.cron!r:<16} → {v.workflow} {extra}")
    if missing:
        print(f"  hint        run `{APPLY_HINT}` to create the missing schedule(s)")


def durable_schedules(args: argparse.Namespace, cfg: DurableCfg) -> None:
    """Dispatch ``apply`` / ``list``; ``cfg`` is already resolved + enabled."""
    sub = getattr(args, "schedules_cmd", None)
    if sub not in SCHEDULES_SUBCOMMANDS:
        print(USAGE, file=sys.stderr)
        sys.exit(EXIT_USAGE)
    runner = apply_schedules if sub == "apply" else list_schedules
    printer = _print_apply if sub == "apply" else _print_list
    try:
        results = asyncio.run(runner(cfg))
    except DurableError as exc:
        _fail(str(exc))
        return
    printer(results)
