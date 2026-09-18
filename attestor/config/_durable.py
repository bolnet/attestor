# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Parser + validation for the top-level ``durable:`` YAML block.

Split out of ``loader.py`` to keep that module within its size budget.
Pure: no I/O, no env reads (``TEMPORAL_*`` passthrough happens at
runtime in ``attestor.durable.config.resolve_durable``).
"""

from __future__ import annotations

from typing import Any

from attestor.config.models import DurableCfg, DurableSchedulesCfg

_CRON_FIELDS = 5
_SCHEDULE_KEYS: tuple[str, ...] = ("retention_sweep", "session_sweep")
_REQUIRED_STRINGS: tuple[str, ...] = ("address", "namespace", "task_queue")


def _fail(msg: str) -> SystemExit:
    return SystemExit(f"[attestor.config] durable.{msg}")


def _non_empty(blk: dict[str, Any], key: str, default: str) -> str:
    value = str(blk.get(key, default)).strip()
    if not value:
        raise _fail(f"{key} must be a non-empty string")
    return value


def _cron(schedules: dict[str, Any], key: str, default: str) -> str:
    value = str(schedules.get(key, default)).strip()
    if len(value.split()) != _CRON_FIELDS:
        raise _fail(
            f"schedules.{key}={value!r} is not a {_CRON_FIELDS}-field cron spec"
        )
    return value


def parse_durable_block(raw: Any) -> DurableCfg:
    """Build a validated :class:`DurableCfg` from the YAML ``durable:`` block.

    ``None`` / missing block → defaults (``enabled=False``). Unknown keys
    are rejected so a typo (``task-queue``) cannot silently fall back.
    """
    blk: dict[str, Any] = dict(raw or {})
    known = {"enabled", "tls", "schedules", *_REQUIRED_STRINGS}
    unknown = sorted(set(blk) - known)
    if unknown:
        raise _fail(f"block has unknown key(s) {unknown}; expected {sorted(known)}")

    defaults = DurableCfg()
    schedules_blk: dict[str, Any] = dict(blk.get("schedules") or {})
    unknown_sched = sorted(set(schedules_blk) - set(_SCHEDULE_KEYS))
    if unknown_sched:
        raise _fail(
            f"schedules has unknown key(s) {unknown_sched}; "
            f"expected {list(_SCHEDULE_KEYS)}"
        )
    sched_defaults = defaults.schedules
    schedules = DurableSchedulesCfg(
        retention_sweep=_cron(
            schedules_blk, "retention_sweep", sched_defaults.retention_sweep
        ),
        session_sweep=_cron(
            schedules_blk, "session_sweep", sched_defaults.session_sweep
        ),
    )
    return DurableCfg(
        enabled=bool(blk.get("enabled", defaults.enabled)),
        address=_non_empty(blk, "address", defaults.address),
        namespace=_non_empty(blk, "namespace", defaults.namespace),
        task_queue=_non_empty(blk, "task_queue", defaults.task_queue),
        tls=bool(blk.get("tls", defaults.tls)),
        schedules=schedules,
    )
