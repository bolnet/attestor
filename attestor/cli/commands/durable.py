# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Durable (Temporal) CLI: ``attestor worker``, ``attestor durable status|rebuild|schedules``.

``temporalio`` is imported lazily inside the thin async wrappers below so
a plain install can still run ``attestor durable status`` and learn that
durable is disabled. The wrappers are module attributes on purpose —
tests patch them.
"""

from __future__ import annotations

import asyncio
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from attestor.durable.config import DurableError, apply_overrides, resolve_durable

if TYPE_CHECKING:
    import argparse

    from attestor.config import DurableCfg
    from attestor.durable.models import RebuildOutcome, RebuildRequest

EXIT_USAGE = 2
EXIT_FAILURE = 1
DURABLE_SUBCOMMANDS = ("status", "rebuild", "schedules")
USAGE = (
    "Usage: attestor durable status [WORKFLOW_ID] | "
    "attestor durable rebuild [--since ISO] [--namespace NS] [--user USER_ID] [--wait] | "
    "attestor durable schedules apply|list"
)
SOLO_SCOPE_LABEL = "(solo default user)"


# ── Lazy SDK-touching wrappers (patchable) ───────────────────────────────

async def run_worker(cfg: DurableCfg, *, store_path: str, activities: Any = None) -> None:
    from attestor.durable.worker import run_worker as _run_worker

    await _run_worker(cfg, store_path=store_path, activities=activities)


async def probe(cfg: DurableCfg) -> Any:
    from attestor.durable.status import probe as _probe

    return await _probe(cfg)


async def describe_workflow(cfg: DurableCfg, workflow_id: str) -> Any:
    from attestor.durable.status import describe_workflow as _describe

    return await _describe(cfg, workflow_id)


async def start_rebuild(cfg: DurableCfg, request: RebuildRequest, *, workflow_id: str) -> str:
    from attestor.durable import client as durable_client

    client = await durable_client.connect(cfg)
    return await durable_client.start_rebuild(client, cfg, request, workflow_id=workflow_id)


async def rebuild_result(cfg: DurableCfg, workflow_id: str) -> RebuildOutcome:
    """Await the rebuild; a FAILED run (e.g. refused tenant scope) is a ``DurableError``."""
    from temporalio.client import WorkflowFailureError

    from attestor.durable import client as durable_client

    client = await durable_client.connect(cfg)
    try:
        return await durable_client.rebuild_result(client, workflow_id)
    except WorkflowFailureError as exc:
        raise DurableError(f"workflow {workflow_id} failed: {exc.cause or exc}") from exc


# ── Helpers ──────────────────────────────────────────────────────────────

def _cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    # ``rebuild`` uses ``--namespace`` for the MEMORY namespace (plan §3
    # packaging) and ``--temporal-namespace`` for the server namespace.
    namespace = getattr(args, "temporal_namespace", None) or getattr(args, "namespace", None)
    return {
        "address": getattr(args, "address", None),
        "namespace": namespace,
        "task_queue": getattr(args, "task_queue", None),
    }


def parse_since(raw: str | None) -> datetime | None:
    """ISO-8601 → aware UTC datetime; naive input is taken as UTC. Loud on junk."""
    if raw is None:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"--since must be ISO-8601 (got {raw!r}): {exc}") from exc
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed


def _resolve(args: argparse.Namespace) -> DurableCfg:
    return apply_overrides(resolve_durable(), _cli_overrides(args))


def _print_config(cfg: DurableCfg) -> None:
    from attestor.config import resolved_config_path

    print("[attestor.durable]")
    print(f"  config      {resolved_config_path() or '(defaults)'}")
    print(f"  enabled     {'true' if cfg.enabled else 'false'}")
    print(f"  address     {cfg.address}")
    print(f"  namespace   {cfg.namespace}")
    print(f"  task_queue  {cfg.task_queue}")
    print(f"  tls         {'true' if cfg.tls else 'false'}")


def _fail(msg: str, *, code: int = EXIT_FAILURE) -> None:
    print(f"[attestor.durable] {msg}", file=sys.stderr)
    sys.exit(code)


# ── Handlers ─────────────────────────────────────────────────────────────

def _cmd_worker(args: argparse.Namespace) -> None:
    """Foreground Temporal worker on the governance task queue."""
    from attestor._paths import resolve_store_path

    cfg = _resolve(args)
    if not cfg.enabled:
        _fail(
            "durable.enabled is false in configs/attestor.yaml — set "
            "`durable.enabled: true` (and install `attestor[durable]`) "
            "before running `attestor worker`",
            code=EXIT_USAGE,
        )
    store_path = resolve_store_path(getattr(args, "path", None))
    print(
        f"[attestor.durable] worker → {cfg.address} namespace={cfg.namespace} "
        f"task_queue={cfg.task_queue} store={store_path}",
        file=sys.stderr,
    )
    try:
        asyncio.run(run_worker(cfg, store_path=store_path))
    except DurableError as exc:
        _fail(str(exc))
    except KeyboardInterrupt:
        print("[attestor.durable] worker stopped", file=sys.stderr)


def _cmd_durable(args: argparse.Namespace) -> None:
    """``attestor durable <subcommand>`` dispatcher."""
    sub = getattr(args, "durable_cmd", None)
    if sub not in DURABLE_SUBCOMMANDS:
        print(USAGE, file=sys.stderr)
        sys.exit(EXIT_USAGE)
    handlers = {"status": _durable_status, "rebuild": _durable_rebuild,
                "schedules": _durable_schedules}
    handlers[sub](args)


def _durable_status(args: argparse.Namespace) -> None:
    cfg = _resolve(args)
    _print_config(cfg)
    if not cfg.enabled:
        print("  state       disabled (in-process fallback; nothing to probe)")
        return
    health = asyncio.run(probe(cfg))
    print(f"  server      {'reachable' if health.reachable else 'UNREACHABLE'}")
    if not health.reachable:
        print(f"  error       {health.error}")
        sys.exit(EXIT_FAILURE)
    workflow_id = getattr(args, "workflow_id", None)
    if not workflow_id:
        return
    try:
        status = asyncio.run(describe_workflow(cfg, workflow_id))
    except DurableError as exc:
        _fail(str(exc))
        return
    print(f"  workflow    {status.workflow_id}")
    print(f"  status      {status.status}")
    print(f"  run_id      {status.run_id}")
    print(f"  started     {status.start_time}")
    print(f"  closed      {status.close_time}")


def _require_enabled_or_exit(cfg: DurableCfg, command: str) -> None:
    if not cfg.enabled:
        _fail(
            "durable.enabled is false in configs/attestor.yaml — set "
            "`durable.enabled: true` (and install `attestor[durable]`) "
            f"before running `attestor durable {command}`",
            code=EXIT_USAGE,
        )


def _rebuild_request(args: argparse.Namespace) -> RebuildRequest:
    from attestor.durable.models import RebuildRequest

    overrides = {
        "since": parse_since(getattr(args, "since", None)),
        "namespace": getattr(args, "memory_namespace", None),
        "user_id": getattr(args, "user_id", None),
        "page_size": getattr(args, "page_size", None),
        "window_size": getattr(args, "window", None),
    }
    return RebuildRequest(**{k: v for k, v in overrides.items() if v is not None})


def _print_rebuild_outcome(outcome: RebuildOutcome) -> None:
    p = outcome.progress
    print(f"  runs        {p.runs}")
    print(f"  pages       {p.pages}")
    print(f"  listed      {p.listed}")
    print(f"  ok          {p.ok}")
    print(f"  failed      {p.failed}")
    print(f"  skipped     {p.skipped}")
    if outcome.failed_ids:
        print(f"  failed_ids  {', '.join(outcome.failed_ids)}")


def _durable_rebuild(args: argparse.Namespace) -> None:
    """``attestor durable rebuild`` — start ``RebuildDerived`` (optionally wait)."""
    from attestor.durable.models import rebuild_workflow_id

    cfg = _resolve(args)
    _require_enabled_or_exit(cfg, "rebuild")
    try:
        request = _rebuild_request(args)
    except ValueError as exc:
        _fail(str(exc), code=EXIT_USAGE)
        return
    workflow_id = rebuild_workflow_id(datetime.now(timezone.utc))
    _print_config(cfg)
    print(f"  since       {request.since.isoformat() if request.since else '(all)'}")
    print(f"  namespace   {request.namespace or '(all)'}")
    print(f"  user        {request.user_id or SOLO_SCOPE_LABEL}")
    print(f"  page_size   {request.page_size}")
    print(f"  window      {request.window_size}")
    try:
        started = asyncio.run(start_rebuild(cfg, request, workflow_id=workflow_id))
    except DurableError as exc:
        _fail(str(exc))
        return
    print(f"  workflow    {started}")
    if not getattr(args, "wait", False):
        print(f"  follow      attestor durable status {started}")
        return
    try:
        outcome = asyncio.run(rebuild_result(cfg, started))
    except DurableError as exc:
        _fail(str(exc))
        return
    _print_rebuild_outcome(outcome)


def _durable_schedules(args: argparse.Namespace) -> None:
    """``attestor durable schedules apply|list``."""
    from attestor.cli.commands.durable_schedules import durable_schedules

    cfg = _resolve(args)
    _require_enabled_or_exit(cfg, "schedules")
    durable_schedules(args, cfg)
