"""Setup / health CLI commands: ``setup-claude-code`` and ``doctor``."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from attestor.cli._setup_helpers import (
    _configure_claude_hooks,
    _configure_claude_mcp,
    _print_mcp_config,
)
from attestor.core import AgentMemory


def _cmd_setup_claude_code(args: argparse.Namespace) -> None:
    import shutil

    attestor_bin = shutil.which("attestor") or "attestor"
    abs_path = str(Path(args.path).resolve())

    if not getattr(args, "install", False):
        print(f"Tip: Use 'attestor init {args.path} --install' for full setup.\n")
        _print_mcp_config("claude-code", attestor_bin, abs_path)
        _print_mcp_config("cursor", attestor_bin, abs_path)
        return

    _configure_claude_mcp(attestor_bin, abs_path)
    if getattr(args, "hooks", False):
        _configure_claude_hooks(attestor_bin)


def _cmd_doctor(args: argparse.Namespace) -> None:
    """Check health of all components."""
    print("Attestor Doctor")
    print("=" * 50)

    # v4 schema check is structural — runs against a Postgres connection
    # without needing an AgentMemory instance. Useful for pre-deploy
    # validation before any user data exists.
    if getattr(args, "v4_schema", False):
        url = getattr(args, "pg_url", None) or os.environ.get("PG_TEST_URL")
        if not url:
            print("\nv4 schema check requires --pg-url or PG_TEST_URL")
            return
        try:
            import psycopg2
        except ImportError:
            print("\nv4 schema check requires psycopg2 (install attestor[postgres])")
            return
        from attestor.doctor_v4 import format_v4_report, run_v4_doctor
        try:
            with psycopg2.connect(url) as conn:
                report = run_v4_doctor(conn)
        except Exception as e:
            print(f"\nFailed to connect to {url!r}: {e}")
            return
        print()
        print(format_v4_report(report))
        return

    store_path = getattr(args, "path", None)
    if store_path:
        try:
            import logging
            logging.getLogger("attestor").setLevel(logging.CRITICAL)
            mem = AgentMemory(store_path)
            report = mem.health()
            _print_health_report(report)
            mem.close()
            return
        except Exception as e:
            print(f"\nFailed to open store at {store_path}: {e}")
            print()
            return

    # No store path -- print usage
    print("\nUsage: attestor doctor <store-path>")
    print("Checks document store, vector store, graph store, and retrieval pipeline.")
    print()


def _print_health_report(report):
    """Pretty-print a health() report as individual check marks."""
    overall = "ALL HEALTHY" if report["healthy"] else "ISSUES DETECTED"
    print(f"\nOverall: {overall}\n")

    for check in report["checks"]:
        name = check["name"]
        status = check["status"]
        icon = "OK" if status == "ok" else "FAIL"

        # Build detail string
        details = []
        if check.get("latency_ms") is not None:
            details.append(f"{check['latency_ms']}ms")
        if check.get("memory_count") is not None:
            details.append(f"{check['memory_count']} memories")
        if check.get("db_size_bytes") is not None:
            details.append(f"{check['db_size_bytes']:,} bytes")
        if check.get("vector_count") is not None:
            details.append(f"{check['vector_count']} vectors")
        if check.get("nodes") is not None:
            details.append(f"{check['nodes']} nodes, {check.get('edges', 0)} edges")
        if check.get("active_layers") is not None:
            details.append(f"{check['active_layers']}/{check['max_layers']} layers")
        detail_str = f" ({', '.join(details)})" if details else ""
        print(f"  [{icon}] {name}{detail_str}")

        if check.get("note"):
            print(f"     ^ {check['note']}")
        if status == "error" and check.get("error"):
            print(f"     {check['error']}")

    print()
