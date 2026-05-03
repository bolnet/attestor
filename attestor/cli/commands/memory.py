"""Memory CRUD CLI commands (split from cli.py).

Handlers: ``init``, ``add``, ``recall``, ``search``, ``list``, ``timeline``,
``stats``, ``export``, ``import``, ``inspect``, ``compact``, ``update``,
``forget``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from attestor.cli._setup_helpers import (
    _configure_claude_hooks,
    _configure_claude_mcp,
    _print_mcp_config,
)
from attestor.core import AgentMemory


def _cmd_init(args: argparse.Namespace) -> None:
    import shutil

    from attestor.init_wizard import init_store, init_store_interactive

    store_path = Path(args.path).resolve()
    store_path.mkdir(parents=True, exist_ok=True)

    config_toml = store_path / "config.toml"
    legacy_json = store_path / "config.json"
    fresh_store = not config_toml.exists() and not legacy_json.exists()

    if fresh_store:
        try:
            is_tty = sys.stdin.isatty() if hasattr(sys.stdin, "isatty") else False
            interactive = is_tty and not args.non_interactive
            if interactive:
                result = init_store_interactive(store_path, verify=args.verify)
            else:
                backend_options: dict | None = None
                if getattr(args, "postgres_url", None):
                    backend_options = {"url": args.postgres_url}
                result = init_store(
                    store_path,
                    backend=args.backend,
                    backend_options=backend_options,
                    verify=args.verify,
                )
            print(
                f"  config.toml: {result.config_path} "
                f"(backend={result.backend}, verified={result.verified})"
            )
        except (FileExistsError, ValueError, RuntimeError) as e:
            print(f"  init_wizard error: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"  config already present at {store_path}; leaving untouched")

    # Skip the legacy AgentMemory bootstrap if --verify already exercised it.
    if not args.verify:
        print(f"\nInitializing memory store at {store_path}...")
        try:
            import logging
            logging.getLogger("attestor").setLevel(logging.WARNING)
            mem = AgentMemory(str(store_path))
            print(f"  Store initialized at {store_path}")
            mem.close()
        except Exception as e:
            print(f"  Store created but error occurred: {e}")

    attestor_bin = shutil.which("attestor") or "attestor"
    abs_path = str(store_path)

    tool = args.tool
    if not tool:
        print("\n" + "=" * 50)
        print("MCP Configuration")
        print("=" * 50)
        _print_mcp_config("claude-code", attestor_bin, abs_path)
        _print_mcp_config("cursor", attestor_bin, abs_path)
    else:
        print()
        _print_mcp_config(tool, attestor_bin, abs_path)

    if args.install_mcp:
        _configure_claude_mcp(attestor_bin, abs_path)

    if args.hooks:
        _configure_claude_hooks(attestor_bin)

    print(f"Run 'attestor doctor {args.path}' to verify all components.")


def _cmd_add(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        memory = mem.add(
            content=args.content,
            tags=tags,
            category=args.category,
            entity=args.entity,
            namespace=args.namespace,
        )
        print(f"Added memory {memory.id}: {memory.content}")


def _cmd_recall(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        results = mem.recall(args.query, budget=args.budget, namespace=args.namespace)
        if not results:
            print("No relevant memories found.")
            return
        for r in results:
            m = r.memory
            print(f"[{r.match_source}:{r.score:.2f}] ({m.id}) {m.content}")
            if m.tags:
                print(f"  tags: {', '.join(m.tags)}")


def _cmd_search(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        results = mem.search(
            query=args.query,
            category=args.category,
            entity=args.entity,
            namespace=args.namespace,
            status=args.status,
            limit=args.limit,
        )
        if not results:
            print("No memories found.")
            return
        for m in results:
            status_marker = f"[{m.status}]" if m.status != "active" else ""
            print(f"({m.id}) {status_marker} {m.content}")
            if m.tags:
                print(f"  tags: {', '.join(m.tags)}  category: {m.category}")


def _cmd_list(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        memories = mem.search(
            status=args.status,
            category=args.category,
            namespace=args.namespace,
            limit=args.limit,
        )
        if not memories:
            print("No memories found.")
            return
        for m in memories:
            entity_str = f" [{m.entity}]" if m.entity else ""
            print(f"({m.id}) {m.category}{entity_str}: {m.content}")


def _cmd_timeline(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        memories = mem.timeline(args.entity, namespace=args.namespace)
        if not memories:
            print(f"No memories found for entity '{args.entity}'.")
            return
        print(f"Timeline for {args.entity}:")
        for m in memories:
            date = m.event_date or m.created_at
            status_marker = f" ({m.status})" if m.status != "active" else ""
            print(f"  {date[:10]}{status_marker}: {m.content}")


def _cmd_stats(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        s = mem.stats()
        print(f"Total memories: {s['total_memories']}")
        if "db_size_bytes" in s:
            print(f"Database size: {s['db_size_bytes']:,} bytes")
        if s["by_status"]:
            print("By status:")
            for status, count in s["by_status"].items():
                print(f"  {status}: {count}")
        if s["by_category"]:
            print("By category:")
            for cat, count in s["by_category"].items():
                print(f"  {cat}: {count}")


def _cmd_export(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        if args.output:
            mem.export_json(args.output)
            print(f"Exported to {args.output}")
        else:
            memories = mem.search(limit=1_000_000)
            print(json.dumps([m.to_dict() for m in memories], indent=2))


def _cmd_import(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        count = mem.import_json(args.file)
        print(f"Imported {count} memories")


def _cmd_inspect(args: argparse.Namespace) -> None:
    # v4 schema renamed `created_at` → `t_created`. Try the v4 column
    # first; fall back to v3 so existing pre-v4 installs keep working.
    with AgentMemory(args.path) as mem:
        try:
            rows = mem.execute(
                "SELECT id, content, tags, category, entity, status, t_created "
                "FROM memories ORDER BY t_created DESC LIMIT 50"
            )
        except Exception:
            rows = mem.execute(
                "SELECT id, content, tags, category, entity, status, created_at "
                "FROM memories ORDER BY created_at DESC LIMIT 50"
            )
        if not rows:
            print("No memories in store.")
            return
        for row in rows:
            print(f"{row['id']} | {row['status']:10s} | {row['category']:12s} | {row['content'][:60]}")


def _cmd_compact(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        count = mem.compact()
        print(f"Removed {count} archived memories")


def _cmd_update(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else None
        updated = mem.update(
            memory_id=args.memory_id,
            content=args.content,
            tags=tags,
            category=args.category,
            entity=args.entity,
        )
        if updated:
            print(f"Updated memory {updated.id}: {updated.content}")
        else:
            print(f"Memory {args.memory_id} not found")


def _cmd_forget(args: argparse.Namespace) -> None:
    with AgentMemory(args.path) as mem:
        if mem.forget(args.memory_id):
            print(f"Archived memory {args.memory_id}")
        else:
            print(f"Memory {args.memory_id} not found")
