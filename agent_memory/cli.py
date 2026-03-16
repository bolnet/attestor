"""CLI entry point for agent-memory."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from agent_memory.core import AgentMemory


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="agent-memory",
        description="Embedded memory for AI agents.",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # init
    p_init = subparsers.add_parser(
        "init",
        help="Initialize a new memory store with tool config",
    )
    p_init.add_argument("path", help="Directory path for the memory store")
    p_init.add_argument(
        "--tool",
        choices=["claude-code", "cursor"],
        default=None,
        help="Print MCP config for a specific tool",
    )

    # add
    p_add = subparsers.add_parser("add", help="Add a memory")
    p_add.add_argument("path", help="Memory store path")
    p_add.add_argument("content", help="Memory content")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--category", default="general", help="Category")
    p_add.add_argument("--entity", default=None, help="Entity name")

    # recall
    p_recall = subparsers.add_parser("recall", help="Recall relevant memories")
    p_recall.add_argument("path", help="Memory store path")
    p_recall.add_argument("query", help="Query string")
    p_recall.add_argument("--budget", type=int, default=2000, help="Token budget")

    # search
    p_search = subparsers.add_parser("search", help="Search memories with filters")
    p_search.add_argument("path", help="Memory store path")
    p_search.add_argument("query", nargs="?", default=None, help="Search query")
    p_search.add_argument("--category", default=None, help="Filter by category")
    p_search.add_argument("--entity", default=None, help="Filter by entity")
    p_search.add_argument("--status", default="active", help="Filter by status")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")

    # list
    p_list = subparsers.add_parser("list", help="List memories")
    p_list.add_argument("path", help="Memory store path")
    p_list.add_argument("--status", default="active", help="Filter by status")
    p_list.add_argument("--category", default=None, help="Filter by category")
    p_list.add_argument("--limit", type=int, default=20, help="Max results")

    # timeline
    p_timeline = subparsers.add_parser("timeline", help="Show entity timeline")
    p_timeline.add_argument("path", help="Memory store path")
    p_timeline.add_argument("--entity", required=True, help="Entity name")

    # stats
    p_stats = subparsers.add_parser("stats", help="Show store statistics")
    p_stats.add_argument("path", help="Memory store path")

    # export
    p_export = subparsers.add_parser("export", help="Export memories to JSON")
    p_export.add_argument("path", help="Memory store path")
    p_export.add_argument("--format", default="json", choices=["json"], help="Export format")
    p_export.add_argument("--output", "-o", default=None, help="Output file (default: stdout)")

    # import
    p_import = subparsers.add_parser("import", help="Import memories from JSON")
    p_import.add_argument("path", help="Memory store path")
    p_import.add_argument("file", help="JSON file to import")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect raw database")
    p_inspect.add_argument("path", help="Memory store path")

    # compact
    p_compact = subparsers.add_parser("compact", help="Remove archived memories")
    p_compact.add_argument("path", help="Memory store path")

    # forget
    p_forget = subparsers.add_parser("forget", help="Archive a memory")
    p_forget.add_argument("path", help="Memory store path")
    p_forget.add_argument("memory_id", help="Memory ID to archive")

    # serve (MCP server)
    p_serve = subparsers.add_parser("serve", help="Start MCP server")
    p_serve.add_argument("path", help="Memory store path")

    # setup-claude-code
    p_setup = subparsers.add_parser(
        "setup-claude-code",
        help="Print Claude Code MCP config for this memory store",
    )
    p_setup.add_argument("path", help="Memory store path")

    # doctor (health check)
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Check health of all components (SQLite, retrieval pipeline)",
    )
    p_doctor.add_argument("path", nargs="?", default=None, help="Memory store path")

    # locomo (LOCOMO benchmark)
    p_locomo = subparsers.add_parser(
        "locomo",
        help="Run LOCOMO benchmark (Long Conversation Memory)",
    )
    p_locomo.add_argument("--data", default=None, help="Path to locomo10.json (downloads if not provided)")
    p_locomo.add_argument(
        "--judge-model", default="openai/gpt-4.1-mini",
        help="LLM for judging answers",
    )
    p_locomo.add_argument(
        "--answer-model", default="openai/gpt-4.1-mini",
        help="LLM for answer synthesis",
    )
    p_locomo.add_argument(
        "--extraction-model", default="openai/gpt-4.1-mini",
        help="LLM for fact extraction",
    )
    p_locomo.add_argument("--use-extraction", action="store_true", help="Extract facts with LLM")
    p_locomo.add_argument("--resolve-pronouns", action="store_true", default=True, help="Resolve pronouns/time refs (default: on)")
    p_locomo.add_argument("--no-resolve-pronouns", action="store_false", dest="resolve_pronouns", help="Disable pronoun resolution")
    p_locomo.add_argument("--max-conversations", type=int, default=None, help="Max conversations")
    p_locomo.add_argument("--max-questions", type=int, default=None, help="Max questions per conversation")
    p_locomo.add_argument("--budget", type=int, default=4000, help="Recall token budget")
    p_locomo.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    p_locomo.add_argument("--output", "-o", default=None, help="Save results JSON to file")
    p_locomo.add_argument("--env-file", default=None, help="Path to .env file for API keys")

    # mab (MemoryAgentBench benchmark)
    p_mab = subparsers.add_parser(
        "mab",
        help="Run MemoryAgentBench benchmark (ICLR 2026)",
    )
    p_mab.add_argument(
        "--categories", nargs="+", default=["AR", "CR"],
        help="Category codes: AR, CR, TTL, LRU (default: AR CR)",
    )
    p_mab.add_argument(
        "--answer-model", default="openai/gpt-4.1-mini",
        help="LLM for answer synthesis (OpenRouter model ID)",
    )
    p_mab.add_argument("--max-examples", type=int, default=None, help="Max examples per category")
    p_mab.add_argument("--skip-examples", type=int, default=0, help="Skip first N examples per category")
    p_mab.add_argument("--max-questions", type=int, default=None, help="Max questions per example")
    p_mab.add_argument("--chunk-size", type=int, default=1024, help="Tokens per chunk")
    p_mab.add_argument("--context-max-tokens", type=int, default=None, help="Truncate context to N tokens")
    p_mab.add_argument("--budget", type=int, default=6000, help="Recall token budget")
    p_mab.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    p_mab.add_argument("--output", "-o", default=None, help="Save results JSON to file")
    p_mab.add_argument("--env-file", default=None, help="Path to .env file for API keys")

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        return

    # Dispatch
    handlers = {
        "init": _cmd_init,
        "add": _cmd_add,
        "recall": _cmd_recall,
        "search": _cmd_search,
        "list": _cmd_list,
        "timeline": _cmd_timeline,
        "stats": _cmd_stats,
        "export": _cmd_export,
        "import": _cmd_import,
        "inspect": _cmd_inspect,
        "compact": _cmd_compact,
        "forget": _cmd_forget,
        "serve": _cmd_serve,
        "setup-claude-code": _cmd_setup_claude_code,
        "doctor": _cmd_doctor,
        "locomo": _cmd_locomo,
        "mab": _cmd_mab,
    }
    handlers[args.command](args)


def _load_env_file(env_file_path: str):
    """Load environment variables from a .env file."""
    env_path = Path(env_file_path)
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("\"'")
            if value:
                os.environ[key] = value


def _cmd_init(args):
    import shutil

    store_path = Path(args.path).resolve()
    store_path.mkdir(parents=True, exist_ok=True)

    print(f"\nInitializing memory store at {store_path}...")
    try:
        import logging
        logging.getLogger("agent_memory").setLevel(logging.WARNING)
        mem = AgentMemory(str(store_path))
        print(f"  SQLite: {store_path}/memory.db")
        mem.close()
    except Exception as e:
        print(f"  Store created but error occurred: {e}")

    # Print MCP config
    agent_memory_bin = shutil.which("agent-memory") or "agent-memory"
    abs_path = str(store_path)

    tool = args.tool
    if not tool:
        print("\n" + "=" * 50)
        print("MCP Configuration")
        print("=" * 50)
        _print_mcp_config("claude-code", agent_memory_bin, abs_path)
        _print_mcp_config("cursor", agent_memory_bin, abs_path)
    else:
        print()
        _print_mcp_config(tool, agent_memory_bin, abs_path)

    print("Run 'agent-memory doctor %s' to verify all components." % args.path)


def _print_mcp_config(tool: str, binary: str, store_path: str):
    """Print MCP config for a specific tool."""
    if tool == "claude-code":
        config = {
            "mcpServers": {
                "memory": {
                    "command": binary,
                    "args": ["serve", store_path],
                }
            }
        }
        print(f"\nClaude Code -- add to .claude/settings.json:")
        print(json.dumps(config, indent=2))
    elif tool == "cursor":
        config = {
            "mcpServers": {
                "memory": {
                    "command": binary,
                    "args": ["serve", store_path],
                }
            }
        }
        print(f"\nCursor -- add to .cursor/mcp.json:")
        print(json.dumps(config, indent=2))


def _cmd_add(args):
    with AgentMemory(args.path) as mem:
        tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
        memory = mem.add(
            content=args.content,
            tags=tags,
            category=args.category,
            entity=args.entity,
        )
        print(f"Added memory {memory.id}: {memory.content}")


def _cmd_recall(args):
    with AgentMemory(args.path) as mem:
        results = mem.recall(args.query, budget=args.budget)
        if not results:
            print("No relevant memories found.")
            return
        for r in results:
            m = r.memory
            print(f"[{r.match_source}:{r.score:.2f}] ({m.id}) {m.content}")
            if m.tags:
                print(f"  tags: {', '.join(m.tags)}")


def _cmd_search(args):
    with AgentMemory(args.path) as mem:
        results = mem.search(
            query=args.query,
            category=args.category,
            entity=args.entity,
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


def _cmd_list(args):
    with AgentMemory(args.path) as mem:
        memories = mem.search(
            status=args.status,
            category=args.category,
            limit=args.limit,
        )
        if not memories:
            print("No memories found.")
            return
        for m in memories:
            entity_str = f" [{m.entity}]" if m.entity else ""
            print(f"({m.id}) {m.category}{entity_str}: {m.content}")


def _cmd_timeline(args):
    with AgentMemory(args.path) as mem:
        memories = mem.timeline(args.entity)
        if not memories:
            print(f"No memories found for entity '{args.entity}'.")
            return
        print(f"Timeline for {args.entity}:")
        for m in memories:
            date = m.event_date or m.created_at
            status_marker = f" ({m.status})" if m.status != "active" else ""
            print(f"  {date[:10]}{status_marker}: {m.content}")


def _cmd_stats(args):
    with AgentMemory(args.path) as mem:
        s = mem.stats()
        print(f"Total memories: {s['total_memories']}")
        print(f"Database size: {s['db_size_bytes']:,} bytes")
        if s["by_status"]:
            print("By status:")
            for status, count in s["by_status"].items():
                print(f"  {status}: {count}")
        if s["by_category"]:
            print("By category:")
            for cat, count in s["by_category"].items():
                print(f"  {cat}: {count}")


def _cmd_export(args):
    with AgentMemory(args.path) as mem:
        if args.output:
            mem.export_json(args.output)
            print(f"Exported to {args.output}")
        else:
            memories = mem.search(limit=1_000_000)
            print(json.dumps([m.to_dict() for m in memories], indent=2))


def _cmd_import(args):
    with AgentMemory(args.path) as mem:
        count = mem.import_json(args.file)
        print(f"Imported {count} memories")


def _cmd_inspect(args):
    with AgentMemory(args.path) as mem:
        rows = mem.execute("SELECT id, content, tags, category, entity, status, created_at FROM memories ORDER BY created_at DESC LIMIT 50")
        if not rows:
            print("No memories in store.")
            return
        for row in rows:
            print(f"{row['id']} | {row['status']:10s} | {row['category']:12s} | {row['content'][:60]}")


def _cmd_compact(args):
    with AgentMemory(args.path) as mem:
        count = mem.compact()
        print(f"Removed {count} archived memories")


def _cmd_forget(args):
    with AgentMemory(args.path) as mem:
        if mem.forget(args.memory_id):
            print(f"Archived memory {args.memory_id}")
        else:
            print(f"Memory {args.memory_id} not found")


def _cmd_serve(args):
    import asyncio
    try:
        from agent_memory.mcp.server import run_server
    except ImportError:
        print("MCP support requires: pip install agent-memory[mcp]")
        sys.exit(1)

    # Ensure the store exists
    AgentMemory(args.path).close()
    print(f"Starting MCP server for {args.path}...", file=sys.stderr)
    asyncio.run(run_server(args.path))


def _cmd_setup_claude_code(args):
    import shutil

    agent_memory_bin = shutil.which("agent-memory") or "agent-memory"
    abs_path = str(Path(args.path).resolve())

    print("Tip: Use 'agent-memory init %s' for full setup.\n" % args.path)
    _print_mcp_config("claude-code", agent_memory_bin, abs_path)
    _print_mcp_config("cursor", agent_memory_bin, abs_path)


def _cmd_doctor(args):
    """Check health of all components."""
    print("Memwright Doctor")
    print("=" * 50)

    store_path = getattr(args, "path", None)
    if store_path:
        try:
            import logging
            logging.getLogger("agent_memory").setLevel(logging.CRITICAL)
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
    print("\nUsage: agent-memory doctor <store-path>")
    print("Checks SQLite and retrieval pipeline health.")
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
        if check.get("version"):
            details.append(check["version"])
        if check.get("provider"):
            details.append(check["provider"])
        if check.get("model"):
            details.append(check["model"])
        if check.get("latency_ms") is not None:
            details.append(f"{check['latency_ms']}ms")
        if check.get("memory_count") is not None:
            details.append(f"{check['memory_count']} memories")
        if check.get("db_size_bytes") is not None:
            details.append(f"{check['db_size_bytes']:,} bytes")
        if check.get("active_layers") is not None:
            details.append(f"{check['active_layers']}/{check['max_layers']} layers")
        if check.get("note"):
            details.append(check["note"])

        detail_str = f" ({', '.join(details)})" if details else ""
        print(f"  [{icon}] {name}{detail_str}")

        if status == "error" and check.get("error"):
            print(f"     {check['error']}")

    print()


def _cmd_locomo(args):
    # Load env file if provided
    if args.env_file:
        _load_env_file(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")

    from agent_memory.locomo import run_locomo, print_locomo

    print("Running LOCOMO benchmark...")
    print("(Long Conversation Memory -- industry-standard benchmark)\n")

    results = run_locomo(
        data_path=args.data,
        judge_model=args.judge_model,
        answer_model=args.answer_model,
        extraction_model=args.extraction_model,
        use_extraction=args.use_extraction,
        resolve_pronouns=args.resolve_pronouns,
        max_conversations=args.max_conversations,
        max_questions_per_conv=args.max_questions,
        recall_budget=args.budget,
        verbose=args.verbose,
        api_key=api_key,
    )

    print_locomo(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


def _cmd_mab(args):
    # Load env file if provided
    if args.env_file:
        _load_env_file(args.env_file)
    api_key = os.environ.get("OPENROUTER_API_KEY")

    from agent_memory.mab import run_mab, print_mab

    print("Running MemoryAgentBench benchmark...")
    print("(ICLR 2026 benchmark -- Accurate Retrieval, Conflict Resolution, etc.)\n")

    results = run_mab(
        categories=args.categories,
        max_examples=args.max_examples,
        max_questions=args.max_questions,
        chunk_size=args.chunk_size,
        context_max_tokens=args.context_max_tokens,
        answer_model=args.answer_model,
        recall_budget=args.budget,
        verbose=args.verbose,
        api_key=api_key,
        skip_examples=args.skip_examples,
    )

    print_mab(results)

    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
