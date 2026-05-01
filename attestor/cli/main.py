"""Top-level CLI entry — argparse setup and subcommand dispatch.

Imports the per-domain ``_cmd_*`` handlers from
:mod:`attestor.cli.commands` and wires them into a single argparse
``ArgumentParser``.  The handler map is intentionally identical (same keys,
same callables) to the pre-split ``attestor/cli.py`` so behavior is
byte-identical.
"""

from __future__ import annotations

import argparse

from attestor.cli._common import _add_backend_args, _suppress_noisy_output
from attestor.cli.commands.bench import (
    _cmd_locomo,
    _cmd_longmemeval,
    _cmd_mab,
)
from attestor.cli.commands.memory import (
    _cmd_add,
    _cmd_compact,
    _cmd_export,
    _cmd_forget,
    _cmd_import,
    _cmd_init,
    _cmd_inspect,
    _cmd_list,
    _cmd_recall,
    _cmd_search,
    _cmd_stats,
    _cmd_timeline,
    _cmd_update,
)
from attestor.cli.commands.server import (
    _cmd_api,
    _cmd_hook,
    _cmd_mcp_serve,
    _cmd_serve,
    _cmd_ui,
)
from attestor.cli.commands.setup import _cmd_doctor, _cmd_setup_claude_code


def main(argv=None):
    _suppress_noisy_output()
    parser = argparse.ArgumentParser(
        prog="attestor",
        description="Memory for AI agents (Postgres + Neo4j).",
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
    p_init.add_argument(
        "--hooks",
        action="store_true",
        help="Auto-configure Claude Code lifecycle hooks in settings.json",
    )
    p_init.add_argument(
        "--backend",
        choices=["postgres", "arangodb"],
        default="postgres",
        help="Backend to record in config.toml (default: postgres)",
    )
    p_init.add_argument(
        "--verify",
        action="store_true",
        help="Run health check after init; roll back config.toml on failure",
    )
    p_init.add_argument(
        "--non-interactive",
        action="store_true",
        help="Skip interactive prompts; use only CLI flags",
    )
    p_init.add_argument(
        "--install",
        action="store_true",
        dest="install_mcp",
        help="Write MCP server entry into ~/.claude/settings.json",
    )

    # add
    p_add = subparsers.add_parser("add", help="Add a memory")
    p_add.add_argument("path", help="Memory store path")
    p_add.add_argument("content", help="Memory content")
    p_add.add_argument("--tags", default="", help="Comma-separated tags")
    p_add.add_argument("--category", default="general", help="Category")
    p_add.add_argument("--entity", default=None, help="Entity name")
    p_add.add_argument("--namespace", default="default", help="Namespace for isolation")

    # recall
    p_recall = subparsers.add_parser("recall", help="Recall relevant memories")
    p_recall.add_argument("path", help="Memory store path")
    p_recall.add_argument("query", help="Query string")
    p_recall.add_argument("--budget", type=int, default=16000, help="Token budget")
    p_recall.add_argument("--namespace", default=None, help="Namespace filter")

    # search
    p_search = subparsers.add_parser("search", help="Search memories with filters")
    p_search.add_argument("path", help="Memory store path")
    p_search.add_argument("query", nargs="?", default=None, help="Search query")
    p_search.add_argument("--category", default=None, help="Filter by category")
    p_search.add_argument("--entity", default=None, help="Filter by entity")
    p_search.add_argument("--status", default="active", help="Filter by status")
    p_search.add_argument("--limit", type=int, default=10, help="Max results")
    p_search.add_argument("--namespace", default=None, help="Namespace filter")

    # list
    p_list = subparsers.add_parser("list", help="List memories")
    p_list.add_argument("path", help="Memory store path")
    p_list.add_argument("--status", default="active", help="Filter by status")
    p_list.add_argument("--category", default=None, help="Filter by category")
    p_list.add_argument("--limit", type=int, default=20, help="Max results")
    p_list.add_argument("--namespace", default=None, help="Namespace filter")

    # timeline
    p_timeline = subparsers.add_parser("timeline", help="Show entity timeline")
    p_timeline.add_argument("path", help="Memory store path")
    p_timeline.add_argument("--entity", required=True, help="Entity name")
    p_timeline.add_argument("--namespace", default=None, help="Namespace filter")

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

    # update
    p_update = subparsers.add_parser("update", help="Update a memory's content")
    p_update.add_argument("path", help="Memory store path")
    p_update.add_argument("memory_id", help="Memory ID to update")
    p_update.add_argument("content", help="New content")
    p_update.add_argument("--tags", default=None, help="New comma-separated tags")
    p_update.add_argument("--category", default=None, help="New category")
    p_update.add_argument("--entity", default=None, help="New entity name")

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
    p_setup.add_argument(
        "--install",
        action="store_true",
        help="Write MCP entry into ~/.claude/settings.json (default: print only)",
    )
    p_setup.add_argument(
        "--hooks",
        action="store_true",
        help="Also write Claude Code lifecycle hooks into settings.json",
    )

    # doctor (health check)
    p_doctor = subparsers.add_parser(
        "doctor",
        help="Check health of all components (document, vector, graph, retrieval)",
    )
    p_doctor.add_argument("path", nargs="?", default=None, help="Memory store path")
    p_doctor.add_argument(
        "--v4-schema",
        action="store_true",
        help=(
            "Run the v4 schema invariants check against PG_TEST_URL "
            "(or --pg-url): extensions, RLS policies on tenant tables, "
            "deletion_audit RLS-EXEMPT, content_tsv + quota counter triggers"
        ),
    )
    p_doctor.add_argument(
        "--pg-url",
        default=None,
        help="Postgres connection URL for --v4-schema (overrides PG_TEST_URL)",
    )

    # locomo (LOCOMO benchmark)
    p_locomo = subparsers.add_parser(
        "locomo",
        help="Run LOCOMO benchmark (Long Conversation Memory)",
    )
    p_locomo.add_argument("--data", default=None, help="Path to locomo10.json (downloads if not provided)")
    p_locomo.add_argument(
        "--judge-model", default=None,
        help="LLM for judging answers (default: stack.models.judge from configs/attestor.yaml)",
    )
    p_locomo.add_argument(
        "--answer-model", default=None,
        help="LLM for answer synthesis (default: stack.models.answerer from configs/attestor.yaml)",
    )
    p_locomo.add_argument(
        "--extraction-model", default=None,
        help="LLM for fact extraction (default: stack.models.extraction from configs/attestor.yaml)",
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
    _add_backend_args(p_locomo)

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
        "--answer-model", default=None,
        help="LLM for answer synthesis (default: stack.models.benchmark_default from configs/attestor.yaml)",
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
    _add_backend_args(p_mab)

    # longmemeval (LongMemEval benchmark — ICLR 2025, temporal-reasoning focus)
    p_lme = subparsers.add_parser(
        "longmemeval",
        help="Run LongMemEval benchmark (Wu et al., ICLR 2025)",
    )
    p_lme.add_argument(
        "--data", default=None,
        help="Path to longmemeval_s_cleaned.json (auto-downloads if not set)",
    )
    p_lme.add_argument(
        "--fixture", action="store_true",
        help="Use the bundled 6-sample mini fixture (for smoke runs)",
    )
    p_lme.add_argument(
        "--variant", default="s", choices=["s", "m", "oracle"],
        help="HuggingFace dataset variant when auto-downloading (default: s)",
    )
    p_lme.add_argument(
        "--answer-model", default=None,
        help="LLM for answer synthesis (default: stack.models.answerer from configs/attestor.yaml)",
    )
    p_lme.add_argument(
        "--judge-model", action="append", default=None,
        help="LLM for judging answers (default: [stack.models.judge, stack.models.verifier]). "
             "Pass multiple times for dual-judge scoring.",
    )
    p_lme.add_argument(
        "--use-extraction", action="store_true",
        help="Extract atomic facts with LLM during ingest (slower, more accurate)",
    )
    p_lme.add_argument(
        "--use-distillation", action="store_true",
        help="Per-turn LLM distillation before storage — canonicalize each turn "
             "into 0..N third-person facts with absolute dates before embed/graph.",
    )
    p_lme.add_argument(
        "--distill-model", default=None,
        help="LLM for the per-turn distiller (default: stack.models.distill from configs/attestor.yaml)",
    )
    p_lme.add_argument("--max-samples", type=int, default=None, help="Cap on samples")
    p_lme.add_argument(
        "--categories", nargs="+", default=None,
        help="Restrict to these question_type categories (e.g. temporal-reasoning)",
    )
    p_lme.add_argument("--budget", type=int, default=4000, help="Recall token budget")
    p_lme.add_argument(
        "--max-facts", type=int, default=40,
        help="Cap on facts injected into answerer prompt",
    )
    p_lme.add_argument("--verbose", "-v", action="store_true", help="Print progress")
    p_lme.add_argument(
        "--parallel", type=int, default=4,
        help="Max concurrent samples (default: 4). Each sample gets its own AgentMemory instance.",
    )
    p_lme.add_argument(
        "--verify", action="store_true",
        help="Run a second-pass verification that re-checks date arithmetic + abstentions.",
    )
    p_lme.add_argument(
        "--verify-model", default=None,
        help="OpenRouter model id for the verifier (defaults to --answer-model).",
    )
    p_lme.add_argument(
        "--output", "-o", default=None, help="Save full report JSON to file",
    )
    p_lme.add_argument("--env-file", default=None, help="Path to .env file for API keys")
    _add_backend_args(p_lme)

    # api (REST API server)
    p_api = subparsers.add_parser(
        "api",
        help="Start Starlette REST API server (uvicorn)",
    )
    p_api.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_api.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    p_api.add_argument(
        "--path",
        default=None,
        help="Memory store path (default: $ATTESTOR_DATA_DIR or ~/.attestor)",
    )

    # ui (read-only web viewer)
    p_ui = subparsers.add_parser(
        "ui",
        help="Start read-only web UI (Forensic Archive viewer)",
    )
    p_ui.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    p_ui.add_argument("--port", type=int, default=8080, help="Bind port (default: 8080)")
    p_ui.add_argument(
        "--path",
        default=None,
        help="Memory store path (default: $ATTESTOR_PATH or ~/.attestor)",
    )
    p_ui.add_argument("--open", action="store_true", help="Open browser on launch")

    # mcp (zero-config MCP server -- used by .mcp.json)
    p_mcp = subparsers.add_parser(
        "mcp",
        help="Start MCP server with default store path (zero-config)",
    )
    p_mcp.add_argument(
        "--path", default=None,
        help="Override store path (default: $ATTESTOR_PATH or ~/.attestor)",
    )

    # hook (delegates to hook handlers)
    p_hook = subparsers.add_parser("hook", help="Run a Claude Code lifecycle hook")
    hook_sub = p_hook.add_subparsers(dest="hook_name", help="Hook to run")
    hook_sub.add_parser("session-start", help="SessionStart hook")
    hook_sub.add_parser("post-tool-use", help="PostToolUse hook")
    hook_sub.add_parser("stop", help="Stop hook")

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
        "update": _cmd_update,
        "forget": _cmd_forget,
        "serve": _cmd_serve,
        "api": _cmd_api,
        "ui": _cmd_ui,
        "setup-claude-code": _cmd_setup_claude_code,
        "doctor": _cmd_doctor,
        "locomo": _cmd_locomo,
        "mab": _cmd_mab,
        "longmemeval": _cmd_longmemeval,
        "mcp": _cmd_mcp_serve,
        "hook": _cmd_hook,
    }
    handlers[args.command](args)


if __name__ == "__main__":
    main()
