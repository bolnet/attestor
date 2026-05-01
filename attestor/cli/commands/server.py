"""Server-style CLI commands: ``serve``, ``api``, ``ui``, ``mcp``, ``hook``.

The ``mcp`` handler intentionally references ``attestor.cli.asyncio`` (rather
than its own ``import asyncio``) so that tests which do
``patch("attestor.cli.asyncio")`` continue to no-op the
``asyncio.run(run_server(...))`` call after the file split.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from attestor.core import AgentMemory


def _cmd_serve(args: argparse.Namespace) -> None:
    import asyncio
    try:
        from attestor.mcp.server import run_server
    except ImportError:
        print("MCP package not found. Run: poetry install")
        sys.exit(1)

    # Ensure the store exists
    AgentMemory(args.path).close()
    print(f"Starting MCP server for {args.path}...", file=sys.stderr)
    asyncio.run(run_server(args.path))


def _cmd_api(args: argparse.Namespace) -> None:
    try:
        import uvicorn
    except ImportError:
        print("uvicorn is required. Run: pip install 'attestor[lambda]' or pip install uvicorn")
        sys.exit(1)

    from attestor import _branding as brand

    if args.path:
        os.environ[brand.ENV_DATA_DIR] = args.path

    print(
        f"Starting attestor REST API on http://{args.host}:{args.port}",
        file=sys.stderr,
    )
    uvicorn.run(
        "attestor.api:app",
        host=args.host,
        port=args.port,
        log_level="info",
    )


def _cmd_mcp_serve(args: argparse.Namespace) -> None:
    """Start MCP server with default store path (zero-config)."""
    try:
        from attestor.mcp.server import run_server
    except ImportError:
        print("MCP package not found. Run: poetry install")
        sys.exit(1)

    from attestor._paths import resolve_store_path

    store_path = resolve_store_path(getattr(args, "path", None))

    # Ensure store directory and DB exist
    Path(store_path).mkdir(parents=True, exist_ok=True)
    try:
        AgentMemory(store_path).close()
    except Exception as init_err:
        if not os.environ.get("ATTESTOR_MCP_TOLERATE_INIT_FAILURE"):
            raise
        print(
            f"[attestor.mcp] Preflight AgentMemory init failed (tolerated): {init_err}",
            file=sys.stderr,
        )

    print(f"Starting MCP server for {store_path}...", file=sys.stderr)
    # Use the package-level asyncio so ``patch("attestor.cli.asyncio")`` in
    # tests continues to mock out ``asyncio.run`` after the file split.
    from attestor import cli as _cli_pkg
    _cli_pkg.asyncio.run(run_server(store_path))


def _cmd_ui(args: argparse.Namespace) -> None:
    """Launch read-only web UI."""
    from attestor import _branding as brand
    from attestor._paths import resolve_store_path

    store_path = resolve_store_path(args.path)
    os.environ[brand.ENV_STORE_PATH] = store_path
    Path(store_path).mkdir(parents=True, exist_ok=True)

    print(f"Attestor UI · {store_path}", file=sys.stderr)
    print(f"→ http://{args.host}:{args.port}/ui/memories", file=sys.stderr)

    if args.open:
        import webbrowser, threading
        url = f"http://{args.host}:{args.port}/ui/memories"
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    import uvicorn
    uvicorn.run(
        "attestor.api:app", host=args.host, port=args.port, log_level="warning"
    )


def _cmd_hook(args: argparse.Namespace) -> None:
    """Dispatch to the appropriate hook handler."""
    hook_name = getattr(args, "hook_name", None)
    if not hook_name:
        print("Usage: attestor hook {session-start|post-tool-use|stop}")
        return

    hook_handlers = {
        "session-start": "attestor.hooks.session_start",
        "post-tool-use": "attestor.hooks.post_tool_use",
        "stop": "attestor.hooks.stop",
    }

    module_name = hook_handlers.get(hook_name)
    if not module_name:
        print(f"Unknown hook: {hook_name}")
        return

    import importlib
    mod = importlib.import_module(module_name)
    mod.main()
