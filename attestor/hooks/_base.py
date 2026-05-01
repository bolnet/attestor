"""Common entry-point plumbing for stdin/stdout hook scripts.

Hooks are invoked by Claude Code with a JSON payload on stdin. They must:
- Return a JSON response on stdout
- Exit 0 in all cases (a crashing hook breaks the host session)
- On error, emit a structured envelope on stderr so operators can debug

This module factors the boilerplate that is identical across every hook
(`session_start`, `post_tool_use`, `stop`) so each hook script only owns
its handler logic.
"""

from __future__ import annotations

import json
import sys
import traceback
from collections.abc import Callable
from typing import Any

HookHandler = Callable[[dict[str, Any]], dict[str, Any]]
"""A hook handler takes the parsed JSON payload and returns the JSON response."""


def run_hook(
    name: str,
    handler: HookHandler,
    *,
    empty_response: dict[str, Any] | None = None,
) -> None:
    """Run a hook end-to-end: read stdin -> handler -> write stdout.

    Defensive: any exception (including malformed stdin) is logged to stderr
    as a structured JSON envelope and ``empty_response`` is emitted on stdout.
    Always exits 0 so a misbehaving hook cannot take down the host session.

    Args:
        name: Identifier embedded in stderr envelopes (e.g. ``"session_start"``).
        handler: Pure function taking the parsed payload and returning the
            response dict. Handler-internal failures may be caught by the
            handler itself; anything that escapes is captured here.
        empty_response: Returned on stdout when an exception escapes the
            handler. Must match the hook's documented response shape (e.g.
            ``{"additionalContext": ""}`` for SessionStart, ``{}`` otherwise).
    """
    safe_empty: dict[str, Any] = empty_response if empty_response is not None else {}
    try:
        payload: dict[str, Any] = json.loads(sys.stdin.read())
        result = handler(payload)
    except Exception as exc:  # noqa: BLE001 -- must not crash the host
        _emit_error(name, exc)
        result = safe_empty
    sys.stdout.write(json.dumps(result))
    sys.stdout.flush()
    # Implicit exit 0. Never call sys.exit with non-zero from a hook.


def _emit_error(name: str, exc: BaseException) -> None:
    """Write a structured JSON error envelope to stderr.

    Hook stdout is parsed by Claude Code as the response payload, so
    debugging info MUST go to stderr. Exit code stays 0 -- a crashing
    hook MUST NOT take down the host session.
    """
    envelope = {
        "hook": name,
        "error": str(exc),
        "type": type(exc).__name__,
        "trace": "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__, limit=3)
        ),
    }
    try:
        sys.stderr.write(json.dumps(envelope) + "\n")
        sys.stderr.flush()
    except Exception:  # noqa: BLE001
        # stderr write itself failed (closed pipe, etc.) -- there is
        # nowhere left to log; swallow rather than crash the host.
        pass
