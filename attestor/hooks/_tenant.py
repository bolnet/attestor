# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Map a Claude Code working directory to the tenant that owns its memory.

This is the single chokepoint that enforces per-project isolation in the hook
path. Every hook recall/add routes its ``cwd`` through here.

On a v4 Postgres store this returns a real ``user_id`` so the Row-Level
Security policy (keyed to ``attestor.current_user_id``) filters to exactly one
project. On a v3 / non-Postgres / degraded store ``ensure_user`` is
unavailable; we fall back to ``user_id=None`` and rely on the ``namespace`` to
isolate the Pinecone vector lane, the Neo4j graph lane, and the Postgres
doc-store ``_namespace`` metadata shim. Either way, memory never bleeds across
projects.
"""

from __future__ import annotations

from typing import Any

from attestor._project import (
    project_external_id,
    project_namespace,
    resolve_project_root,
)


def resolve_tenant(mem: Any, cwd: str) -> tuple[str | None, str]:
    """Return ``(user_id, namespace)`` for the project containing ``cwd``.

    Idempotently ensures an RLS user keyed to the project root when the store
    supports it. ``ensure_user`` is a cheap lookup after first creation. Any
    failure (v3 store, Postgres down) degrades to ``user_id=None`` with the
    namespace still set — never raises, so hooks never crash the host.
    """
    root = resolve_project_root(cwd)
    namespace = project_namespace(root)

    ensure_user = getattr(mem, "ensure_user", None)
    if ensure_user is None:
        return None, namespace
    try:
        # external_id is a hash; carry the human path in display_name/metadata
        # so the tenant is debuggable in the users table.
        user = ensure_user(
            external_id=project_external_id(root),
            display_name=root,
            metadata={"project_root": root, "source": "claude-code"},
        )
        return user.id, namespace
    except Exception:  # noqa: BLE001 -- degrade to namespace-only isolation
        return None, namespace
