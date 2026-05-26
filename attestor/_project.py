# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Resolve a Claude Code working directory to a stable project identity.

A *project* is the unit of memory isolation: the git repository root if the
working directory is inside one, otherwise the working directory itself. Two
unrelated directories always resolve to different identities, so memory never
bleeds across projects.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

# external_id of the RLS tenant that owns a project's memory. Distinct from the
# SOLO singleton ("local") so per-project tenants never collide with it.
_PROJECT_EXTERNAL_PREFIX = "cc-project:"

# Defense-in-depth namespace stamped on Pinecone vectors and Neo4j nodes (and
# the Postgres doc-store ``_namespace`` metadata shim).
_PROJECT_NAMESPACE_PREFIX = "project:"


def _digest(root: str) -> str:
    """Stable 64-char hex digest of a project root.

    Hashing keeps the derived identifiers fixed-length and charset-safe:
    ``users.external_id`` is ``VARCHAR(256)`` (a deep absolute path could
    overflow it, silently dropping the RLS tenant), and a Pinecone namespace
    must avoid path separators. The human-readable root is preserved in the
    user's metadata/display_name by ``resolve_tenant`` for debugging.
    """
    return hashlib.sha256(root.encode("utf-8")).hexdigest()


def resolve_project_root(cwd: str | Path) -> str:
    """Return the project root for ``cwd``.

    Git repository root if ``cwd`` is inside one (so subdirectories of a repo
    share memory); otherwise the resolved absolute ``cwd``. If ``cwd`` is a
    file, its parent directory is used.
    """
    start = Path(cwd).expanduser().resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return str(start)


def project_external_id(root: str) -> str:
    """RLS tenant external_id for a project root (fixed-length, charset-safe)."""
    return f"{_PROJECT_EXTERNAL_PREFIX}{_digest(root)}"


def project_namespace(root: str) -> str:
    """Vector/graph namespace for a project root (fixed-length, charset-safe)."""
    return f"{_PROJECT_NAMESPACE_PREFIX}{_digest(root)}"
