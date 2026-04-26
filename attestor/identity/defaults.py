"""Default identity provisioning for SOLO mode (defaults.md §2–§4).

Three primitives:

  ensure_solo_user(user_repo)
      → idempotent singleton user with external_id="local"

  resolve_project(user_id, project_id, session_id, project_repo, session_repo)
      → explicit project_id wins; else session.project_id; else Inbox

  get_or_create_daily_session(user_id, project_id, session_repo, day=None)
      → one session per (user, ISO-date) for SOLO/CLI/MCP usage

These functions take repos as arguments rather than holding their own
DB handles. Keeps them composable, easy to test, and avoids duplicating
the connection-management logic that AgentMemory already does.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Optional

from attestor.identity.projects import ProjectRepo
from attestor.identity.sessions import SessionRepo
from attestor.identity.users import UserRepo
from attestor.mode import SOLO_USER_DISPLAY_NAME, SOLO_USER_EXTERNAL_ID
from attestor.models import Project, Session, User


def ensure_solo_user(user_repo: UserRepo) -> User:
    """Idempotent: return the SOLO singleton user, creating on first call.

    Uses ``create_or_get`` so concurrent first-time boots don't race on
    the unique external_id constraint.
    """
    return user_repo.create_or_get(
        external_id=SOLO_USER_EXTERNAL_ID,
        display_name=SOLO_USER_DISPLAY_NAME,
        metadata={"mode": "solo", "created_by": "auto"},
    )


def resolve_project(
    user_id: str,
    project_id: Optional[str],
    session_id: Optional[str],
    project_repo: ProjectRepo,
    session_repo: SessionRepo,
) -> Project:
    """Implement defaults.md §3.2 resolution chain:

      1. explicit project_id   → use it (after authz check)
      2. session_id given      → use that session's project_id
      3. neither given         → user's Inbox

    Authz: if the explicit project_id or session_id doesn't belong to
    user_id, raises ``LookupError`` (404 not 403, per defaults.md — don't
    leak existence of other users' rows).
    """
    if project_id:
        proj = project_repo.get(project_id)
        if proj is None or proj.user_id != user_id:
            raise LookupError(f"project {project_id} not found for this user")
        return proj

    if session_id:
        sess = session_repo.get(session_id)
        if sess is None or sess.user_id != user_id:
            raise LookupError(f"session {session_id} not found for this user")
        if sess.project_id:
            proj = project_repo.get(sess.project_id)
            if proj is not None:
                return proj

    # Fall through to Inbox (idempotent)
    return project_repo.ensure_inbox(user_id)


def get_or_create_daily_session(
    user_id: str,
    project_id: str,
    session_repo: SessionRepo,
    day: Optional[str] = None,
) -> Session:
    """SOLO autostart: one session per (user, ISO-date).

    Naturally rotates at midnight. Calls before/after midnight land in
    different sessions; the previous day's session stays around for
    history but isn't extended.

    Args:
        day: ISO date string (e.g. ``"2026-04-26"``). Defaults to today
            in the local timezone.
    """
    iso_day = day or date.today().isoformat()
    return session_repo.get_or_create_daily(
        user_id=user_id, project_id=project_id, day=iso_day,
    )


def resolve_session(
    user_id: str,
    project_id: str,
    session_id: Optional[str],
    session_repo: SessionRepo,
    *,
    autostart: bool,
    mode_is_solo: bool,
) -> Optional[Session]:
    """Resolve a session for the call. Mirrors defaults.md §5 step 3.

    If session_id is explicit, look it up and authz-check.
    Else if autostart=True:
        SOLO mode → daily session
        Other modes → fresh session bound to project_id
    Else (autostart=False, e.g. for recall) → None.
    """
    if session_id is not None:
        sess = session_repo.get(session_id)
        if sess is None or sess.user_id != user_id:
            raise LookupError(f"session {session_id} not found for this user")
        return sess

    if not autostart:
        return None

    if mode_is_solo:
        return get_or_create_daily_session(user_id, project_id, session_repo)
    return session_repo.create(
        user_id=user_id, project_id=project_id,
        metadata={"created_by": "autostart"},
    )
