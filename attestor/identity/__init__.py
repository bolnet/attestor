"""Attestor v4 identity layer — User, Project, Session repositories.

These repos are pure data-access. They take a psycopg2 connection and
do CRUD against the v4 schema. No business logic, no LLM calls. Higher-
level orchestration (mode detection, defaults, autostart) lives in
``attestor.identity.defaults`` and ``attestor.core``.

All methods assume the connection is already RLS-scoped to the right
user (via ``set_config('attestor.current_user_id', ...)``) when reading
tenant-scoped data. The bootstrap user-creation path is the one
exception — it must run with admin/no-RLS context.
"""

from attestor.identity.projects import ProjectRepo
from attestor.identity.sessions import SessionRepo
from attestor.identity.users import UserRepo

__all__ = ["UserRepo", "ProjectRepo", "SessionRepo"]
