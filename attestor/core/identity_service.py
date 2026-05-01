"""Identity-resolution mixin for AgentMemory (split from core.py).

Provides:
    * v4 require / connection helpers (``_require_v4``, ``_pg_conn``)
    * lazy repo accessors (``_user_repo``, ``_project_repo``, ``_session_repo``)
    * public identity surface (ensure_user, get_user, find_user_by_external_id,
      ensure_inbox, create_project, list_projects, start_session,
      get_or_create_daily_session, end_session, list_sessions)
    * the central resolver ``_resolve``
    * mode / default_user properties

The mixin assumes the composing class wires up:
    - ``self._store`` (DocumentStore with optional ``_v4`` and ``_conn``)
    - ``self._mode`` (AttestorMode)
    - ``self._default_user`` (Optional[User])
    - ``self._quotas`` (Optional[QuotaRepo])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from attestor.mode import AttestorMode
from attestor.models import Project, Session, User


@dataclass(frozen=True)
class ResolvedContext:
    """Result of AgentMemory._resolve() — the identity tuple for a call.

    Public methods take optional user_id/project_id/session_id; _resolve()
    fills any that are missing using the mode's defaults (SOLO singleton
    user, Inbox project, daily session) and returns the fully-resolved
    triple plus a derived AgentContext for downstream provenance.

    session can be None when autostart=False (e.g. for read-only recall
    that doesn't need a session anchor)."""
    user: User
    project: Project
    session: Session | None


class _IdentityMixin:
    """Mixin holding identity provisioning + resolution for AgentMemory.

    Methods read state set up in ``AgentMemory.__init__``: ``self._store``,
    ``self._mode``, ``self._default_user``, ``self._quotas``.
    """

    # -- v4 identity (Phase 1 chunk 4) --
    #
    # The repos are constructed lazily and share the document store's
    # psycopg2 connection. That keeps the RLS variable consistent: any
    # SELECT / INSERT issued through a repo runs in the same session as
    # the memory writes that follow.
    #
    # Provisioning (create user / ensure inbox) requires admin / RLS-
    # bypassing context — the policies do not match yet because the user
    # row hasn't been seen by the policy yet. Callers running in v4 mode
    # are expected to use a Postgres role that owns the tables (RLS skipped
    # for table owners) for the bootstrap path. Once a user exists we set
    # the RLS variable and the rest of the surface is tenant-scoped.

    def _require_v4(self) -> None:
        if not getattr(self._store, "_v4", False):
            raise RuntimeError(
                "Identity APIs require v4 mode. Set ATTESTOR_V4=1 or pass "
                "v4=True in the postgres backend config."
            )

    def _pg_conn(self) -> Any:
        """Return the underlying psycopg2 connection on the document store.

        Raises if the store doesn't expose ``_conn`` — only Postgres-backed
        v4 deployments are supported in this chunk; ArangoDB / Cosmos / etc.
        get identity wiring in Phase 1.5."""
        conn = getattr(self._store, "_conn", None)
        if conn is None:
            raise RuntimeError(
                "Identity APIs currently require the Postgres backend "
                f"(no _conn on store={type(self._store).__name__})"
            )
        return conn

    def _user_repo(self) -> Any:
        from attestor.identity.users import UserRepo
        if not hasattr(self, "_users"):
            self._users = UserRepo(self._pg_conn())
        return self._users

    def _project_repo(self) -> Any:
        from attestor.identity.projects import ProjectRepo
        if not hasattr(self, "_projects"):
            self._projects = ProjectRepo(self._pg_conn())
        return self._projects

    def _session_repo(self) -> Any:
        from attestor.identity.sessions import SessionRepo
        if not hasattr(self, "_sessions"):
            self._sessions = SessionRepo(self._pg_conn())
        return self._sessions

    def ensure_user(
        self,
        external_id: str,
        email: str | None = None,
        display_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> User:
        """Idempotent first-login provisioning. Returns the user; safe to
        call on every request. Inbox project is created on first call so
        the caller never has to think about it."""
        self._require_v4()
        user = self._user_repo().create_or_get(
            external_id=external_id,
            email=email,
            display_name=display_name,
            metadata=metadata,
        )
        # Auto-provision Inbox so chat can start a session immediately.
        self._project_repo().ensure_inbox(user.id)
        return user

    def get_user(self, user_id: str) -> User | None:
        self._require_v4()
        return self._user_repo().get(user_id)

    def find_user_by_external_id(self, external_id: str) -> User | None:
        self._require_v4()
        return self._user_repo().find_by_external_id(external_id)

    def ensure_inbox(self, user_id: str) -> Project:
        """Idempotent. The Inbox is the default project for sessions that
        haven't been assigned to one. Lives forever; cannot be deleted."""
        self._require_v4()
        return self._project_repo().ensure_inbox(user_id)

    def create_project(
        self,
        user_id: str,
        name: str,
        description: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Project:
        self._require_v4()
        if self._quotas is not None:
            self._quotas.check_project_quota(user_id)
        return self._project_repo().create(
            user_id=user_id, name=name,
            description=description, metadata=metadata,
        )

    def list_projects(
        self, user_id: str, include_inbox: bool = False, limit: int = 100,
    ) -> list[Project]:
        self._require_v4()
        return self._project_repo().list_for_user(
            user_id, include_inbox=include_inbox, limit=limit,
        )

    def start_session(
        self,
        user_id: str,
        project_id: str | None = None,
        title: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Session:
        """Open a new session. If ``project_id`` is None, drops it into the
        user's Inbox so the caller never has to provision one upfront."""
        self._require_v4()
        if self._quotas is not None:
            self._quotas.check_session_quota(user_id)
        if project_id is None:
            project_id = self.ensure_inbox(user_id).id
        return self._session_repo().create(
            user_id=user_id, project_id=project_id,
            title=title, metadata=metadata,
        )

    def get_or_create_daily_session(
        self, user_id: str, day: str, project_id: str | None = None,
    ) -> Session:
        """SOLO-mode helper — one session per (user, ISO-date)."""
        self._require_v4()
        if project_id is None:
            project_id = self.ensure_inbox(user_id).id
        return self._session_repo().get_or_create_daily(
            user_id=user_id, project_id=project_id, day=day,
        )

    def end_session(self, session_id: str) -> Session | None:
        self._require_v4()
        return self._session_repo().end(session_id)

    def list_sessions(
        self,
        user_id: str,
        project_id: str | None = None,
        status: str = "active",
        limit: int = 20,
    ) -> list[Session]:
        self._require_v4()
        return self._session_repo().list_for_user(
            user_id=user_id, project_id=project_id,
            status=status, limit=limit,
        )

    # -- v4 resolution chain (Phase 2 chunk 3, defaults.md §5) --

    def _resolve(
        self,
        user_id: str | None = None,
        project_id: str | None = None,
        session_id: str | None = None,
        autostart: bool = True,
    ) -> ResolvedContext:
        """Single entry point for identity resolution. Public methods call
        this first to get the (user, project, session) triple they should
        operate against.

        Resolution rules (defaults.md §5):
          User:    explicit user_id wins. Else SOLO singleton. Else raise.
          Project: explicit project_id wins. Else session.project_id. Else Inbox.
          Session: explicit session_id wins. Else autostart (daily for SOLO,
                   fresh for HOSTED). If autostart=False → None.

        v3 / non-Postgres backends: this method is a no-op and is not
        called. Public methods only invoke it when v4 is on.
        """
        self._require_v4()
        from attestor.identity.defaults import (
            ensure_solo_user,
            resolve_project,
            resolve_session,
        )

        # ── User ────────────────────────────────────────────────────────
        if user_id is None:
            if self._mode is AttestorMode.SOLO:
                # Lazy-recover: if SOLO bootstrap was skipped at __init__,
                # try once more here. Keeps the zero-config path working
                # even if the schema wasn't ready at construction time.
                if self._default_user is None:
                    self._default_user = ensure_solo_user(self._user_repo())
                user = self._default_user
            else:
                raise PermissionError(
                    f"user_id is required in {self._mode.value} mode"
                )
        else:
            # Pre-set RLS so the users-table lookup itself is admitted —
            # the policy on `users` is `id = current_setting(...)`. Without
            # this, even a valid user_id would resolve to "not found".
            if hasattr(self._store, "_set_rls_user"):
                self._store._set_rls_user(user_id)
            user = self._user_repo().get(user_id)
            if user is None or user.status != "active":
                raise LookupError(f"user {user_id} not found")

        # Set RLS to the resolved user (no-op when we already set it above).
        if hasattr(self._store, "_set_rls_user"):
            self._store._set_rls_user(user.id)

        # ── Project ────────────────────────────────────────────────────
        project = resolve_project(
            user_id=user.id,
            project_id=project_id,
            session_id=session_id,
            project_repo=self._project_repo(),
            session_repo=self._session_repo(),
        )

        # ── Session ────────────────────────────────────────────────────
        session = resolve_session(
            user_id=user.id,
            project_id=project.id,
            session_id=session_id,
            session_repo=self._session_repo(),
            autostart=autostart,
            mode_is_solo=(self._mode is AttestorMode.SOLO),
        )

        return ResolvedContext(user=user, project=project, session=session)

    @property
    def mode(self) -> AttestorMode:
        """The detected operating mode. Settable via ``config["mode"]`` or
        the ``ATTESTOR_MODE`` env var."""
        return self._mode

    @property
    def default_user(self) -> User | None:
        """The SOLO singleton user, if SOLO mode is active. None otherwise."""
        return self._default_user
