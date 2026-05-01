"""Quota / budget enforcement mixin for AgentMemory (split from core.py).

Exposes the public quota surface (``set_quota`` / ``get_quota``). The
enforcement check-points themselves (``_quotas.check_memory_quota`` etc.)
are inlined inside the call paths in ``agent_memory.py`` because they
participate in transactional ordering with insert / resolve.

Mixin assumes the composing class wires up ``self._quotas`` (Optional
``QuotaRepo``) in ``__init__``.
"""

from __future__ import annotations

from typing import Any


class _QuotaMixin:
    """Mixin holding the public quota surface for AgentMemory."""

    # -- v4 quota management (Phase 8.3) --

    def set_quota(
        self,
        user_id: str,
        *,
        max_memories: int | None = None,
        max_sessions: int | None = None,
        max_projects: int | None = None,
        max_writes_per_day: int | None = None,
    ) -> Any:
        """Set per-user quotas. NULL/omitted limits leave that limit
        unchanged (no implicit unlimited reset)."""
        self._require_v4()
        if self._quotas is None:
            raise RuntimeError("quota repo unavailable on this backend")
        return self._quotas.set_limits(
            user_id,
            max_memories=max_memories,
            max_sessions=max_sessions,
            max_projects=max_projects,
            max_writes_per_day=max_writes_per_day,
        )

    def get_quota(self, user_id: str) -> Any | None:
        """Return the current UserQuota row (counters + limits)."""
        self._require_v4()
        if self._quotas is None:
            return None
        return self._quotas.get(user_id)
