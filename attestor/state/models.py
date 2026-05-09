# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Models for the typed profile memory lane.

The state lane is the *non*-retrieval-based personalization surface.
Where memories are recalled by similarity, state values are looked up by
key. ``StateRecord`` is the immutable row representation: every set /
delete generates a new record and supersedes the previous active one
via ``t_expired``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datetime import datetime


class StateValidationError(ValueError):
    """Raised when a value fails its declared JSON-Schema or the schema
    name cannot be resolved.

    Subclasses ``ValueError`` so callers that already catch validation
    errors generically keep working. The first positional argument is a
    human-readable message that names the schema_id when available.
    """


@dataclass(frozen=True)
class StateRecord:
    """One row from the ``state`` table.

    Every set / delete writes a new record; the previous active record's
    ``t_expired`` is stamped with NOW() so ``history()`` and ``as_of()``
    have the full bi-temporal trail.

    Attributes:
        key: Dotted key, e.g. ``"preferences.theme"``.
        value: Arbitrary JSON-serializable value.
        user_id: Owner of the value. Required.
        project_id: Optional project anchor. ``None`` means "user-scoped".
        agent_id: ID of the agent that wrote this row, if known.
        scope: One of ``user`` / ``project`` / ``agent``. The default is
            inferred from which of project_id / agent_id is set.
        schema_id: Name of the JSON-Schema that validated this row, if any.
        set_at: When the value was set (event time).
        set_by_agent: ID of the agent that wrote the row (denormalized
            from ``agent_id`` for audit reads that don't need a join).
        t_created: Transaction time start (when the row landed in the DB).
        t_expired: Transaction time end. ``None`` while active; set on
            supersession or delete.
    """

    key: str
    value: Any
    user_id: str
    project_id: str | None = None
    agent_id: str | None = None
    scope: str = "user"
    schema_id: str | None = None
    set_at: datetime | None = None
    set_by_agent: str | None = None
    t_created: datetime | None = None
    t_expired: datetime | None = None
    signature: str | None = None
