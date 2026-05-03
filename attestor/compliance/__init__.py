"""Compliance primitives — declarative retention policies + GDPR forget.

Two surfaces ship in this module:

  * ``retention`` — declarative YAML/SQL-backed retention rules. Define
    ("delete episodic memories older than 90 days", "archive
    namespace=demo after 30 days") once; let
    :func:`attestor.compliance.retention.apply_retention` evaluate them
    on a schedule.
  * ``forget_user`` — physical deletion across the full backend stack
    (Postgres + Pinecone + Neo4j + state lane) so a GDPR right-to-be-
    forgotten request actually erases the user. Audit row is written
    BEFORE any backend touches state — the audit trail is the source of
    truth even if a backend partially fails.

Wired onto :class:`attestor.AgentMemory` as ``mem.retention.*`` and
``mem.forget_user(user_id)`` (see :mod:`attestor.core.agent_memory`).
"""

from __future__ import annotations

from attestor.compliance.retention import (
    ForgetUserResult,
    RetentionApplyResult,
    RetentionFacade,
    RetentionPolicy,
    add_retention_policy,
    apply_retention,
    forget_user,
    list_retention_policies,
    remove_retention_policy,
)

__all__ = [
    "ForgetUserResult",
    "RetentionApplyResult",
    "RetentionFacade",
    "RetentionPolicy",
    "add_retention_policy",
    "apply_retention",
    "forget_user",
    "list_retention_policies",
    "remove_retention_policy",
]
