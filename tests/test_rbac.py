"""RBAC enforcement at the AgentContext layer.

Until 2026-04-28 ``AgentRole`` was advisory metadata -- the role enum
was set on the context, recorded in memory metadata, and otherwise
ignored. This module pins the actual enforcement matrix introduced in
``attestor/context.py`` (ROLE_PERMISSIONS):

    ORCHESTRATOR        : READ + WRITE + FORGET
    PLANNER / EXECUTOR /
        RESEARCHER      : READ + WRITE
    REVIEWER / MONITOR  : READ only

The tests use a small in-memory fake of the Attestor memory backend
because we only need to verify the permission gate fires *before* the
backend is touched. This keeps the suite hermetic (no Postgres / Neo4j
required) and catches regressions where a future refactor removes the
``_require_permission`` call.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

from attestor.context import (
    ROLE_PERMISSIONS,
    AgentContext,
    AgentRole,
    RolePermission,
)
from attestor.models import Memory


@dataclass
class _FakeMem:
    """Minimal stand-in for AgentMemory used to assert RBAC fires first."""

    added: list[str] = field(default_factory=list)
    forgotten: list[str] = field(default_factory=list)

    def add(self, content: str, **kwargs) -> Memory:
        self.added.append(content)
        return Memory(
            id=f"mem-{len(self.added)}",
            content=content,
            namespace=kwargs.get("namespace", "default"),
        )

    def forget(self, memory_id: str) -> bool:
        self.forgotten.append(memory_id)
        return True

    def recall(self, query: str, **kwargs):
        return []

    def health(self):  # pragma: no cover - unused in these tests
        return {"ok": True}

    def close(self):  # pragma: no cover
        pass


def _ctx(role: AgentRole, *, read_only: bool = False) -> tuple[AgentContext, _FakeMem]:
    fake = _FakeMem()
    ctx = AgentContext(
        agent_id=f"agent-{role.value}",
        role=role,
        memory=fake,
        read_only=read_only,
    )
    return ctx, fake


# ---------------------------------------------------------------------------
# Matrix completeness
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_role_permissions_matrix_covers_every_role():
    """A new AgentRole without a matrix entry would silently default to
    `frozenset()` (deny-all). That is safe but surprising -- this guards
    against forgetting to update the table when adding a role."""
    for role in AgentRole:
        assert role in ROLE_PERMISSIONS, f"Role {role} missing from ROLE_PERMISSIONS"
        assert (
            RolePermission.READ in ROLE_PERMISSIONS[role]
        ), f"Role {role} cannot READ -- recall path would be blinded"


# ---------------------------------------------------------------------------
# Write enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "role",
    [AgentRole.ORCHESTRATOR, AgentRole.PLANNER, AgentRole.EXECUTOR, AgentRole.RESEARCHER],
)
def test_write_allowed_roles_can_add_memory(role: AgentRole):
    ctx, fake = _ctx(role)
    ctx.add_memory("hello", category="note")
    assert fake.added == ["hello"]


@pytest.mark.unit
@pytest.mark.parametrize("role", [AgentRole.REVIEWER, AgentRole.MONITOR])
def test_write_denied_roles_cannot_add_memory(role: AgentRole):
    ctx, fake = _ctx(role)
    with pytest.raises(PermissionError, match=role.value):
        ctx.add_memory("should be blocked", category="note")
    assert fake.added == [], "backend should not be touched after a deny"


@pytest.mark.unit
def test_default_executor_add_still_works():
    """Backwards-compat smoke: the default role is EXECUTOR, and the
    pre-existing add() call shape must continue to work without changes."""
    fake = _FakeMem()
    ctx = AgentContext(agent_id="a", memory=fake)
    ctx.add_memory("default-role write")
    assert fake.added == ["default-role write"]


# ---------------------------------------------------------------------------
# Forget enforcement
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_forget_allowed_for_orchestrator():
    ctx, fake = _ctx(AgentRole.ORCHESTRATOR)
    assert ctx.forget("mem-x") is True
    assert fake.forgotten == ["mem-x"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "role",
    [
        AgentRole.PLANNER,
        AgentRole.EXECUTOR,
        AgentRole.RESEARCHER,
        AgentRole.REVIEWER,
        AgentRole.MONITOR,
    ],
)
def test_forget_denied_for_non_orchestrators(role: AgentRole):
    ctx, fake = _ctx(role)
    with pytest.raises(PermissionError, match="forget"):
        ctx.forget("mem-x")
    assert fake.forgotten == []


# ---------------------------------------------------------------------------
# read_only flag is independent of role and always wins
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_read_only_flag_strips_write_even_for_orchestrator():
    """An ORCHESTRATOR with read_only=True still cannot write -- the
    flag is a hard kill switch, not a role override."""
    ctx, fake = _ctx(AgentRole.ORCHESTRATOR, read_only=True)
    with pytest.raises(PermissionError, match="read-only"):
        ctx.add_memory("blocked")
    assert fake.added == []


@pytest.mark.unit
def test_read_only_flag_strips_forget_even_for_orchestrator():
    ctx, fake = _ctx(AgentRole.ORCHESTRATOR, read_only=True)
    with pytest.raises(PermissionError, match="read-only"):
        ctx.forget("mem-x")
    assert fake.forgotten == []


@pytest.mark.unit
def test_read_only_flag_does_not_affect_recall():
    """READ stays available regardless of read_only -- otherwise an
    auditor sub-context could not read what it is auditing."""
    ctx, _ = _ctx(AgentRole.MONITOR, read_only=True)
    # Should not raise. The fake returns [] but we are testing the gate,
    # not the retrieval pipeline.
    assert ctx.recall("anything") == []
