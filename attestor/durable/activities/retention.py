# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``RetentionSweep`` activities — ``list_due_policies`` + ``apply_policy``.

Every enabled ``retention_policies`` row is due on every sweep (a policy
has no schedule of its own; the Schedule's cron IS the cadence).
``apply_policy`` wraps ``attestor.compliance.retention.apply_retention``
restricted to one policy id, so each policy is its own retry unit and a
failing policy never blocks the rest of the sweep.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from temporalio import activity

from attestor.compliance import retention
from attestor.durable.activities._errors import raise_for
from attestor.durable.models import (
    APPLY_POLICY_ACTIVITY,
    LIST_DUE_POLICIES_ACTIVITY,
    ApplyPolicyRequest,
    DuePolicies,
    DuePolicy,
    PolicyApplyOutcome,
    RetentionSweepRequest,
)

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger("attestor.durable.activities.retention")

ERROR_TYPE_TRANSIENT = "RetentionFailed"
ERROR_TYPE_PERMANENT = "RetentionPermanent"
STEP_LIST = "list_due_policies"
STEP_APPLY = "apply_policy"


def _outcome_for(policy_id: str, result: Any) -> PolicyApplyOutcome:
    row = result.by_policy.get(policy_id)
    if row is None:  # disabled / removed between list and apply → nothing to do
        return PolicyApplyOutcome(policy_id=policy_id, ok=True)
    return PolicyApplyOutcome(
        policy_id=policy_id, ok=True, action=str(row.get("action", "")),
        matched=int(row.get("matched", 0)), applied=int(row.get("applied", 0)),
        vector_purged=int(row.get("vector_purged", 0)),
    )


class RetentionActivities:
    """Activity set bound to a lazily-built ``AgentMemory``."""

    def __init__(self, memory_provider: Callable[[], Any]) -> None:
        self._provider = memory_provider

    def _get(self) -> Any:
        # Always ask the provider: it caches the healthy instance and
        # rebuilds one whose backends failed to initialise (worker.py).
        return self._provider()

    @activity.defn(name=LIST_DUE_POLICIES_ACTIVITY)
    def list_due_policies(self, req: RetentionSweepRequest) -> DuePolicies:
        try:
            policies = retention.list_retention_policies(self._get())
        except Exception as exc:
            raise_for(
                exc, subject="retention_policies", step=STEP_LIST,
                transient_type=ERROR_TYPE_TRANSIENT, permanent_type=ERROR_TYPE_PERMANENT,
                logger=logger,
            )
        return DuePolicies(policies=tuple(
            DuePolicy(policy_id=p.id, name=p.name, action=p.action) for p in policies
        ))

    @activity.defn(name=APPLY_POLICY_ACTIVITY)
    def apply_policy(self, req: ApplyPolicyRequest) -> PolicyApplyOutcome:
        try:
            result = retention.apply_retention(
                self._get(), dry_run=req.dry_run, initiated_by=req.initiated_by,
                policy_ids=(req.policy_id,),
            )
        except Exception as exc:
            raise_for(
                exc, subject=f"policy={req.policy_id}", step=STEP_APPLY,
                transient_type=ERROR_TYPE_TRANSIENT, permanent_type=ERROR_TYPE_PERMANENT,
                logger=logger,
            )
        return _outcome_for(req.policy_id, result)
