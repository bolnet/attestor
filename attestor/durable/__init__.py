# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Durable governance jobs on Temporal (durable execution).

Named ``durable`` — ``attestor/temporal/`` is temporal *reasoning*
(supersession, ``as_of`` replay) and unrelated.

Hard rules (docs/plans/temporal-integration.md §2):
  * Recall never touches Temporal — ``attestor.retrieval`` and
    ``attestor.hooks`` never import this package (enforced by
    ``tests/test_durable_isolation.py``).
  * Postgres stays the source of truth; Temporal holds job state only.
  * Opt-in: ``durable.enabled: false`` by default; every entry point
    raises ``DurableDisabledError`` rather than silently doing nothing.

Importing this package does NOT import ``temporalio``; the SDK is pulled
in by ``client`` / ``worker`` / ``status`` and the ``activities`` /
``workflows`` subpackages.
"""

from attestor.durable.config import (
    DurableDisabledError,
    DurableError,
    DurableUnavailableError,
    apply_overrides,
    require_enabled,
    require_temporalio,
    resolve_durable,
)
from attestor.durable.models import (
    CONSOLIDATE_EPISODE_ACTIVITY,
    CONSOLIDATE_WORKFLOW,
    DERIVE_WORKFLOW,
    EMBED_AND_UPSERT_ACTIVITY,
    FORGET_WORKFLOW,
    GRAPH_EXTRACT_ACTIVITY,
    LIST_MEMORY_IDS_ACTIVITY,
    REBUILD_WORKFLOW,
    RETENTION_WORKFLOW,
    SESSION_WORKFLOW,
    ActivityPolicy,
    ConsolidateRequest,
    ConsolidationOutcome,
    DeriveOutcome,
    DeriveRef,
    DeriveRequest,
    DurableDispatch,
    EpisodeRef,
    ForgetRequest,
    ForgetUserOutcome,
    RebuildOutcome,
    RebuildRequest,
    RetentionSweepOutcome,
    RetentionSweepRequest,
    SessionSweepOutcome,
    SessionSweepRequest,
    consolidate_workflow_id,
    derive_workflow_id,
    forget_workflow_id,
    rebuild_workflow_id,
    retention_sweep_workflow_id,
    session_sweep_workflow_id,
)

__all__ = [
    "CONSOLIDATE_EPISODE_ACTIVITY",
    "CONSOLIDATE_WORKFLOW",
    "DERIVE_WORKFLOW",
    "EMBED_AND_UPSERT_ACTIVITY",
    "FORGET_WORKFLOW",
    "GRAPH_EXTRACT_ACTIVITY",
    "LIST_MEMORY_IDS_ACTIVITY",
    "REBUILD_WORKFLOW",
    "RETENTION_WORKFLOW",
    "SESSION_WORKFLOW",
    "ActivityPolicy",
    "ConsolidateRequest",
    "ConsolidationOutcome",
    "DeriveOutcome",
    "DeriveRef",
    "DeriveRequest",
    "DurableDisabledError",
    "DurableDispatch",
    "DurableError",
    "DurableUnavailableError",
    "EpisodeRef",
    "ForgetRequest",
    "ForgetUserOutcome",
    "RebuildOutcome",
    "RebuildRequest",
    "RetentionSweepOutcome",
    "RetentionSweepRequest",
    "SessionSweepOutcome",
    "SessionSweepRequest",
    "apply_overrides",
    "consolidate_workflow_id",
    "derive_workflow_id",
    "forget_workflow_id",
    "rebuild_workflow_id",
    "require_enabled",
    "require_temporalio",
    "resolve_durable",
    "retention_sweep_workflow_id",
    "session_sweep_workflow_id",
]
