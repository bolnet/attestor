# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Audit-trail explorer for Attestor recall traces.

Reads the JSONL trace log written by ``attestor.trace`` (when
``ATTESTOR_TRACE=1`` and ``ATTESTOR_TRACE_FILE`` are set), groups events
by ``recall_id``, and surfaces summaries + details for the audit
dashboard mounted at ``/audit/*`` on the Starlette ASGI app.

The explorer is strictly read-only: it never writes to or rewrites the
trace log. The JSONL log remains the source of truth — the explorer is
a query-side index.
"""

from __future__ import annotations

from attestor.audit.explorer import (
    RecallDetail,
    RecallSummary,
    TraceExplorer,
    get_recall,
    list_recalls,
    memory_access_timeline,
    reset_cache,
    user_activity,
)

__all__ = [
    "RecallDetail",
    "RecallSummary",
    "TraceExplorer",
    "get_recall",
    "list_recalls",
    "memory_access_timeline",
    "reset_cache",
    "user_activity",
]
