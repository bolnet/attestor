# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Helpers shared by the sandboxed workflow modules (Temporal + models only)."""

from __future__ import annotations

from datetime import timedelta
from typing import TYPE_CHECKING

from temporalio.common import RetryPolicy
from temporalio.exceptions import ActivityError, ApplicationError

if TYPE_CHECKING:
    from attestor.durable.models import ActivityPolicy


def retry_policy(policy: ActivityPolicy) -> RetryPolicy:
    return RetryPolicy(
        initial_interval=timedelta(seconds=policy.initial_interval_seconds),
        backoff_coefficient=policy.backoff_coefficient,
        maximum_interval=timedelta(seconds=policy.maximum_interval_seconds),
        maximum_attempts=policy.maximum_attempts,
    )


def activity_error_message(exc: ActivityError) -> str:
    cause = exc.cause
    if isinstance(cause, ApplicationError):
        return cause.message
    return str(cause or exc)
