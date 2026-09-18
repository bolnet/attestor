# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Shared retryable-vs-permanent classification for governance activities.

Reuses the derive lane's ``PERMANENT_ERROR_PATTERNS`` (auth, permission,
malformed input): those are config problems a retry cannot fix, so the
activity raises ``ApplicationError(non_retryable=True)``. Everything else
(store down, timeout, 5xx) is retried under the workflow's RetryPolicy.

The message is sanitised (``_sanitize``) and the raw exception is NOT
chained: Temporal persists both the message and the cause chain in
workflow history, outside every retention / forget control.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from temporalio.exceptions import ApplicationError

from attestor.durable.activities._sanitize import sanitize_error_message
from attestor.durable.activities.derive import describe_error, is_permanent_derive_error

if TYPE_CHECKING:
    import logging


def raise_for(
    exc: BaseException,
    *,
    subject: str,
    step: str,
    transient_type: str,
    permanent_type: str,
    logger: logging.Logger,
) -> None:
    """Translate ``exc`` into a typed ``ApplicationError`` (always raises)."""
    permanent = is_permanent_derive_error(exc)
    message = sanitize_error_message(describe_error(exc))
    logger.warning(
        "%s %s failed (%s): %s",
        step, subject, "permanent" if permanent else "retryable", message,
    )
    raise ApplicationError(
        message, type=permanent_type if permanent else transient_type, non_retryable=permanent,
    ) from None
