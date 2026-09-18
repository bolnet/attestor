# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Redact activity failure messages before they enter workflow history.

Temporal persists an ``ApplicationError``'s message (and its cause chain)
in the workflow's event history for the namespace retention window — a
second store that neither the retention sweep nor a forget saga governs.
Driver exceptions (psycopg2, Pinecone, Neo4j) can echo DSNs, passwords
and API keys, so governance activities pass every message through
:func:`sanitize_error_message` and raise ``from None``.
"""

from __future__ import annotations

import re

MAX_ERROR_MESSAGE_CHARS = 512
REDACTED = "[redacted]"
_ELLIPSIS = "…"

# ``scheme://user:secret@host`` — anything with credentials in the authority.
_CREDENTIALED_URL = re.compile(r"\b[a-z][a-z0-9+.\-]*://[^\s/@]+@\S+", re.IGNORECASE)
# ``password=…`` / ``api_key: …`` / ``Authorization: Bearer …`` (value redacted, key kept).
_SECRET_KV = re.compile(
    r"(?P<key>\b(?:password|passwd|pwd|secret|token|api[_\- ]?key|access[_\- ]?key|"
    r"authorization|bearer)\b)\s*[:=]\s*(?:bearer\s+)?\S+",
    re.IGNORECASE,
)
# Bare key shapes: OpenAI ``sk-…``, Pinecone ``pcsk_…``, Anthropic ``sk-ant-…``.
_BARE_KEY = re.compile(r"\b(?:sk|pcsk|pcsk_[a-z]+)[-_][A-Za-z0-9_\-]{8,}")
_WHITESPACE = re.compile(r"\s+")


def sanitize_error_message(text: str, *, max_chars: int = MAX_ERROR_MESSAGE_CHARS) -> str:
    """Redact credentials, collapse whitespace and cap ``text`` for durable storage."""
    cleaned = _CREDENTIALED_URL.sub(REDACTED, text)
    cleaned = _SECRET_KV.sub(lambda m: f"{m.group('key')}={REDACTED}", cleaned)
    cleaned = _BARE_KEY.sub(REDACTED, cleaned)
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    if len(cleaned) > max_chars:
        return cleaned[:max_chars] + _ELLIPSIS
    return cleaned
