# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Ingest-time pipelines: pre-write transforms applied before persistence.

Currently exposes :mod:`attestor.ingest.contextual` — Anthropic-style
chunk-context prefixing that runs an LLM pre-pass to situate each new
memory in its session before embedding. See ``contextual.py`` for the
full contract.
"""

from attestor.ingest.contextual import (
    ContextPrefix,
    ContextualEmbedder,
    build_context_prefix,
)

__all__ = [
    "ContextPrefix",
    "ContextualEmbedder",
    "build_context_prefix",
]
