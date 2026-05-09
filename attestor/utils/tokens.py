# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Token estimation utilities."""


def estimate_tokens(text: str) -> int:
    """Estimate token count for a string. Approximation: ~1.3 tokens per word."""
    return int(len(text.split()) * 1.3)
