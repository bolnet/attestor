# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Shared helpers for the extraction / consolidation LLM pipelines.

Single source of truth for the markdown-fence stripper that was
previously duplicated across five modules. A bug fix here propagates
to every caller automatically.
"""

from __future__ import annotations


def strip_markdown_fences(text: str) -> str:
    """Remove ``\\`\\`\\`json ... \\`\\`\\``` style code fences.

    Tolerates the common LLM patterns:
      - ``\\`\\`\\`json\\n{...}\\n\\`\\`\\```` (language tag + body + closer)
      - ``\\`\\`\\`\\n{...}\\n\\`\\`\\```` (no language tag)
      - bare body without fences (returned unchanged)

    Returns the inner text stripped of surrounding whitespace.
    """
    s = text.strip()
    if not s.startswith("```"):
        return s
    # Drop every line that starts with the fence sentinel — covers both
    # the leading ``\\`\\`\\`json`` and the trailing ``\\`\\`\\```` even when
    # they sit on otherwise-empty lines or are indented.
    lines = [line for line in s.split("\n") if not line.strip().startswith("```")]
    return "\n".join(lines).strip()
