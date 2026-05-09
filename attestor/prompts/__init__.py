# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Default prompt modules shipped with attestor.

Currently:
  chain_of_note — Chain-of-Note reading prompt for ContextPack consumers.
"""

from attestor.prompts.chain_of_note import DEFAULT_CHAIN_OF_NOTE_PROMPT

__all__ = ["DEFAULT_CHAIN_OF_NOTE_PROMPT"]
