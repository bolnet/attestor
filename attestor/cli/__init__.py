"""Attestor CLI — split from a 1220-line module.

The console-script entry point ``attestor = "attestor.cli:main"`` resolves
through this package's re-export of :func:`main`.

We also re-export ``asyncio`` here because tests historically monkey-patch
``attestor.cli.asyncio`` to no-op ``asyncio.run`` calls inside the ``mcp``
subcommand handler.  The handler accesses asyncio via this package
attribute so that ``patch("attestor.cli.asyncio")`` continues to work after
the file split.
"""

from __future__ import annotations

import asyncio  # re-exported so tests can patch ``attestor.cli.asyncio``

from attestor.cli.main import main

__all__ = ["main", "asyncio"]
