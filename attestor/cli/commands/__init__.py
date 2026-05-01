"""CLI command handlers, grouped by domain.

Each module exposes a set of ``_cmd_*`` functions with the signature
``(args: argparse.Namespace) -> None`` — wired up by
:mod:`attestor.cli.main`'s argparse dispatcher.
"""

from __future__ import annotations
