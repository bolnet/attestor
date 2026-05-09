# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Read-only web UI for Attestor — Forensic Archive aesthetic.

Mount ``attestor.ui.app:app`` into a Starlette application, or launch
standalone via ``attestor ui``.
"""

from attestor.ui.app import create_ui_app, ui_routes  # noqa: F401
