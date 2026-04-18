"""Read-only web UI for memwright — Forensic Archive aesthetic.

Mount `attestor.ui.app:app` into a Starlette application, or launch
standalone via `memwright ui`.
"""

from attestor.ui.app import create_ui_app  # noqa: F401
