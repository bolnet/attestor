"""Read-only web UI for memwright — Forensic Archive aesthetic.

Mount `agent_memory.ui.app:app` into a Starlette application, or launch
standalone via `memwright ui`.
"""

from agent_memory.ui.app import create_ui_app  # noqa: F401
