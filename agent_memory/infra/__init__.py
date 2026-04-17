# agent_memory/infra/__init__.py
"""Optional infrastructure helpers (docker, etc.).

Sub-modules in this package are opt-in and guarded by pyproject extras.
Importing a sub-module whose extra is not installed raises
:class:`agent_memory.store._extras.MissingExtraError` with install instructions.
"""
