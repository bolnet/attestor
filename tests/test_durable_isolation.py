"""Hard rule: recall + hooks never touch Temporal.

``attestor/retrieval/**`` and ``attestor/hooks/**`` must not import
``attestor.durable`` (statically or dynamically), and importing them at
runtime must not pull ``attestor.durable`` or ``temporalio`` into
``sys.modules``. See docs/plans/temporal-integration.md §2.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GUARDED_PACKAGES = ("attestor/retrieval", "attestor/hooks")
FORBIDDEN_PREFIXES = ("attestor.durable", "temporalio")


def _python_files() -> list[Path]:
    files: list[Path] = []
    for pkg in GUARDED_PACKAGES:
        files.extend(sorted((REPO_ROOT / pkg).rglob("*.py")))
    assert files, "guarded packages not found — repo layout changed?"
    return files


def _static_offenders(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    hits: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            hits += [a.name for a in node.names if a.name.startswith(FORBIDDEN_PREFIXES)]
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith(FORBIDDEN_PREFIXES):
                hits.append(mod)
            # ``from attestor import durable`` — same violation, different spelling.
            if mod == "attestor":
                hits += [a.name for a in node.names if a.name == "durable"]
        elif (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith(FORBIDDEN_PREFIXES)
        ):
            # importlib.import_module("attestor.durable.x") style escapes.
            hits.append(node.value)
    return hits


@pytest.mark.unit
@pytest.mark.parametrize("path", _python_files(), ids=lambda p: str(p.relative_to(REPO_ROOT)))
def test_no_static_reference_to_durable(path: Path):
    offenders = _static_offenders(path)
    assert not offenders, f"{path.relative_to(REPO_ROOT)} references {offenders}"


@pytest.mark.unit
def test_runtime_import_of_recall_and_hooks_does_not_load_durable():
    """Import the guarded packages in a fresh interpreter and inspect sys.modules."""
    modules = [
        "attestor.retrieval",
        "attestor.retrieval.orchestrator",
        "attestor.hooks",
        "attestor.hooks.session_start",
        "attestor.hooks.post_tool_use",
        "attestor.hooks.stop",
    ]
    code = (
        "import importlib, sys\n"
        f"for m in {modules!r}:\n"
        "    importlib.import_module(m)\n"
        "bad = sorted(m for m in sys.modules "
        f"if m.startswith({FORBIDDEN_PREFIXES!r}))\n"
        "print(','.join(bad))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(REPO_ROOT), check=False,
    )
    assert proc.returncode == 0, proc.stderr
    loaded = proc.stdout.strip()
    assert loaded == "", f"recall/hooks import pulled in: {loaded}"


@pytest.mark.unit
def test_consolidation_package_does_not_import_temporalio_eagerly():
    """The durable branch in the consolidator is lazy — plain installs
    (no ``attestor[durable]``) must still import the consolidation package."""
    code = (
        "import sys\n"
        "import attestor.consolidation\n"
        "loaded = sorted(m for m in sys.modules "
        "if m.startswith(('temporalio', 'attestor.durable')))\n"
        "print(','.join(loaded))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True,
        cwd=str(REPO_ROOT), check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""
