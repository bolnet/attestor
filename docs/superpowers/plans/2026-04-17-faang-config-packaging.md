# FAANG-Tier Config & Packaging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Bring memwright's config and packaging to FAANG-grade — typed, layered, plugin-extensible, and trivially installable like Cargo/Poetry/uv.

**Architecture:** Six independent slices, each shippable on its own.
1. Quick wins (version sync, `config show/validate`, JSON→TOML migration shim)
2. Extras-based packaging with import gating and helpful errors
3. Pydantic-typed layered config with profiles and env-var resolution
4. Entry-point plugin registry for backends
5. `memwright init` interactive + non-interactive wizard
6. Optional docker-extra restored or `mode: local` formally dropped

**Tech Stack:** Python 3.10+, `pydantic>=2`, `tomli`/`tomllib`, `importlib.metadata`, `prompt_toolkit` (for wizard), pytest.

**Recommended order:** 1 → 2 → 3 → 4 → 5 → 6

---

## File Map

| Path | Responsibility | Action |
|------|----------------|--------|
| `agent_memory/__init__.py` | Public API, version | Modify (read version from metadata) |
| `agent_memory/_version.py` | Single source of truth fallback | Create |
| `agent_memory/utils/config.py` | Legacy JSON config | Modify (add migration shim) |
| `agent_memory/config/__init__.py` | New typed config package | Create |
| `agent_memory/config/schema.py` | Pydantic models | Create |
| `agent_memory/config/loader.py` | Layered TOML/env loader | Create |
| `agent_memory/config/migrate.py` | JSON→TOML migration | Create |
| `agent_memory/store/registry.py` | Backend registry | Modify (entry-points + import gating) |
| `agent_memory/store/_extras.py` | Helpful import-error wrapper | Create |
| `agent_memory/cli.py` | CLI entry | Modify (add `config`, `init` subcommands) |
| `agent_memory/cli/init_wizard.py` | Interactive init flow | Create |
| `pyproject.toml` | Packaging | Modify (extras, entry-points, version source) |
| `tests/test_version.py` | Version sync test | Create |
| `tests/test_config_*.py` | Config tests | Create per slice |

---

# SLICE 1 — Quick Wins (version sync, `config show/validate`, JSON→TOML migration)

**Estimated effort:** half day. Smallest blast radius — start here.

## Task 1.1: Single-source version

**Files:**
- Create: `agent_memory/_version.py`
- Modify: `agent_memory/__init__.py`
- Modify: `pyproject.toml` (add dynamic version)
- Test: `tests/test_version.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_version.py
from importlib.metadata import version as pkg_version

import agent_memory


def test_version_matches_metadata():
    """agent_memory.__version__ must match the installed package metadata."""
    assert agent_memory.__version__ == pkg_version("memwright")


def test_version_is_semver():
    """Version must be a valid semver-ish string."""
    parts = agent_memory.__version__.split(".")
    assert len(parts) >= 2, agent_memory.__version__
    assert all(p[0].isdigit() for p in parts[:2])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_version.py -v`
Expected: FAIL — `__version__` is `"0.1.2"`, package metadata is `"2.0.1"`.

- [ ] **Step 3: Create `_version.py` fallback**

```python
# agent_memory/_version.py
"""Single source of truth for version. Reads from installed package metadata."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version as _pkg_version


def get_version() -> str:
    try:
        return _pkg_version("memwright")
    except PackageNotFoundError:
        return "0.0.0+local"
```

- [ ] **Step 4: Update `__init__.py`**

Replace the literal `__version__ = "0.1.2"` line with:

```python
# agent_memory/__init__.py
"""AgentMemory — Embedded memory for AI agents."""

from agent_memory._version import get_version
from agent_memory.core import AgentMemory
from agent_memory.models import Memory, RetrievalResult

__all__ = ["AgentMemory", "Memory", "RetrievalResult"]
__version__ = get_version()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/test_version.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_memory/_version.py agent_memory/__init__.py tests/test_version.py
git commit -m "refactor: read __version__ from package metadata to eliminate drift"
```

---

## Task 1.2: `memwright config show`

**Files:**
- Modify: `agent_memory/cli.py` (add subcommand handler + parser)
- Test: `tests/test_cli_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_cli_config.py
import json
import subprocess
import sys
from pathlib import Path


def test_config_show_outputs_json(tmp_path: Path):
    """`memwright config show <path>` prints the resolved config as JSON."""
    cfg = {"backends": ["sqlite"], "default_token_budget": 8000}
    (tmp_path / "config.json").write_text(json.dumps(cfg))

    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "show", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    out = json.loads(result.stdout)
    assert out["backends"] == ["sqlite"]
    assert out["default_token_budget"] == 8000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_cli_config.py::test_config_show_outputs_json -v`
Expected: FAIL — `config` is not a known subcommand.

- [ ] **Step 3: Add the subcommand to `cli.py`**

In `agent_memory/cli.py`, add a parser and handler. Find the section where existing subparsers are registered (search for `subparsers.add_parser`) and add:

```python
# In the parser-registration block
config_parser = subparsers.add_parser("config", help="Inspect or modify configuration")
config_sub = config_parser.add_subparsers(dest="config_command", required=True)

config_show = config_sub.add_parser("show", help="Print resolved configuration as JSON")
config_show.add_argument("path", type=Path, help="Memory store path")

config_validate = config_sub.add_parser("validate", help="Validate configuration")
config_validate.add_argument("path", type=Path, help="Memory store path")

# In the handlers dict
handlers["config"] = _cmd_config
```

Add the handler function:

```python
def _cmd_config(args):
    from agent_memory.utils.config import load_config
    if args.config_command == "show":
        cfg = load_config(args.path)
        print(json.dumps(cfg.to_dict(), indent=2))
    elif args.config_command == "validate":
        try:
            load_config(args.path)
            print("OK: config is valid")
        except Exception as e:
            print(f"INVALID: {e}", file=sys.stderr)
            sys.exit(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_cli_config.py::test_config_show_outputs_json -v`
Expected: PASS.

- [ ] **Step 5: Add validate test**

```python
# Append to tests/test_cli_config.py
def test_config_validate_passes(tmp_path: Path):
    (tmp_path / "config.json").write_text('{"backends": ["sqlite"]}')
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "validate", str(tmp_path)],
        capture_output=True, text=True, check=True,
    )
    assert "OK" in result.stdout


def test_config_validate_fails_on_bad_json(tmp_path: Path):
    (tmp_path / "config.json").write_text("{not json")
    result = subprocess.run(
        [sys.executable, "-m", "agent_memory.cli", "config", "validate", str(tmp_path)],
        capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "INVALID" in result.stderr
```

- [ ] **Step 6: Run all config tests**

Run: `poetry run pytest tests/test_cli_config.py -v`
Expected: 3 PASS.

- [ ] **Step 7: Commit**

```bash
git add agent_memory/cli.py tests/test_cli_config.py
git commit -m "feat(cli): add 'memwright config show' and 'config validate' subcommands"
```

---

## Task 1.3: TOML reader with JSON fallback (transparent to users)

**Files:**
- Modify: `agent_memory/utils/config.py`
- Test: `tests/test_config_toml.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_toml.py
from pathlib import Path

from agent_memory.utils.config import load_config


def test_load_toml(tmp_path: Path):
    """load_config prefers config.toml over config.json when both exist."""
    (tmp_path / "config.toml").write_text(
        'backends = ["sqlite"]\n'
        'default_token_budget = 9000\n'
    )
    (tmp_path / "config.json").write_text('{"backends": ["arangodb"], "default_token_budget": 1}')

    cfg = load_config(tmp_path)
    assert cfg.backends == ["sqlite"]
    assert cfg.default_token_budget == 9000


def test_json_still_works(tmp_path: Path):
    """Existing JSON configs continue to load unchanged."""
    (tmp_path / "config.json").write_text('{"backends": ["sqlite"]}')
    cfg = load_config(tmp_path)
    assert cfg.backends == ["sqlite"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_config_toml.py -v`
Expected: FAIL on `test_load_toml` — TOML not yet supported.

- [ ] **Step 3: Update `load_config` in `agent_memory/utils/config.py`**

Replace the `load_config` function with:

```python
def load_config(path: Path) -> MemoryConfig:
    """Load config from TOML (preferred) or JSON, falling back to defaults."""
    toml_file = path / "config.toml"
    json_file = path / "config.json"

    if toml_file.exists():
        try:
            import tomllib  # Python 3.11+
        except ImportError:
            import tomli as tomllib  # Fallback for 3.10
        with open(toml_file, "rb") as f:
            data = tomllib.load(f)
        return MemoryConfig.from_dict(data)

    if json_file.exists():
        with open(json_file) as f:
            data = json.load(f)
        return MemoryConfig.from_dict(data)

    return MemoryConfig()
```

- [ ] **Step 4: Add `tomli` to deps for Python 3.10 in `pyproject.toml`**

Find the `dependencies = [` block and add a conditional dep:

```toml
dependencies = [
    "chromadb>=0.4.0",
    "networkx>=3.0",
    "sentence-transformers>=2.0.0",
    "mcp>=1.0.0",
    'tomli>=2.0; python_version < "3.11"',
]
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `poetry install && poetry run pytest tests/test_config_toml.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_memory/utils/config.py pyproject.toml tests/test_config_toml.py
git commit -m "feat(config): support config.toml (preferred) with JSON fallback"
```

---

## Task 1.4: `memwright config migrate` (JSON → TOML)

**Files:**
- Create: `agent_memory/config/__init__.py` (placeholder)
- Create: `agent_memory/config/migrate.py`
- Modify: `agent_memory/cli.py`
- Test: `tests/test_config_migrate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_migrate.py
import json
import tomllib
from pathlib import Path

from agent_memory.config.migrate import migrate_json_to_toml


def test_migrate_writes_equivalent_toml(tmp_path: Path):
    src = tmp_path / "config.json"
    src.write_text(json.dumps({
        "backends": ["arangodb"],
        "default_token_budget": 16000,
        "arangodb": {"mode": "cloud", "url": "http://localhost:8529"},
    }))

    out = migrate_json_to_toml(src)
    assert out == tmp_path / "config.toml"

    with open(out, "rb") as f:
        data = tomllib.load(f)
    assert data["backends"] == ["arangodb"]
    assert data["arangodb"]["mode"] == "cloud"


def test_migrate_preserves_original(tmp_path: Path):
    src = tmp_path / "config.json"
    src.write_text('{"backends": ["sqlite"]}')
    migrate_json_to_toml(src)
    assert src.exists(), "original JSON should be kept (rename to .bak instead)"
    assert (tmp_path / "config.json.bak").exists() or src.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_config_migrate.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the migration module**

```python
# agent_memory/config/__init__.py
"""Typed, layered configuration for memwright."""
```

```python
# agent_memory/config/migrate.py
"""Convert legacy config.json to config.toml."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any


def _to_toml(data: dict[str, Any], indent: int = 0) -> str:
    """Tiny TOML emitter — only handles the shapes memwright produces."""
    scalars: list[str] = []
    tables: list[str] = []
    for key, val in data.items():
        if isinstance(val, dict):
            tables.append(f"\n[{key}]\n" + _to_toml(val, indent + 1))
        elif isinstance(val, str):
            scalars.append(f'{key} = "{val}"')
        elif isinstance(val, bool):
            scalars.append(f"{key} = {'true' if val else 'false'}")
        elif isinstance(val, (int, float)):
            scalars.append(f"{key} = {val}")
        elif isinstance(val, list):
            inner = ", ".join(f'"{x}"' if isinstance(x, str) else str(x) for x in val)
            scalars.append(f"{key} = [{inner}]")
        else:
            scalars.append(f"{key} = {json.dumps(val)}")
    return "\n".join(scalars) + "".join(tables)


def migrate_json_to_toml(src: Path) -> Path:
    """Convert config.json to config.toml in the same directory.

    The original JSON is preserved as config.json.bak.
    Returns the path to the new TOML file.
    """
    if src.suffix != ".json":
        raise ValueError(f"Expected .json file, got {src}")
    with open(src) as f:
        data = json.load(f)

    dst = src.with_suffix(".toml")
    dst.write_text(_to_toml(data) + "\n")
    shutil.copy(src, src.with_suffix(".json.bak"))
    return dst
```

- [ ] **Step 4: Wire CLI subcommand**

In `agent_memory/cli.py`, extend the `config` subparser block from Task 1.2:

```python
config_migrate_p = config_sub.add_parser("migrate", help="Migrate config.json to config.toml")
config_migrate_p.add_argument("path", type=Path, help="Memory store path")
```

Extend `_cmd_config`:

```python
elif args.config_command == "migrate":
    from agent_memory.config.migrate import migrate_json_to_toml
    src = args.path / "config.json"
    if not src.exists():
        print(f"No config.json at {args.path}", file=sys.stderr)
        sys.exit(1)
    out = migrate_json_to_toml(src)
    print(f"Wrote {out}; original kept as {src.with_suffix('.json.bak')}")
```

- [ ] **Step 5: Run all tests**

Run: `poetry run pytest tests/test_config_migrate.py tests/test_cli_config.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_memory/config/ agent_memory/cli.py tests/test_config_migrate.py
git commit -m "feat(config): add 'config migrate' to convert JSON to TOML"
```

---

## Task 1.5: Slice 1 wrap — README update

- [ ] **Step 1: Update `README.md`** with a new "Configuration" section showing both formats:

```markdown
## Configuration

memwright reads `config.toml` (preferred) or `config.json` from your store path.

### Inspect / validate / migrate
```bash
memwright config show ./mystore       # print resolved config as JSON
memwright config validate ./mystore   # exit 1 on errors
memwright config migrate ./mystore    # convert legacy JSON → TOML
```

### Example `config.toml`
```toml
backends = ["arangodb"]
default_token_budget = 16000

[arangodb]
mode = "cloud"
url = "http://localhost:8529"
database = "memwright_user"
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document config show/validate/migrate and TOML format"
```

---

# SLICE 2 — Extras-Based Packaging with Import Gating

**Estimated effort:** half day. Cleans up install footprint and gives users actionable errors.

## Task 2.1: Helpful import-error wrapper

**Files:**
- Create: `agent_memory/store/_extras.py`
- Test: `tests/test_extras.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extras.py
import pytest

from agent_memory.store._extras import require_extra


def test_require_extra_passes_when_present():
    # `json` is always available, so this should not raise
    mod = require_extra("json", extra="core", package="memwright")
    assert mod is not None


def test_require_extra_raises_helpful_error():
    with pytest.raises(ImportError) as exc_info:
        require_extra("definitely_not_a_module_xyz", extra="arangodb", package="memwright")

    msg = str(exc_info.value)
    assert "memwright[arangodb]" in msg
    assert "pip install" in msg
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_extras.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the helper**

```python
# agent_memory/store/_extras.py
"""Helper for gating optional backend imports with actionable error messages."""

from __future__ import annotations

import importlib
from typing import Any


def require_extra(module_name: str, *, extra: str, package: str = "memwright") -> Any:
    """Import a module, raising an actionable ImportError if it's missing.

    Args:
        module_name: The module to import (e.g., "arango").
        extra: The pyproject.toml extras name (e.g., "arangodb").
        package: The distribution name. Defaults to "memwright".

    Returns:
        The imported module.

    Raises:
        ImportError: With a message telling the user exactly which extras to install.
    """
    try:
        return importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(
            f"Missing dependency for backend {extra!r}: {module_name}. "
            f"Install with: pip install '{package}[{extra}]'"
        ) from e
```

- [ ] **Step 4: Run test to verify it passes**

Run: `poetry run pytest tests/test_extras.py -v`
Expected: 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_memory/store/_extras.py tests/test_extras.py
git commit -m "feat(store): add require_extra helper for actionable backend import errors"
```

---

## Task 2.2: Gate backend imports through `require_extra`

**Files:**
- Modify: `agent_memory/store/registry.py`
- Modify: `agent_memory/store/arango_backend.py` (top imports)
- Modify: `agent_memory/store/postgres_backend.py`
- Modify: `agent_memory/store/aws_backend.py`
- Modify: `agent_memory/store/azure_backend.py`
- Modify: `agent_memory/store/gcp_backend.py`
- Test: `tests/test_registry_gating.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_gating.py
import builtins
import sys
from unittest.mock import patch

import pytest

from agent_memory.store.registry import instantiate_backend


def test_arango_missing_raises_actionable_error(tmp_path):
    """When python-arango is missing, instantiation gives a helpful message."""
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "arango":
            raise ImportError("No module named 'arango'")
        return real_import(name, *args, **kwargs)

    sys.modules.pop("agent_memory.store.arango_backend", None)
    with patch("builtins.__import__", fake_import):
        with pytest.raises(ImportError) as exc_info:
            instantiate_backend("arangodb", tmp_path, {"mode": "local"})
    assert "memwright[arangodb]" in str(exc_info.value)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_registry_gating.py -v`
Expected: FAIL — current import error is the raw `ImportError: No module named 'arango'`.

- [ ] **Step 3: Update `arango_backend.py`**

Replace the top of `agent_memory/store/arango_backend.py`:

```python
"""ArangoDB backend — single backend for document, vector, and graph roles."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent_memory.store._extras import require_extra

_arango = require_extra("arango", extra="arangodb")
ArangoClient = _arango.ArangoClient

from agent_memory.models import Memory
from agent_memory.store.base import DocumentStore, VectorStore, GraphStore
from agent_memory.store.connection import CloudConnection
```

- [ ] **Step 4: Repeat for each cloud backend**

Apply the same pattern to:
- `postgres_backend.py` → `require_extra("psycopg2", extra="postgres")`
- `aws_backend.py` → `require_extra("boto3", extra="aws")`
- `azure_backend.py` → `require_extra("azure.cosmos", extra="azure")`
- `gcp_backend.py` → `require_extra("google.cloud.firestore", extra="gcp")`

For each, identify the top-level third-party import and route it through `require_extra`.

- [ ] **Step 5: Run gating test**

Run: `poetry run pytest tests/test_registry_gating.py -v`
Expected: PASS.

- [ ] **Step 6: Run full suite to confirm no regressions**

Run: `poetry run pytest tests/ -x -q`
Expected: all green (or only pre-existing failures unrelated to this work).

- [ ] **Step 7: Commit**

```bash
git add agent_memory/store/
git commit -m "feat(store): gate cloud backend imports with actionable extras errors"
```

---

## Task 2.3: Add `[mcp]` and `[docker]` extras

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Edit `pyproject.toml`**

Move `mcp>=1.0.0` from required `dependencies` into `[project.optional-dependencies]`, and add `docker`:

```toml
dependencies = [
    "chromadb>=0.4.0",
    "networkx>=3.0",
    "sentence-transformers>=2.0.0",
    'tomli>=2.0; python_version < "3.11"',
]

[project.optional-dependencies]
mcp = ["mcp>=1.0.0"]
docker = ["docker>=7.0"]
extraction = ["openai>=1.0.0"]
arangodb = ["python-arango>=8.0.0"]
postgres = ["psycopg2-binary>=2.9.0"]
aws = ["boto3>=1.28.0"]
azure = ["azure-cosmos>=4.5.0", "azure-identity>=1.14.0", "azure-search-documents>=11.4.0"]
gcp = ["google-cloud-firestore>=2.11.0", "google-cloud-aiplatform>=1.38.0"]
cloud-embeddings = ["openai>=1.0.0"]
lambda = ["mangum>=0.17.0", "starlette>=0.27.0"]
all = [
    "mcp>=1.0.0",
    "docker>=7.0",
    "openai>=1.0.0",
    "python-arango>=8.0.0",
    "psycopg2-binary>=2.9.0",
    "boto3>=1.28.0",
    "azure-cosmos>=4.5.0",
    "azure-identity>=1.14.0",
    "azure-search-documents>=11.4.0",
    "google-cloud-firestore>=2.11.0",
    "google-cloud-aiplatform>=1.38.0",
]
```

- [ ] **Step 2: Gate MCP server import**

In `agent_memory/cli.py`, find the `mcp` subcommand handler and route through `require_extra`. In `agent_memory/mcp/server.py` add at the top:

```python
from agent_memory.store._extras import require_extra
require_extra("mcp", extra="mcp")
```

- [ ] **Step 3: Update README install line**

```markdown
## Install
```bash
pipx install "memwright[mcp]"               # core + MCP server (most users)
pipx install "memwright[mcp,arangodb]"      # add ArangoDB backend
pipx install "memwright[all]"               # everything
```
```

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml agent_memory/mcp/server.py agent_memory/cli.py README.md
git commit -m "feat(packaging): split mcp and docker into optional extras"
```

---

# SLICE 3 — Pydantic-Typed Layered Config

**Estimated effort:** 1 day. Replaces the ad-hoc dataclass loader with a typed, layered, profile-aware config system.

## Task 3.1: Pydantic schema

**Files:**
- Modify: `pyproject.toml` (add `pydantic>=2`)
- Create: `agent_memory/config/schema.py`
- Test: `tests/test_config_schema.py`

- [ ] **Step 1: Add pydantic dep**

```toml
# pyproject.toml — under dependencies
dependencies = [
    ...
    "pydantic>=2.5",
]
```

Run: `poetry install`

- [ ] **Step 2: Write the failing test**

```python
# tests/test_config_schema.py
import pytest
from pydantic import ValidationError

from agent_memory.config.schema import MemwrightSettings


def test_defaults():
    s = MemwrightSettings()
    assert s.backends == ["sqlite", "chroma", "networkx"]
    assert s.default_token_budget == 16000
    assert s.fusion_mode == "rrf"


def test_validates_fusion_mode():
    with pytest.raises(ValidationError):
        MemwrightSettings(fusion_mode="invalid_mode")


def test_validates_token_budget_positive():
    with pytest.raises(ValidationError):
        MemwrightSettings(default_token_budget=-1)


def test_arango_settings_nested():
    s = MemwrightSettings(
        backends=["arangodb"],
        arangodb={"mode": "local", "port": 8529},
    )
    assert s.arangodb.mode == "local"
    assert s.arangodb.port == 8529
```

- [ ] **Step 3: Run test to verify it fails**

Run: `poetry run pytest tests/test_config_schema.py -v`
Expected: FAIL — schema does not exist.

- [ ] **Step 4: Create the schema**

```python
# agent_memory/config/schema.py
"""Typed configuration schema using Pydantic v2."""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class AuthSettings(BaseModel):
    username: str = ""
    password: str = ""
    token: str = ""
    api_key: str = ""

    model_config = {"extra": "allow"}


class TLSSettings(BaseModel):
    verify: bool = True
    ca_cert: Optional[str] = None
    ca_cert_base64: Optional[str] = None


class ArangoSettings(BaseModel):
    mode: Literal["local", "cloud"] = "local"
    url: str = "http://localhost:8529"
    port: int = 8529
    database: str = "memwright"
    auth: AuthSettings = Field(default_factory=AuthSettings)
    tls: TLSSettings = Field(default_factory=TLSSettings)
    docker: bool = False


class PostgresSettings(BaseModel):
    mode: Literal["local", "cloud"] = "local"
    url: str = "postgresql://localhost:5432"
    port: int = 5432
    database: str = "memwright"
    auth: AuthSettings = Field(default_factory=AuthSettings)
    tls: TLSSettings = Field(default_factory=TLSSettings)


class MemwrightSettings(BaseModel):
    """Top-level memwright configuration."""

    backends: List[str] = Field(default_factory=lambda: ["sqlite", "chroma", "networkx"])
    default_token_budget: int = Field(16000, gt=0)
    min_results: int = Field(3, ge=0)
    enable_mmr: bool = True
    mmr_lambda: float = Field(0.7, ge=0.0, le=1.0)
    fusion_mode: Literal["rrf", "graph_blend"] = "rrf"
    confidence_gate: float = Field(0.0, ge=0.0, le=1.0)
    confidence_decay_rate: float = Field(0.001, ge=0.0)
    confidence_boost_rate: float = Field(0.03, ge=0.0)

    arangodb: ArangoSettings = Field(default_factory=ArangoSettings)
    postgres: PostgresSettings = Field(default_factory=PostgresSettings)

    profiles: Dict[str, Dict] = Field(default_factory=dict)

    model_config = {"extra": "allow"}

    @field_validator("backends")
    @classmethod
    def _backends_non_empty(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("backends list cannot be empty")
        return v
```

- [ ] **Step 5: Run test to verify it passes**

Run: `poetry run pytest tests/test_config_schema.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_memory/config/schema.py tests/test_config_schema.py pyproject.toml
git commit -m "feat(config): add Pydantic schema with validation"
```

---

## Task 3.2: Layered loader (defaults → file → env vars → CLI)

**Files:**
- Create: `agent_memory/config/loader.py`
- Test: `tests/test_config_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_loader.py
from pathlib import Path

import pytest

from agent_memory.config.loader import load_settings


def test_loads_defaults_when_no_file(tmp_path: Path):
    s = load_settings(store_path=tmp_path)
    assert s.backends == ["sqlite", "chroma", "networkx"]


def test_loads_from_toml(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        'backends = ["arangodb"]\n[arangodb]\nport = 9999\n'
    )
    s = load_settings(store_path=tmp_path)
    assert s.backends == ["arangodb"]
    assert s.arangodb.port == 9999


def test_env_var_overrides_file(tmp_path: Path, monkeypatch):
    (tmp_path / "config.toml").write_text("default_token_budget = 1000\n")
    monkeypatch.setenv("MEMWRIGHT_DEFAULT_TOKEN_BUDGET", "5555")
    s = load_settings(store_path=tmp_path)
    assert s.default_token_budget == 5555


def test_profile_selection(tmp_path: Path):
    (tmp_path / "config.toml").write_text(
        'backends = ["sqlite"]\n'
        '[profiles.prod]\n'
        'backends = ["arangodb"]\n'
    )
    s = load_settings(store_path=tmp_path, profile="prod")
    assert s.backends == ["arangodb"]


def test_unknown_profile_raises(tmp_path: Path):
    (tmp_path / "config.toml").write_text('backends = ["sqlite"]\n')
    with pytest.raises(ValueError, match="profile.*nope"):
        load_settings(store_path=tmp_path, profile="nope")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `poetry run pytest tests/test_config_loader.py -v`
Expected: FAIL — loader does not exist.

- [ ] **Step 3: Create the loader**

```python
# agent_memory/config/loader.py
"""Layered config loader: defaults → file → env vars → CLI overrides."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from agent_memory.config.schema import MemwrightSettings


_ENV_PREFIX = "MEMWRIGHT_"


def _read_toml(path: Path) -> Dict[str, Any]:
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    with open(path, "rb") as f:
        return tomllib.load(f)


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path) as f:
        return json.load(f)


def _load_file(store_path: Path) -> Dict[str, Any]:
    toml_file = store_path / "config.toml"
    json_file = store_path / "config.json"
    if toml_file.exists():
        return _read_toml(toml_file)
    if json_file.exists():
        return _read_json(json_file)
    return {}


def _coerce_env_value(raw: str) -> Any:
    """Best-effort coerce env string to int/float/bool/JSON; fall back to string."""
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.startswith(("[", "{")):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return raw


def _env_overrides() -> Dict[str, Any]:
    """Read MEMWRIGHT_* env vars into a flat dict (lowercased keys)."""
    overrides: Dict[str, Any] = {}
    for key, val in os.environ.items():
        if not key.startswith(_ENV_PREFIX):
            continue
        name = key[len(_ENV_PREFIX):].lower()
        overrides[name] = _coerce_env_value(val)
    return overrides


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_settings(
    store_path: Path,
    *,
    profile: Optional[str] = None,
    cli_overrides: Optional[Dict[str, Any]] = None,
) -> MemwrightSettings:
    """Resolve config across all layers and return a validated MemwrightSettings.

    Layer order (last wins):
        1. Pydantic defaults
        2. File (config.toml or config.json)
        3. Selected profile section
        4. MEMWRIGHT_* env vars
        5. cli_overrides
    """
    file_data = _load_file(store_path)

    profiles = file_data.pop("profiles", {})
    if profile:
        if profile not in profiles:
            raise ValueError(f"Unknown profile {profile!r}; available: {sorted(profiles)}")
        file_data = _deep_merge(file_data, profiles[profile])

    env_data = _env_overrides()
    merged = _deep_merge(file_data, env_data)
    if cli_overrides:
        merged = _deep_merge(merged, cli_overrides)

    return MemwrightSettings(**merged)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `poetry run pytest tests/test_config_loader.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_memory/config/loader.py tests/test_config_loader.py
git commit -m "feat(config): layered loader with profiles and env-var overrides"
```

---

## Task 3.3: Wire new loader into core (keep legacy path as fallback)

**Files:**
- Modify: `agent_memory/core.py` (replace `load_config` call site)
- Modify: `agent_memory/utils/config.py` (delegate to new loader)
- Test: `tests/test_core_uses_new_loader.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_core_uses_new_loader.py
from pathlib import Path
from agent_memory import AgentMemory


def test_core_reads_toml_via_new_loader(tmp_path: Path):
    """AgentMemory should respect TOML configs."""
    (tmp_path / "config.toml").write_text(
        'backends = ["sqlite"]\n'
        'default_token_budget = 7777\n'
    )
    with AgentMemory(tmp_path) as mem:
        assert mem.config.default_token_budget == 7777
```

- [ ] **Step 2: Run test**

Run: `poetry run pytest tests/test_core_uses_new_loader.py -v`
Expected: depends on existing wiring — likely FAIL because old loader doesn't read TOML in core init.

- [ ] **Step 3: Update `agent_memory/utils/config.py` to delegate**

Replace the body of `load_config` with:

```python
def load_config(path: Path) -> MemoryConfig:
    """Load config via the new layered loader, returning a legacy MemoryConfig.

    Kept for backward compat; new code should use agent_memory.config.load_settings.
    """
    from agent_memory.config.loader import load_settings
    settings = load_settings(path)
    data = settings.model_dump()
    return MemoryConfig.from_dict(data)
```

- [ ] **Step 4: Run all config tests**

Run: `poetry run pytest tests/test_config_loader.py tests/test_config_schema.py tests/test_config_toml.py tests/test_core_uses_new_loader.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add agent_memory/utils/config.py tests/test_core_uses_new_loader.py
git commit -m "refactor(config): route legacy loader through new Pydantic loader"
```

---

# SLICE 4 — Entry-Point Plugin Registry

**Estimated effort:** half day. Lets third parties ship `memwright-redis` and have it auto-register.

## Task 4.1: Discover backends from entry points

**Files:**
- Modify: `agent_memory/store/registry.py`
- Modify: `pyproject.toml` (register built-ins as entry points)
- Test: `tests/test_registry_entrypoints.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry_entrypoints.py
from agent_memory.store.registry import BACKEND_REGISTRY, discover_backends


def test_builtins_present_after_discover():
    discover_backends()
    for name in ("sqlite", "chroma", "networkx", "arangodb", "postgres"):
        assert name in BACKEND_REGISTRY, f"{name} should be discovered"


def test_discover_is_idempotent():
    discover_backends()
    snapshot = dict(BACKEND_REGISTRY)
    discover_backends()
    assert BACKEND_REGISTRY == snapshot
```

- [ ] **Step 2: Run test**

Run: `poetry run pytest tests/test_registry_entrypoints.py -v`
Expected: FAIL — `discover_backends` does not exist.

- [ ] **Step 3: Add discovery to registry**

In `agent_memory/store/registry.py`, add at the bottom:

```python
def discover_backends() -> None:
    """Populate BACKEND_REGISTRY from importlib.metadata entry points.

    Entry-point group: 'memwright.backends'
    Each entry point should resolve to a dict with the same shape as the static registry entries:
        { "module": "...", "class": "...", "roles": {...}, "init_style": "..." }
    """
    from importlib.metadata import entry_points

    for ep in entry_points(group="memwright.backends"):
        if ep.name in BACKEND_REGISTRY:
            continue
        try:
            entry = ep.load()
        except Exception as e:
            logger.warning("Failed to load backend plugin %r: %s", ep.name, e)
            continue
        BACKEND_REGISTRY[ep.name] = entry
```

Call `discover_backends()` lazily on first use — modify `resolve_backends`:

```python
def resolve_backends(backends: Optional[List[str]] = None) -> Dict[str, str]:
    discover_backends()  # populate from entry points (idempotent)
    ...
```

- [ ] **Step 4: Register built-ins as entry points in `pyproject.toml`**

```toml
[project.entry-points."memwright.backends"]
sqlite = "agent_memory.store.registry:_BUILTIN_SQLITE"
chroma = "agent_memory.store.registry:_BUILTIN_CHROMA"
networkx = "agent_memory.store.registry:_BUILTIN_NETWORKX"
arangodb = "agent_memory.store.registry:_BUILTIN_ARANGO"
postgres = "agent_memory.store.registry:_BUILTIN_POSTGRES"
aws = "agent_memory.store.registry:_BUILTIN_AWS"
azure = "agent_memory.store.registry:_BUILTIN_AZURE"
gcp = "agent_memory.store.registry:_BUILTIN_GCP"
```

In `agent_memory/store/registry.py`, expose the registry entries as module-level constants:

```python
_BUILTIN_SQLITE = BACKEND_REGISTRY["sqlite"]
_BUILTIN_CHROMA = BACKEND_REGISTRY["chroma"]
_BUILTIN_NETWORKX = BACKEND_REGISTRY["networkx"]
_BUILTIN_ARANGO = BACKEND_REGISTRY["arangodb"]
_BUILTIN_POSTGRES = BACKEND_REGISTRY["postgres"]
_BUILTIN_AWS = BACKEND_REGISTRY["aws"]
_BUILTIN_AZURE = BACKEND_REGISTRY["azure"]
_BUILTIN_GCP = BACKEND_REGISTRY["gcp"]
```

- [ ] **Step 5: Reinstall to register entry points**

Run: `poetry install`

- [ ] **Step 6: Run tests**

Run: `poetry run pytest tests/test_registry_entrypoints.py -v`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add agent_memory/store/registry.py pyproject.toml tests/test_registry_entrypoints.py
git commit -m "feat(registry): discover backends via importlib.metadata entry points"
```

---

## Task 4.2: Document third-party plugin contract

**Files:**
- Create: `docs/plugins.md`

- [ ] **Step 1: Write `docs/plugins.md`**

```markdown
# Writing a memwright Backend Plugin

Third parties can ship a backend (e.g., `memwright-redis`) that auto-registers.

## Minimal plugin layout

```
memwright_redis/
  pyproject.toml
  src/memwright_redis/
    __init__.py
    backend.py
```

## `pyproject.toml`

```toml
[project]
name = "memwright-redis"
version = "0.1.0"
dependencies = ["memwright>=2.1", "redis>=5.0"]

[project.entry-points."memwright.backends"]
redis = "memwright_redis:REGISTRY_ENTRY"
```

## `__init__.py`

```python
REGISTRY_ENTRY = {
    "module": "memwright_redis.backend",
    "class": "RedisBackend",
    "roles": {"document"},
    "init_style": "config",
}
```

## `backend.py`

Implement `agent_memory.store.base.DocumentStore` (and/or `VectorStore`, `GraphStore`).

## Selecting your backend

Users add it to `config.toml`:

```toml
backends = ["redis", "chroma", "networkx"]

[redis]
url = "redis://localhost:6379"
```
```

- [ ] **Step 2: Commit**

```bash
git add docs/plugins.md
git commit -m "docs: add backend plugin authoring guide"
```

---

# SLICE 5 — `memwright init` Wizard

**Estimated effort:** 1 day. The on-ramp moment that converts curious devs into users.

## Task 5.1: Non-interactive `init` (foundation)

**Files:**
- Create: `agent_memory/cli/__init__.py`
- Create: `agent_memory/cli/init_wizard.py`
- Modify: `agent_memory/cli.py` (delegate `init` subcommand)
- Test: `tests/test_init_wizard.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_wizard.py
from pathlib import Path

from agent_memory.cli.init_wizard import init_store


def test_init_creates_store_with_default_backend(tmp_path: Path):
    init_store(tmp_path, backend="sqlite", non_interactive=True)
    assert (tmp_path / "config.toml").exists()
    text = (tmp_path / "config.toml").read_text()
    assert "sqlite" in text


def test_init_with_arango_writes_backend_section(tmp_path: Path):
    init_store(
        tmp_path,
        backend="arangodb",
        non_interactive=True,
        backend_options={"mode": "local", "port": 8529},
    )
    text = (tmp_path / "config.toml").read_text()
    assert 'backends = ["arangodb"]' in text
    assert "[arangodb]" in text
    assert "port = 8529" in text


def test_init_refuses_to_overwrite(tmp_path: Path):
    (tmp_path / "config.toml").write_text("# existing\n")
    import pytest
    with pytest.raises(FileExistsError):
        init_store(tmp_path, backend="sqlite", non_interactive=True)
```

- [ ] **Step 2: Run tests**

Run: `poetry run pytest tests/test_init_wizard.py -v`
Expected: FAIL — module does not exist.

- [ ] **Step 3: Create the wizard module**

```python
# agent_memory/cli/__init__.py
"""CLI sub-package for memwright."""
```

```python
# agent_memory/cli/init_wizard.py
"""Interactive and non-interactive store initialization."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from agent_memory.config.migrate import _to_toml


SUPPORTED_BACKENDS = ["sqlite", "arangodb", "postgres"]


def init_store(
    path: Path,
    *,
    backend: str = "sqlite",
    non_interactive: bool = False,
    backend_options: Optional[Dict[str, Any]] = None,
) -> Path:
    """Create a new memwright store at the given path with a starter config.

    Returns the path to the written config file.

    Raises:
        FileExistsError: If config.toml or config.json already exists.
        ValueError: If backend is not supported.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported backend {backend!r}. Choose from {SUPPORTED_BACKENDS}.")

    path.mkdir(parents=True, exist_ok=True)
    config_file = path / "config.toml"
    if config_file.exists() or (path / "config.json").exists():
        raise FileExistsError(f"Config already exists at {path}")

    cfg: Dict[str, Any] = {
        "backends": [backend] if backend != "sqlite" else ["sqlite", "chroma", "networkx"],
        "default_token_budget": 16000,
    }
    if backend_options:
        cfg[backend] = backend_options

    config_file.write_text(_to_toml(cfg) + "\n")
    return config_file


def init_store_interactive(path: Path) -> Path:
    """Run the interactive prompt flow, then call init_store."""
    print(f"Initializing memwright store at: {path}")
    print(f"\nAvailable backends: {', '.join(SUPPORTED_BACKENDS)}")
    backend = input("Backend [sqlite]: ").strip() or "sqlite"

    backend_options: Dict[str, Any] = {}
    if backend == "arangodb":
        mode = input("Mode (local/cloud) [local]: ").strip() or "local"
        port = int(input("Port [8529]: ").strip() or "8529")
        backend_options = {"mode": mode, "port": port}
        if mode == "cloud":
            url = input("URL: ").strip()
            user = input("Username [root]: ").strip() or "root"
            pw = input("Password (or $ENV_VAR): ").strip()
            backend_options.update({"url": url, "auth": {"username": user, "password": pw}})
    elif backend == "postgres":
        url = input("URL [postgresql://localhost:5432]: ").strip() or "postgresql://localhost:5432"
        backend_options = {"url": url}

    return init_store(path, backend=backend, non_interactive=False, backend_options=backend_options)
```

- [ ] **Step 4: Wire into `cli.py`**

Find the existing `init` subparser in `agent_memory/cli.py` and replace its handler. If `init` does not yet exist, add:

```python
init_p = subparsers.add_parser("init", help="Initialize a new memory store")
init_p.add_argument("path", type=Path, help="Path to create the store at")
init_p.add_argument("--backend", default=None, help="Backend (sqlite, arangodb, postgres)")
init_p.add_argument("--non-interactive", action="store_true")
init_p.add_argument("--port", type=int, default=None)
init_p.add_argument("--mode", choices=["local", "cloud"], default=None)

handlers["init"] = _cmd_init


def _cmd_init(args):
    from agent_memory.cli.init_wizard import init_store, init_store_interactive
    if args.non_interactive:
        opts = {}
        if args.mode:
            opts["mode"] = args.mode
        if args.port:
            opts["port"] = args.port
        init_store(args.path, backend=args.backend or "sqlite",
                   non_interactive=True, backend_options=opts or None)
    else:
        init_store_interactive(args.path)
    print(f"Initialized store at {args.path}")
```

- [ ] **Step 5: Run tests**

Run: `poetry run pytest tests/test_init_wizard.py -v`
Expected: 3 PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_memory/cli/ agent_memory/cli.py tests/test_init_wizard.py
git commit -m "feat(cli): add 'memwright init' with interactive and non-interactive modes"
```

---

## Task 5.2: Connection-test step in init

**Files:**
- Modify: `agent_memory/cli/init_wizard.py`
- Test: `tests/test_init_connection_check.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_init_connection_check.py
from pathlib import Path
from unittest.mock import MagicMock, patch

from agent_memory.cli.init_wizard import init_store


def test_init_runs_health_check_when_requested(tmp_path: Path):
    """When verify=True, init opens the store and runs health check."""
    with patch("agent_memory.cli.init_wizard._verify_store") as mock_verify:
        mock_verify.return_value = True
        init_store(tmp_path, backend="sqlite", non_interactive=True, verify=True)
        mock_verify.assert_called_once_with(tmp_path)


def test_init_aborts_on_failed_health_check(tmp_path: Path):
    """If verify fails, init raises and removes the partial config."""
    import pytest
    with patch("agent_memory.cli.init_wizard._verify_store", return_value=False):
        with pytest.raises(RuntimeError, match="health check"):
            init_store(tmp_path, backend="sqlite", non_interactive=True, verify=True)
    assert not (tmp_path / "config.toml").exists()
```

- [ ] **Step 2: Run test**

Run: `poetry run pytest tests/test_init_connection_check.py -v`
Expected: FAIL — `verify` kwarg not yet supported.

- [ ] **Step 3: Extend `init_store`**

Add to `agent_memory/cli/init_wizard.py`:

```python
def _verify_store(path: Path) -> bool:
    """Open the store and run a health check; return True on success."""
    try:
        from agent_memory import AgentMemory
        with AgentMemory(path) as mem:
            mem.health() if hasattr(mem, "health") else mem.stats()
        return True
    except Exception:
        return False
```

Update `init_store` signature and body:

```python
def init_store(
    path: Path,
    *,
    backend: str = "sqlite",
    non_interactive: bool = False,
    backend_options: Optional[Dict[str, Any]] = None,
    verify: bool = False,
) -> Path:
    # ... existing body ...
    config_file.write_text(_to_toml(cfg) + "\n")

    if verify:
        if not _verify_store(path):
            config_file.unlink(missing_ok=True)
            raise RuntimeError("Store health check failed; rolled back config")

    return config_file
```

- [ ] **Step 4: Add `--verify` CLI flag**

In the `init` argparse block:

```python
init_p.add_argument("--verify", action="store_true", help="Run health check after init")
```

In `_cmd_init`:

```python
init_store(args.path, backend=args.backend or "sqlite",
           non_interactive=True, backend_options=opts or None,
           verify=args.verify)
```

- [ ] **Step 5: Run tests**

Run: `poetry run pytest tests/test_init_connection_check.py -v`
Expected: 2 PASS.

- [ ] **Step 6: Commit**

```bash
git add agent_memory/cli/init_wizard.py agent_memory/cli.py tests/test_init_connection_check.py
git commit -m "feat(init): add --verify to run health check after init"
```

---

## Task 5.3: `setup-claude-code` integration

**Files:**
- Modify: `agent_memory/cli.py` (extend existing `setup-claude-code` to call `init` first)

- [ ] **Step 1: Locate existing handler**

Run: `grep -n "setup-claude-code\|setup_claude_code\|_cmd_setup" /Users/aarjay/agent-memory/agent_memory/cli.py`

- [ ] **Step 2: Extend the handler**

In `_cmd_setup_claude_code` (or whatever name it has), prepend:

```python
def _cmd_setup_claude_code(args):
    from agent_memory.cli.init_wizard import init_store
    if not (args.path / "config.toml").exists() and not (args.path / "config.json").exists():
        init_store(args.path, backend=args.backend or "sqlite", non_interactive=True)
        print(f"Initialized store at {args.path}")
    # ... existing claude-code mcp config printing logic ...
```

- [ ] **Step 3: Manual verification**

Run: `poetry run memwright setup-claude-code /tmp/memwright-test --backend sqlite`
Expected: prints MCP config block; `/tmp/memwright-test/config.toml` exists.

- [ ] **Step 4: Commit**

```bash
git add agent_memory/cli.py
git commit -m "feat(cli): setup-claude-code now auto-inits store if missing"
```

---

# SLICE 6 — Decide on `mode: local` and Optional `[docker]` Extra

**Estimated effort:** few hours. Decision + small code change.

## Task 6.1: Restore `infra.docker` as optional module

**Files:**
- Create: `agent_memory/infra/__init__.py`
- Create: `agent_memory/infra/docker.py`
- Test: `tests/test_docker_manager.py`

- [ ] **Step 1: Write the failing test (skipped if docker SDK not installed)**

```python
# tests/test_docker_manager.py
import pytest

docker_sdk = pytest.importorskip("docker")

from agent_memory.infra.docker import DockerManager


def test_docker_manager_constructs():
    """DockerManager can be instantiated when docker SDK is present."""
    mgr = DockerManager()
    assert mgr is not None


def test_ensure_running_skips_when_no_daemon(monkeypatch):
    """If the daemon is unreachable, ensure_running raises a clear error."""
    mgr = DockerManager()
    monkeypatch.setattr(mgr, "_ping", lambda: False)
    with pytest.raises(RuntimeError, match="docker daemon"):
        mgr.ensure_running("test", "alpine:latest", 8080, {})
```

- [ ] **Step 2: Run test**

Run: `poetry run pytest tests/test_docker_manager.py -v`
Expected: FAIL or SKIP — module does not exist.

- [ ] **Step 3: Create the docker manager**

```python
# agent_memory/infra/__init__.py
"""Optional infrastructure helpers (docker, etc.)."""
```

```python
# agent_memory/infra/docker.py
"""Optional Docker container manager for local development backends.

Requires `pip install memwright[docker]`.
"""

from __future__ import annotations

import logging
from typing import Dict

from agent_memory.store._extras import require_extra

_docker = require_extra("docker", extra="docker")

logger = logging.getLogger("agent_memory.infra.docker")


class DockerManager:
    """Thin wrapper around the docker SDK to start/check local backend containers."""

    def __init__(self) -> None:
        self._client = _docker.from_env()

    def _ping(self) -> bool:
        try:
            self._client.ping()
            return True
        except Exception:
            return False

    def ensure_running(
        self,
        name: str,
        image: str,
        port: int,
        env: Dict[str, str],
    ) -> None:
        """Start `image` as container `memwright-{name}` on `port` if not already running."""
        if not self._ping():
            raise RuntimeError("docker daemon not reachable")

        container_name = f"memwright-{name}"
        try:
            container = self._client.containers.get(container_name)
            if container.status != "running":
                container.start()
            return
        except _docker.errors.NotFound:
            pass

        self._client.containers.run(
            image,
            name=container_name,
            detach=True,
            ports={f"{port}/tcp": port},
            environment=env,
            restart_policy={"Name": "unless-stopped"},
        )
        logger.info("Started container %s", container_name)
```

- [ ] **Step 4: Run tests**

Run: `poetry run pip install docker && poetry run pytest tests/test_docker_manager.py -v`
Expected: 2 PASS (or SKIP if docker SDK can't install in CI).

- [ ] **Step 5: Update README**

Add to install docs:

```markdown
### Optional: Docker auto-start
Install with `pipx install "memwright[arangodb,docker]"` to enable auto-starting a local Docker container when `mode = "local"` and `docker = true` are set in the backend config.
```

- [ ] **Step 6: Commit**

```bash
git add agent_memory/infra/ tests/test_docker_manager.py README.md
git commit -m "feat(infra): restore optional docker auto-start under [docker] extra"
```

---

## Task 6.2: Final integration smoke test

**Files:**
- Test: `tests/test_e2e_smoke.py`

- [ ] **Step 1: Write the smoke test**

```python
# tests/test_e2e_smoke.py
"""End-to-end smoke test: init → add → recall → stats."""
import subprocess
import sys
from pathlib import Path


def test_full_flow_sqlite(tmp_path: Path):
    py = sys.executable

    subprocess.run([py, "-m", "agent_memory.cli", "init", str(tmp_path),
                    "--backend", "sqlite", "--non-interactive"], check=True)
    assert (tmp_path / "config.toml").exists()

    subprocess.run([py, "-m", "agent_memory.cli", "config", "validate", str(tmp_path)],
                   check=True)

    subprocess.run([py, "-m", "agent_memory.cli", "add", str(tmp_path),
                    "--content", "ArangoDB is configured for local mode"], check=True)

    result = subprocess.run([py, "-m", "agent_memory.cli", "recall", str(tmp_path),
                             "--query", "arango"], capture_output=True, text=True, check=True)
    assert "ArangoDB" in result.stdout

    result = subprocess.run([py, "-m", "agent_memory.cli", "stats", str(tmp_path)],
                            capture_output=True, text=True, check=True)
    assert "Total memories: 1" in result.stdout
```

- [ ] **Step 2: Run the smoke test**

Run: `poetry run pytest tests/test_e2e_smoke.py -v`
Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e_smoke.py
git commit -m "test: end-to-end smoke test for init → add → recall → stats"
```

---

# Wrap-Up

## Final verification across all slices

- [ ] Run the full suite

```bash
poetry run pytest tests/ -v --tb=short
```

- [ ] Verify version sync

```bash
poetry run python -c "import agent_memory; print(agent_memory.__version__)"
poetry run python -c "from importlib.metadata import version; print(version('memwright'))"
```
Both must print the same value.

- [ ] Smoke install

```bash
poetry build
pipx install --force "dist/memwright-*.whl[mcp,arangodb]"
memwright init /tmp/memwright-faang --backend arangodb --mode local --port 8529 --non-interactive
memwright config show /tmp/memwright-faang
memwright config validate /tmp/memwright-faang
```

- [ ] Final commit

```bash
git commit --allow-empty -m "chore: complete FAANG-tier config + packaging overhaul"
```

## Definition of Done

- [ ] All 6 slices merged with green CI
- [ ] `__version__` and `pyproject.toml` and installed metadata all match
- [ ] `pipx install "memwright[mcp]"` is the documented one-liner
- [ ] `memwright init /path` creates a working store with no further edits
- [ ] `config.toml` is preferred; `config.json` still loads for backward compat
- [ ] Third party can ship `memwright-redis` and `backends = ["redis"]` works after `pip install memwright-redis`
- [ ] `mode: local` no longer crashes when `docker` extra not installed
- [ ] README quickstart works on a fresh machine in ≤3 commands
