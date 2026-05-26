# Attestor Native Claude Code Plugin + Per-Project Isolation — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Attestor a marketplace-installable Claude Code plugin whose memory is hard-isolated per project (current working directory), with zero cross-project bleed across Postgres, Pinecone, and Neo4j.

**Architecture:** Derive a stable project root from `cwd` (git root if present, else the cwd). Map that root to a dedicated RLS tenant via `AgentMemory.ensure_user(external_id="cc-project:<root>")`, and pass that `user_id` (plus `namespace="project:<root>"` as defense-in-depth) into every hook recall/add and into the MCP server's default tool tenant. Postgres Row-Level Security (keyed to `attestor.current_user_id`, already implemented) provides DB-enforced isolation; Pinecone namespaces and Neo4j namespace-scoped nodes (already implemented) isolate the vector and graph lanes. Then package the MCP server + hooks + skill + install command as a Claude Code plugin.

**Tech Stack:** Python 3.12, Poetry, pytest. Existing modules: `attestor/core/agent_memory.py` (`AgentMemory`), `attestor/core/identity_service.py` (`_IdentityMixin`), `attestor/store/postgres_backend.py` (RLS via `_set_rls_user`), `attestor/hooks/*`, `attestor/mcp/server.py`. Claude Code plugin manifest format (`.claude-plugin/plugin.json`).

---

## File Structure

**Create:**
- `attestor/_project.py` — pure functions: `resolve_project_root(cwd)`, `project_external_id(root)`, `project_namespace(root)`.
- `attestor/hooks/_tenant.py` — `resolve_tenant(mem, cwd) -> (user_id, namespace)`; the one place hooks turn a cwd into a tenant.
- `.claude-plugin/plugin.json` — plugin manifest (MCP server + hooks + skill + command).
- `.claude-plugin/marketplace.json` — single-plugin marketplace manifest.
- `tests/test_project_resolution.py` — unit tests for `_project.py`.
- `tests/test_hook_tenant.py` — unit tests for `_tenant.py` + hook wiring (fake `AgentMemory`).
- `tests/test_tenant_isolation_live.py` — live, env-gated cross-store no-bleed test.

**Modify:**
- `attestor/hooks/session_start.py` — derive tenant from `payload["cwd"]`; pass `user_id` + `namespace` to `recall()`.
- `attestor/hooks/post_tool_use.py` — pass `user_id` + `namespace` + `scope="project"` to `add()`.
- `attestor/hooks/stop.py` — same tenant wiring on the summary write.
- `attestor/mcp/server.py` — resolve a default tenant from the server launch `cwd`; use it as default `user_id`/`namespace` for every tool.
- `commands/install-attestor.md` — note the plugin auto-wires MCP + hooks; trim those steps.
- `CLAUDE.md` — fix the stale "Neo4j namespace isolation not enforced" claim.

---

## Task 1: Project root resolution (pure functions)

**Files:**
- Create: `attestor/_project.py`
- Test: `tests/test_project_resolution.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_project_resolution.py
# SPDX-License-Identifier: MIT
from pathlib import Path

import pytest

from attestor._project import (
    project_external_id,
    project_namespace,
    resolve_project_root,
)


def test_resolve_uses_git_root_when_present(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sub = repo / "src" / "pkg"
    sub.mkdir(parents=True)
    assert resolve_project_root(sub) == str(repo.resolve())


def test_resolve_falls_back_to_cwd_without_git(tmp_path: Path):
    plain = tmp_path / "plain"
    plain.mkdir()
    assert resolve_project_root(plain) == str(plain.resolve())


def test_resolve_handles_file_path_by_using_parent(tmp_path: Path):
    f = tmp_path / "a.txt"
    f.write_text("x")
    assert resolve_project_root(f) == str(tmp_path.resolve())


def test_unrelated_dirs_resolve_differently(tmp_path: Path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    assert resolve_project_root(a) != resolve_project_root(b)


def test_external_id_and_namespace_are_prefixed():
    root = "/Users/x/code/proj"
    assert project_external_id(root) == "cc-project:/Users/x/code/proj"
    assert project_namespace(root) == "project:/Users/x/code/proj"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_project_resolution.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'attestor._project'`

- [ ] **Step 3: Write the implementation**

```python
# attestor/_project.py
# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Resolve a Claude Code working directory to a stable project identity.

A *project* is the unit of memory isolation: the git repository root if the
working directory is inside one, otherwise the working directory itself. Two
unrelated directories always resolve to different identities, so memory never
bleeds across projects.
"""

from __future__ import annotations

from pathlib import Path

# external_id of the RLS tenant that owns a project's memory. Distinct from the
# SOLO singleton ("local") so per-project tenants never collide with it.
_PROJECT_EXTERNAL_PREFIX = "cc-project:"

# Defense-in-depth namespace stamped on Pinecone vectors and Neo4j nodes.
_PROJECT_NAMESPACE_PREFIX = "project:"


def resolve_project_root(cwd: str | Path) -> str:
    """Return the project root for ``cwd``.

    Git repository root if ``cwd`` is inside one (so subdirectories of a repo
    share memory); otherwise the resolved absolute ``cwd``. If ``cwd`` is a
    file, its parent directory is used.
    """
    start = Path(cwd).expanduser().resolve()
    if start.is_file():
        start = start.parent
    for candidate in (start, *start.parents):
        if (candidate / ".git").exists():
            return str(candidate)
    return str(start)


def project_external_id(root: str) -> str:
    """RLS tenant external_id for a project root."""
    return f"{_PROJECT_EXTERNAL_PREFIX}{root}"


def project_namespace(root: str) -> str:
    """Vector/graph namespace for a project root."""
    return f"{_PROJECT_NAMESPACE_PREFIX}{root}"
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_project_resolution.py -q`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add attestor/_project.py tests/test_project_resolution.py
git commit -m "feat(cc): resolve a working directory to a stable project identity"
```

---

## Task 2: Tenant resolution helper (cwd → RLS user + namespace)

**Files:**
- Create: `attestor/hooks/_tenant.py`
- Test: `tests/test_hook_tenant.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_hook_tenant.py
# SPDX-License-Identifier: MIT
from pathlib import Path
from types import SimpleNamespace

from attestor.hooks._tenant import resolve_tenant


class _FakeMem:
    """Records ensure_user calls and returns a deterministic user id."""

    def __init__(self):
        self.ensured: list[str] = []

    def ensure_user(self, external_id: str, **_):
        self.ensured.append(external_id)
        return SimpleNamespace(id=f"uid::{external_id}")


def test_resolve_tenant_keys_user_to_project_root(tmp_path: Path):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    mem = _FakeMem()

    user_id, namespace = resolve_tenant(mem, str(repo))

    assert user_id == f"uid::cc-project:{repo.resolve()}"
    assert namespace == f"project:{repo.resolve()}"
    assert mem.ensured == [f"cc-project:{repo.resolve()}"]


def test_resolve_tenant_distinct_dirs_distinct_users(tmp_path: Path):
    a = tmp_path / "a"; a.mkdir()
    b = tmp_path / "b"; b.mkdir()
    mem = _FakeMem()

    ua, _ = resolve_tenant(mem, str(a))
    ub, _ = resolve_tenant(mem, str(b))

    assert ua != ub
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_hook_tenant.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'attestor.hooks._tenant'`

- [ ] **Step 3: Write the implementation**

```python
# attestor/hooks/_tenant.py
# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Map a Claude Code working directory to the RLS tenant that owns its memory.

This is the single chokepoint that enforces per-project isolation in the hook
path. Every hook recall/add routes its ``cwd`` through here so the underlying
Postgres Row-Level Security policy (keyed to ``attestor.current_user_id``)
filters to exactly one project.
"""

from __future__ import annotations

from typing import Any

from attestor._project import (
    project_external_id,
    project_namespace,
    resolve_project_root,
)


def resolve_tenant(mem: Any, cwd: str) -> tuple[str, str]:
    """Return ``(user_id, namespace)`` for the project containing ``cwd``.

    Idempotently ensures an RLS user keyed to the project root. ``ensure_user``
    is a no-op lookup after first creation, so this is cheap to call per hook.
    """
    root = resolve_project_root(cwd)
    user = mem.ensure_user(external_id=project_external_id(root))
    return user.id, project_namespace(root)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_hook_tenant.py -q`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add attestor/hooks/_tenant.py tests/test_hook_tenant.py
git commit -m "feat(cc): map a working directory to an RLS tenant for hooks"
```

---

## Task 3: Wire SessionStart recall to the project tenant

**Files:**
- Modify: `attestor/hooks/session_start.py`
- Test: `tests/test_hook_tenant.py` (extend)

- [ ] **Step 1: Write the failing test (append to tests/test_hook_tenant.py)**

```python
def test_session_start_recall_is_tenant_scoped(monkeypatch, tmp_path):
    """session_start must pass the project user_id + namespace to recall()."""
    import attestor.hooks.session_start as ss

    captured = {}

    class _Mem:
        def ensure_user(self, external_id, **_):
            from types import SimpleNamespace
            return SimpleNamespace(id=f"uid::{external_id}")

        def recall(self, query, budget=None, user_id=None, namespace=None, **_):
            captured["user_id"] = user_id
            captured["namespace"] = namespace
            return []

        def pagerank(self):
            return {}

        def close(self):
            pass

    monkeypatch.setattr(ss, "AgentMemory", lambda *a, **k: _Mem(), raising=False)
    monkeypatch.setattr(ss, "resolve_store_path", lambda *a, **k: str(tmp_path), raising=False)

    out = ss.handle({"cwd": str(tmp_path)})

    assert captured["user_id"] == f"uid::cc-project:{tmp_path.resolve()}"
    assert captured["namespace"] == f"project:{tmp_path.resolve()}"
    assert "additionalContext" in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_hook_tenant.py::test_session_start_recall_is_tenant_scoped -q`
Expected: FAIL — `AgentMemory`/`resolve_store_path` are imported inside `_do_recall`, not module-level, so the monkeypatch + scoping don't apply yet. (This drives moving the imports to module scope and adding tenant wiring.)

- [ ] **Step 3: Edit `attestor/hooks/session_start.py`**

Replace the `_do_recall` function and `handle` so the tenant is resolved and module-level names exist for patching. New `_do_recall` signature takes `cwd`:

```python
# top of file, add to imports:
from attestor._paths import resolve_store_path
from attestor.core import AgentMemory
from attestor.hooks._tenant import resolve_tenant
from attestor.retrieval.scorer import pagerank_boost


def _do_recall(cwd: str) -> dict[str, Any]:
    """The actual recall + pagerank pass — runs inside the timeout shim."""
    store_path = resolve_store_path()
    mem = AgentMemory(store_path)
    try:
        user_id, namespace = resolve_tenant(mem, cwd)
        results = mem.recall(
            _SESSION_QUERY,
            budget=_SESSION_BUDGET,
            user_id=user_id,
            namespace=namespace,
        )

        pr_scores = mem.pagerank()
        if pr_scores and results:
            results = pagerank_boost(results, pr_scores, weight=0.3)
            results.sort(key=lambda r: r.score, reverse=True)

        if not results:
            return _EMPTY_RESPONSE

        lines = ["Relevant memories:"]
        for r in results:
            prefix = f"[{r.match_source}:{r.score:.2f}]"
            lines.append(f"- {prefix} {r.memory.content}")
        return {"additionalContext": "\n".join(lines)}
    finally:
        mem.close()
```

And in `handle`, pass `cwd` into the worker:

```python
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(_do_recall, cwd)
            try:
                return fut.result(timeout=_HOOK_TIMEOUT_S)
            except concurrent.futures.TimeoutError:
                return _EMPTY_RESPONSE
```

Remove the now-duplicated lazy imports inside `_do_recall`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_hook_tenant.py::test_session_start_recall_is_tenant_scoped -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add attestor/hooks/session_start.py tests/test_hook_tenant.py
git commit -m "feat(cc): scope SessionStart recall to the project tenant"
```

---

## Task 4: Wire PostToolUse add to the project tenant

**Files:**
- Modify: `attestor/hooks/post_tool_use.py`
- Test: `tests/test_hook_tenant.py` (extend)

- [ ] **Step 1: Write the failing test (append)**

```python
def test_post_tool_use_add_is_tenant_scoped(monkeypatch, tmp_path):
    import attestor.hooks.post_tool_use as ptu

    captured = {}

    class _Mem:
        def ensure_user(self, external_id, **_):
            from types import SimpleNamespace
            return SimpleNamespace(id=f"uid::{external_id}")

        def add(self, content, tags=None, category="general",
                user_id=None, namespace=None, scope="user", **_):
            captured.update(
                user_id=user_id, namespace=namespace, scope=scope,
                content=content,
            )

        def close(self):
            pass

    monkeypatch.setattr(ptu, "AgentMemory", lambda *a, **k: _Mem(), raising=False)
    monkeypatch.setattr(ptu, "resolve_store_path", lambda *a, **k: str(tmp_path), raising=False)

    ptu.handle({
        "cwd": str(tmp_path),
        "tool_name": "Write",
        "tool_input": {"file_path": "foo.py"},
    })

    assert captured["user_id"] == f"uid::cc-project:{tmp_path.resolve()}"
    assert captured["namespace"] == f"project:{tmp_path.resolve()}"
    assert captured["scope"] == "project"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_hook_tenant.py::test_post_tool_use_add_is_tenant_scoped -q`
Expected: FAIL — imports are lazy inside `handle`; `add()` is called without `user_id`/`namespace`/`scope`.

- [ ] **Step 3: Edit `attestor/hooks/post_tool_use.py`**

Move the lazy imports to module scope and add tenant wiring:

```python
# top of file, add:
from attestor._paths import resolve_store_path
from attestor.core import AgentMemory
from attestor.hooks._tenant import resolve_tenant
```

Replace the `if content:` block body:

```python
        if content:
            store_path = resolve_store_path()

            def _do_add() -> None:
                mem = AgentMemory(store_path)
                try:
                    user_id, namespace = resolve_tenant(mem, cwd)
                    mem.add(
                        content,
                        tags=tags,
                        category=category,
                        user_id=user_id,
                        namespace=namespace,
                        scope="project",
                    )
                finally:
                    mem.close()

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_do_add)
                try:
                    fut.result(timeout=_HOOK_TIMEOUT_S)
                except concurrent.futures.TimeoutError:
                    pass
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_hook_tenant.py::test_post_tool_use_add_is_tenant_scoped -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add attestor/hooks/post_tool_use.py tests/test_hook_tenant.py
git commit -m "feat(cc): scope PostToolUse capture to the project tenant"
```

---

## Task 5: Wire Stop summary to the project tenant

**Files:**
- Modify: `attestor/hooks/stop.py`
- Test: `tests/test_hook_tenant.py` (extend)

- [ ] **Step 1: Read `attestor/hooks/stop.py`** to find where it constructs `AgentMemory` and calls `add()` for the `session-summary` memory. It mirrors `post_tool_use` (lazy import + `_do_*` worker).

- [ ] **Step 2: Write the failing test (append)**

```python
def test_stop_summary_is_tenant_scoped(monkeypatch, tmp_path):
    import attestor.hooks.stop as st

    captured = {}

    class _Mem:
        def ensure_user(self, external_id, **_):
            from types import SimpleNamespace
            return SimpleNamespace(id=f"uid::{external_id}")

        def search(self, *a, **k):
            return []

        def recall(self, *a, **k):
            return []

        def add(self, content, tags=None, category="general",
                user_id=None, namespace=None, scope="user", **_):
            captured.update(user_id=user_id, namespace=namespace, scope=scope)

        def close(self):
            pass

    monkeypatch.setattr(st, "AgentMemory", lambda *a, **k: _Mem(), raising=False)
    monkeypatch.setattr(st, "resolve_store_path", lambda *a, **k: str(tmp_path), raising=False)

    st.handle({"cwd": str(tmp_path)})

    # If there were observations to summarize, the write is tenant-scoped.
    if captured:
        assert captured["user_id"] == f"uid::cc-project:{tmp_path.resolve()}"
        assert captured["namespace"] == f"project:{tmp_path.resolve()}"
        assert captured["scope"] == "project"
```

- [ ] **Step 3: Run test to verify it fails or trivially passes**

Run: `.venv/bin/pytest tests/test_hook_tenant.py::test_stop_summary_is_tenant_scoped -q`
Expected: FAIL if `stop.handle` queries observations and writes without scoping. (If `stop` reads observations via `search`/`recall`, those reads ALSO need `user_id` scoping — otherwise the summary is computed from other projects' data. Scope both the read and the write.)

- [ ] **Step 4: Edit `attestor/hooks/stop.py`**

Move imports to module scope (`resolve_store_path`, `AgentMemory`, `resolve_tenant`). In the worker, resolve the tenant once and pass `user_id`/`namespace` to BOTH the observation read (`search`/`recall`) and the summary `add(..., scope="project")`. Mirror the exact wiring shape from Task 4.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_hook_tenant.py -q`
Expected: PASS (all hook tenant tests)

- [ ] **Step 6: Commit**

```bash
git add attestor/hooks/stop.py tests/test_hook_tenant.py
git commit -m "feat(cc): scope Stop summary read+write to the project tenant"
```

---

## Task 6: Default the MCP server's tools to the launch-cwd tenant

**Files:**
- Modify: `attestor/mcp/server.py`
- Test: `tests/test_mcp_tenant_default.py`

- [ ] **Step 1: Read `attestor/mcp/server.py`** around the `AgentMemory` construction and the tool handlers (lines ~440-525 for `memory_add`/`memory_recall`/`memory_search`/`memory_timeline`). Confirm where `mem` is built and where `namespace`/`user_id` are read from tool `args`.

- [ ] **Step 2: Write the failing test**

```python
# tests/test_mcp_tenant_default.py
# SPDX-License-Identifier: MIT
import os
from pathlib import Path

from attestor.mcp import server as mcp_server


def test_default_tenant_derives_from_cwd(tmp_path: Path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    monkeypatch.chdir(repo)

    user_id, namespace = mcp_server._default_tenant_for_cwd()

    assert namespace == f"project:{repo.resolve()}"
    assert user_id == f"cc-project:{repo.resolve()}"  # external_id form pre-ensure
```

Note: `_default_tenant_for_cwd()` returns the external_id form (no DB call); the
server calls `mem.ensure_user(...)` lazily on first tool use to get the real
`user_id`. This keeps the unit test infra-free.

- [ ] **Step 3: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_mcp_tenant_default.py -q`
Expected: FAIL — `_default_tenant_for_cwd` not defined.

- [ ] **Step 4: Edit `attestor/mcp/server.py`**

Add a module-level helper and a cached per-server tenant resolver:

```python
import os
from attestor._project import (
    project_external_id, project_namespace, resolve_project_root,
)


def _default_tenant_for_cwd() -> tuple[str, str]:
    """(external_id, namespace) for the server's launch directory.

    Claude Code launches the MCP server with cwd = the workspace root, so the
    project the user is working in is os.getcwd() at server start.
    """
    root = resolve_project_root(os.getcwd())
    return project_external_id(root), project_namespace(root)
```

In the server's tool-dispatch path, resolve and memoize the real `user_id`
once (lazily, on first tool call):

```python
        # one-time per server process
        if self._project_user_id is None:
            ext_id, self._project_namespace = _default_tenant_for_cwd()
            self._project_user_id = self._mem.ensure_user(external_id=ext_id).id
```

Then change the tool handlers so the tenant defaults flow through:
- `memory_add`: `user_id=self._project_user_id`, `scope="project"`, and
  `namespace=args.get("namespace") or self._project_namespace` (was
  `args.get("namespace", "default")`).
- `memory_recall` / `memory_search` / `memory_timeline`: pass
  `user_id=self._project_user_id` and
  `namespace=args.get("namespace") or self._project_namespace`.

Initialize `self._project_user_id = None` and `self._project_namespace = None`
where the server stores `self._mem`.

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv/bin/pytest tests/test_mcp_tenant_default.py -q`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add attestor/mcp/server.py tests/test_mcp_tenant_default.py
git commit -m "feat(cc): default MCP tools to the launch-directory project tenant"
```

---

## Task 7: Live cross-store isolation test (env-gated)

**Files:**
- Create: `tests/test_tenant_isolation_live.py`

This is the proof that there is no cross-project bleed across Postgres + Pinecone + Neo4j. It is **env-gated** (`ATTESTOR_LIVE=1`) and uses the live canonical stack. Do NOT run it during a benchmark (it writes to the live DB); see the project memory `feedback_no_pytest_during_live_bench`.

- [ ] **Step 1: Write the test**

```python
# tests/test_tenant_isolation_live.py
# SPDX-License-Identifier: MIT
import os
from pathlib import Path

import pytest

from attestor.core import AgentMemory
from attestor.hooks._tenant import resolve_tenant

pytestmark = pytest.mark.skipif(
    os.environ.get("ATTESTOR_LIVE") != "1",
    reason="requires live Postgres+Pinecone+Neo4j (set ATTESTOR_LIVE=1)",
)


def test_two_projects_do_not_share_memory(tmp_path: Path):
    store = str(tmp_path / "store")
    mem = AgentMemory(store)
    try:
        proj_a = tmp_path / "alpha"; (proj_a / ".git").mkdir(parents=True)
        proj_b = tmp_path / "beta"; (proj_b / ".git").mkdir(parents=True)

        ua, ns_a = resolve_tenant(mem, str(proj_a))
        ub, ns_b = resolve_tenant(mem, str(proj_b))
        assert ua != ub

        mem.add(
            "Alpha uses Rust for the parser",
            tags=["decision"], category="project",
            user_id=ua, namespace=ns_a, scope="project",
        )

        # Recall from project B must NOT see project A's memory.
        hits_b = mem.recall(
            "what language for the parser", user_id=ub, namespace=ns_b,
        )
        assert all("Alpha uses Rust" not in r.memory.content for r in hits_b)

        # Recall from project A DOES see it.
        hits_a = mem.recall(
            "what language for the parser", user_id=ua, namespace=ns_a,
        )
        assert any("Alpha uses Rust" in r.memory.content for r in hits_a)
    finally:
        mem.close()
```

- [ ] **Step 2: Run the test (only if the live stack is up)**

Run: `ATTESTOR_LIVE=1 .venv/bin/pytest tests/test_tenant_isolation_live.py -q`
Expected: PASS. If the stack is not running, the test SKIPS (that is acceptable for CI without secrets).

- [ ] **Step 3: Commit**

```bash
git add tests/test_tenant_isolation_live.py
git commit -m "test(cc): live cross-store per-project isolation proof (env-gated)"
```

---

## Task 8: Claude Code plugin manifest

**Files:**
- Create: `.claude-plugin/plugin.json`

- [ ] **Step 1: Verify the hook command names** that the CLI accepts: `grep -n "session-start\|post-tool-use\|stop" attestor/cli/commands/server.py` — confirm `attestor hook session-start|post-tool-use|stop` and `attestor mcp` exist (they do per the inventory).

- [ ] **Step 2: Write the manifest**

```json
{
  "name": "attestor",
  "version": "4.0.0",
  "description": "The memory layer for agent teams. Deterministic retrieval, per-project isolation, zero LLM in the critical path.",
  "author": { "name": "Surendra Singh", "url": "https://github.com/bolnet/attestor" },
  "license": "MIT",
  "mcpServers": {
    "attestor": {
      "command": "attestor",
      "args": ["mcp"]
    }
  },
  "hooks": {
    "SessionStart": [
      { "matcher": "*", "hooks": [{ "type": "command", "command": "attestor hook session-start" }] }
    ],
    "PostToolUse": [
      { "matcher": "Write|Edit|Bash", "hooks": [{ "type": "command", "command": "attestor hook post-tool-use" }] }
    ],
    "Stop": [
      { "matcher": "*", "hooks": [{ "type": "command", "command": "attestor hook stop" }] }
    ]
  },
  "skills": ["skills/attestor-memory"],
  "commands": ["commands/install-attestor.md"]
}
```

Note: `attestor mcp` is invoked with NO `--path`; the server resolves its store
via `resolve_store_path()` and derives the project tenant from its launch cwd
(Task 6). The hooks derive their tenant from each event's `cwd` (Tasks 3-5).

- [ ] **Step 3: Validate JSON**

Run: `python -c "import json,sys; json.load(open('.claude-plugin/plugin.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 4: Commit**

```bash
git add .claude-plugin/plugin.json
git commit -m "feat(cc): add Claude Code plugin manifest"
```

---

## Task 9: Marketplace manifest

**Files:**
- Create: `.claude-plugin/marketplace.json`

- [ ] **Step 1: Write the manifest**

```json
{
  "name": "attestor",
  "owner": { "name": "bolnet", "url": "https://github.com/bolnet" },
  "plugins": [
    {
      "name": "attestor",
      "source": ".",
      "description": "Self-hosted memory layer for Claude Code with hard per-project isolation."
    }
  ]
}
```

- [ ] **Step 2: Validate JSON**

Run: `python -c "import json; json.load(open('.claude-plugin/marketplace.json')); print('ok')"`
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add .claude-plugin/marketplace.json
git commit -m "feat(cc): add single-plugin marketplace manifest"
```

---

## Task 10: Trim the install command + fix the stale CLAUDE.md claim

**Files:**
- Modify: `commands/install-attestor.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Edit `commands/install-attestor.md`**

Add a note at the top: when installed as a plugin (`/plugin install attestor`),
the MCP server and hooks are auto-wired by `.claude-plugin/plugin.json`, so the
wizard only needs to (a) install the `attestor` package, (b) collect backend
connection details (Postgres + Pinecone + Neo4j; local Docker or cloud), (c)
collect the embedding provider, and (d) run `attestor doctor`. The
`settings.json` MCP + hook writing steps are skipped in plugin mode (they only
apply to manual, non-plugin installs).

- [ ] **Step 2: Edit `CLAUDE.md`**

Replace the stale sentence in the multi-agent primitives paragraph:

> "Namespace isolation is row-level on Postgres but **not yet enforced on Neo4j** (graph entity nodes are global across namespaces)."

with:

> "Namespace isolation is enforced across all three roles: row-level + RLS on Postgres, per-namespace on Pinecone, and a `namespace` property with a composite `(key, namespace)` constraint and namespace-scoped BFS on Neo4j. The Claude Code plugin keys each project (git root, else cwd) to its own RLS tenant (`external_id=cc-project:<root>`) so memory never bleeds across projects."

- [ ] **Step 3: Commit**

```bash
git add commands/install-attestor.md CLAUDE.md
git commit -m "docs(cc): plugin-mode install notes + correct Neo4j isolation claim"
```

---

## Task 11: Full unit suite + lint gate

- [ ] **Step 1: Run the new unit tests together**

Run: `.venv/bin/pytest tests/test_project_resolution.py tests/test_hook_tenant.py tests/test_mcp_tenant_default.py -q`
Expected: PASS (all)

- [ ] **Step 2: Run the broader hook/MCP unit tests to catch regressions**

Run: `.venv/bin/pytest tests/ -q -k "hook or mcp or identity or tenant" `
Expected: PASS (no regressions). Do NOT run the full live suite here.

- [ ] **Step 3: Lint the changed files**

Run: `.venv/bin/ruff check attestor/_project.py attestor/hooks/_tenant.py attestor/hooks/session_start.py attestor/hooks/post_tool_use.py attestor/hooks/stop.py attestor/mcp/server.py`
Expected: no errors.

- [ ] **Step 4: Commit any lint fixes**

```bash
git add -A && git commit -m "chore(cc): lint fixes for project-isolation wiring"
```

---

## Self-Review (completed during planning)

- **Spec coverage:** Part 1 (isolation) → Tasks 1-7; Part 2 (plugin packaging) → Tasks 8-10; testing requirement → Tasks 1-7, 11. Stale-doc fix → Task 10. All spec sections mapped.
- **Type consistency:** `resolve_project_root` / `project_external_id` / `project_namespace` (Task 1) used identically in Tasks 2 & 6. `resolve_tenant(mem, cwd) -> (user_id, namespace)` (Task 2) used identically in Tasks 3-5. `add(..., user_id, namespace, scope)` and `recall(..., user_id, namespace)` match the real signatures in `agent_memory.py:716` and `:1199`.
- **Placeholder scan:** Tasks 5, 6, 8 contain a "Read X first" step (the existing files' internal shape must be confirmed before editing); the concrete edit is fully specified after it. No TODO/TBD left.

## Open risk carried from the spec

- **MCP launch cwd:** Task 6 assumes Claude Code starts the MCP server with `cwd = workspace`. If that proves false during integration, switch from a once-per-process tenant to per-tool-call resolution using a `cwd` the client provides. The hook path (Tasks 3-5) is unaffected — it always gets `cwd` per event.
