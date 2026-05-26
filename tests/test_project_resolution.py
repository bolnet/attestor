# SPDX-License-Identifier: MIT
"""Unit tests for working-directory -> project identity resolution."""

from pathlib import Path

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
    a = tmp_path / "a"
    a.mkdir()
    b = tmp_path / "b"
    b.mkdir()
    assert resolve_project_root(a) != resolve_project_root(b)


def test_external_id_is_prefixed_bounded_and_deterministic():
    root = "/Users/x/code/proj"
    eid = project_external_id(root)
    assert eid.startswith("cc-project:")
    # Must fit users.external_id VARCHAR(256) regardless of path depth.
    assert len(eid) <= 80
    assert eid == project_external_id(root)  # deterministic
    assert project_external_id("/other/proj") != eid  # collision-free


def test_namespace_is_prefixed_and_deterministic():
    root = "/Users/x/code/proj"
    ns = project_namespace(root)
    assert ns.startswith("project:")
    assert ns == project_namespace(root)
    assert project_namespace("/other/proj") != ns


def test_external_id_unaffected_by_path_length():
    """A pathologically deep path still yields a bounded external_id."""
    deep = "/" + "/".join(f"segment-{i}" for i in range(50))
    assert len(project_external_id(deep)) <= 80


def test_symlink_resolves_to_canonical_tenant(tmp_path: Path):
    """A symlink and its target resolve to the same project identity."""
    real = tmp_path / "real"
    (real / ".git").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(real)
    assert resolve_project_root(link) == resolve_project_root(real)
