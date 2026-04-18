"""Phase 3: ``attestor migrate`` subcommand + library function.

The migration copies a legacy ``~/.memwright`` store to ``~/.attestor``,
verifies byte-equal on the important files, leaves a breadcrumb in the
source, and supports ``--dry-run``, ``--force``, ``--source``, ``--dest``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from attestor.migrate import MigrationError, migrate_store


# ----------------------------------------------------------------------------
# Fixtures
# ----------------------------------------------------------------------------


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def legacy_store(tmp_path):
    """A minimal ``~/.memwright``-shaped tree with the three core artifacts."""
    src = tmp_path / ".memwright"
    src.mkdir()
    (src / "memory.db").write_bytes(b"SQLITE_FAKE_DOCUMENT_STORE_BYTES")
    (src / "graph.json").write_text('{"nodes": [], "edges": []}', encoding="utf-8")
    chroma = src / "chroma"
    chroma.mkdir()
    (chroma / "chroma.sqlite3").write_bytes(b"FAKE_CHROMA_DB")
    (chroma / "data_level0.bin").write_bytes(b"\x00\x01\x02\x03")
    (src / "config.json").write_text('{"store": "local"}', encoding="utf-8")
    return src


# ----------------------------------------------------------------------------
# Happy path
# ----------------------------------------------------------------------------


def test_migrate_copies_store(legacy_store, tmp_path):
    dest = tmp_path / ".attestor"

    report = migrate_store(source=legacy_store, dest=dest)

    assert dest.is_dir()
    # Every file from source now exists at dest with identical bytes.
    for src_file in legacy_store.rglob("*"):
        if not src_file.is_file():
            continue
        if src_file.name == "MIGRATED_TO_ATTESTOR.txt":
            continue  # breadcrumb lives only in source
        rel = src_file.relative_to(legacy_store)
        dst_file = dest / rel
        assert dst_file.is_file(), f"missing at dest: {rel}"
        assert _sha(src_file) == _sha(dst_file), f"bytes differ: {rel}"

    assert report.copied_bytes > 0
    assert report.verified is True
    assert report.dry_run is False


def test_migrate_leaves_source_intact_with_breadcrumb(legacy_store, tmp_path):
    dest = tmp_path / ".attestor"

    migrate_store(source=legacy_store, dest=dest)

    assert (legacy_store / "memory.db").exists()
    assert (legacy_store / "graph.json").exists()
    breadcrumb = legacy_store / "MIGRATED_TO_ATTESTOR.txt"
    assert breadcrumb.exists()
    assert str(dest) in breadcrumb.read_text()


# ----------------------------------------------------------------------------
# Safety: refuse to overwrite, --force, dry-run
# ----------------------------------------------------------------------------


def test_migrate_refuses_existing_dest(legacy_store, tmp_path):
    dest = tmp_path / ".attestor"
    dest.mkdir()
    (dest / "memory.db").write_bytes(b"PRE_EXISTING")

    with pytest.raises(MigrationError, match="already exists"):
        migrate_store(source=legacy_store, dest=dest)

    # Pre-existing bytes must be left untouched.
    assert (dest / "memory.db").read_bytes() == b"PRE_EXISTING"


def test_migrate_force_overwrites(legacy_store, tmp_path):
    dest = tmp_path / ".attestor"
    dest.mkdir()
    (dest / "memory.db").write_bytes(b"PRE_EXISTING")

    report = migrate_store(source=legacy_store, dest=dest, force=True)

    assert report.verified is True
    assert (
        (dest / "memory.db").read_bytes()
        == (legacy_store / "memory.db").read_bytes()
    )


def test_dry_run_reports_but_does_not_write(legacy_store, tmp_path):
    dest = tmp_path / ".attestor"

    report = migrate_store(source=legacy_store, dest=dest, dry_run=True)

    assert report.dry_run is True
    assert report.copied_bytes > 0  # would-have-copied accounting
    assert not dest.exists()
    assert not (legacy_store / "MIGRATED_TO_ATTESTOR.txt").exists()


# ----------------------------------------------------------------------------
# Error conditions
# ----------------------------------------------------------------------------


def test_migrate_missing_source(tmp_path):
    missing = tmp_path / "nope"
    dest = tmp_path / ".attestor"

    with pytest.raises(MigrationError, match="not found"):
        migrate_store(source=missing, dest=dest)


def test_migrate_source_equals_dest(legacy_store):
    with pytest.raises(MigrationError, match="same"):
        migrate_store(source=legacy_store, dest=legacy_store)


# ----------------------------------------------------------------------------
# Loadability after migrate — an AgentMemory pointed at dest works.
# ----------------------------------------------------------------------------


def test_migrated_store_is_loadable(tmp_path):
    """End-to-end: write a real store, migrate it, open the migrated copy."""
    from attestor import AgentMemory

    legacy = tmp_path / ".memwright"
    mem = AgentMemory(str(legacy), config={"default_token_budget": 2000})
    mem.add("Venue is the Blue Rose Cafe", tags=["venue"], category="fact")
    mem.close()

    dest = tmp_path / ".attestor"
    report = migrate_store(source=legacy, dest=dest)
    assert report.verified

    migrated = AgentMemory(str(dest), config={"default_token_budget": 2000})
    results = migrated.recall("venue")
    migrated.close()

    texts = " ".join(r.content for r in results)
    assert "Blue Rose" in texts
