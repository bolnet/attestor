"""Copy a legacy ``~/.memwright`` store to ``~/.attestor``.

Public surface:

* ``migrate_store(source, dest, *, dry_run, force, verify)`` — library call.
* ``MigrationError`` — raised on refuse-to-overwrite, missing source, etc.
* ``MigrationReport`` — what happened (file count, bytes, verified).

Semantics:
    * Source is left intact (copy, not move); a breadcrumb
      ``MIGRATED_TO_ATTESTOR.txt`` is written inside the source.
    * Destination is written atomically through a staging dir that's
      ``os.replace``-d into place; partial writes can't be observed.
    * Every copied file is verified byte-equal before the staging dir
      is promoted.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class MigrationError(RuntimeError):
    """Raised when a migration cannot proceed safely."""


@dataclass(frozen=True)
class MigrationReport:
    source: Path
    dest: Path
    file_count: int
    copied_bytes: int
    verified: bool
    dry_run: bool


_BREADCRUMB_NAME = "MIGRATED_TO_ATTESTOR.txt"


def _iter_files(root: Path):
    for path in root.rglob("*"):
        if path.is_file() and path.name != _BREADCRUMB_NAME:
            yield path


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _plan(source: Path) -> tuple[int, int]:
    count = 0
    total = 0
    for f in _iter_files(source):
        count += 1
        total += f.stat().st_size
    return count, total


def _verify_identical(source: Path, dest: Path) -> None:
    for src_file in _iter_files(source):
        rel = src_file.relative_to(source)
        dst_file = dest / rel
        if not dst_file.is_file():
            raise MigrationError(f"post-copy verify: missing {rel}")
        if _sha256(src_file) != _sha256(dst_file):
            raise MigrationError(f"post-copy verify: bytes differ at {rel}")


def _write_breadcrumb(source: Path, dest: Path) -> None:
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    (source / _BREADCRUMB_NAME).write_text(
        f"Migrated to: {dest}\n"
        f"At: {stamp}\n"
        "The legacy store at this path still works, but Attestor now "
        "prefers the new location.\n",
        encoding="utf-8",
    )


def migrate_store(
    source: Path | str,
    dest: Path | str,
    *,
    dry_run: bool = False,
    force: bool = False,
) -> MigrationReport:
    """Copy a legacy store at ``source`` to ``dest``.

    Raises ``MigrationError`` if ``source`` is missing, if ``source`` and
    ``dest`` resolve to the same path, or if ``dest`` already exists and
    ``force`` is False.
    """
    source = Path(source).expanduser()
    dest = Path(dest).expanduser()

    if not source.exists() or not source.is_dir():
        raise MigrationError(f"source store not found: {source}")
    if source.resolve() == dest.resolve():
        raise MigrationError("source and dest are the same directory")

    count, total = _plan(source)

    if dry_run:
        return MigrationReport(
            source=source,
            dest=dest,
            file_count=count,
            copied_bytes=total,
            verified=False,
            dry_run=True,
        )

    dest_preexisted = dest.exists()
    if dest_preexisted and not force:
        raise MigrationError(
            f"destination already exists: {dest}. Use --force to overwrite."
        )

    dest.parent.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{dest.name}.migrate-", dir=dest.parent
        )
    )
    try:
        # copytree into an empty subdir of the staging dir so the rename
        # target is atomic even though copytree won't merge into a
        # non-empty dir.
        staged = staging / "store"
        shutil.copytree(source, staged, ignore=_ignore_breadcrumb)
        _verify_identical(source, staged)

        if dest_preexisted:
            shutil.rmtree(dest)
        os.replace(staged, dest)
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    _write_breadcrumb(source, dest)

    return MigrationReport(
        source=source,
        dest=dest,
        file_count=count,
        copied_bytes=total,
        verified=True,
        dry_run=False,
    )


def _ignore_breadcrumb(directory: str, names: list[str]) -> list[str]:
    return [n for n in names if n == _BREADCRUMB_NAME]
