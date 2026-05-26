# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Interactive and non-interactive store initialization."""
from __future__ import annotations

import getpass
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Mapping

import tomlkit


log = logging.getLogger(__name__)

SUPPORTED_BACKENDS = ("postgres",)
_CREDENTIAL_KEYS = frozenset({"auth_password", "password", "secret", "api_key", "token"})


@dataclass(frozen=True)
class InitResult:
    """Outcome of a store initialization."""

    config_path: Path
    backend: str
    verified: bool


def _build_config(backend: str, backend_options: Mapping[str, Any] | None) -> tomlkit.TOMLDocument:
    """Emit a complete, working local store config.

    Earlier versions listed ``["postgres","pinecone","neo4j"]`` but defined a
    table for only one backend and stripped the Postgres password to nothing,
    so the first connection died with "no password supplied" and the listed
    pinecone/neo4j backends had no config. This writes the full local default:

      * `backends = ["postgres", "neo4j"]` — pgvector serves the VECTOR role
        inside the v4 Postgres store (`v4 = true`), so no separate Pinecone
        backend is needed locally. (Add a `[pinecone]` table + "pinecone" to
        `backends` to use a dedicated vector store.)
      * Structured `[postgres.auth]` / `[neo4j.auth]` with **whole-value
        `$ENV_VAR` references** (resolved from the environment / `.env`).
        Inline `${VAR}` inside a URL is NOT expanded — only a whole-value
        `$VAR` in a structured auth block is — so the password MUST live here,
        never in the URL.
      * `v4 = true` + `skip_schema_init = false` so a fresh store provisions
        its own schema (bench harnesses set these the other way; an install
        must not).
    """
    opts = dict(backend_options or {})
    doc = tomlkit.document()
    doc.add(tomlkit.comment("Attestor store config -- local install (Postgres + Neo4j)."))
    doc.add(tomlkit.comment("Secrets use $ENV_VAR references (resolved from the env / ~/.attestor/.env), never plaintext."))
    doc.add(tomlkit.nl())

    doc["backends"] = ["postgres", "neo4j"]
    doc["default_token_budget"] = 10000
    doc.add(tomlkit.nl())

    # Document + vector role. v4 enables the pgvector embedding column (vector
    # lane) + content_tsv (BM25 lane). url is host-only; password via env ref.
    pg = tomlkit.table()
    pg["url"] = opts.get("url") or "postgresql://localhost:5432"
    pg["database"] = opts.get("database") or "attestor"
    pg["v4"] = True
    pg["skip_schema_init"] = False
    pg_auth = tomlkit.table()
    pg_auth["username"] = opts.get("username") or "postgres"
    pg_auth["password"] = "$PGPASSWORD"
    pg["auth"] = pg_auth
    doc["postgres"] = pg
    doc.add(tomlkit.nl())

    # Graph role.
    neo = tomlkit.table()
    neo["url"] = opts.get("neo4j_url") or "bolt://localhost:7687"
    neo["database"] = opts.get("neo4j_database") or "neo4j"
    neo_auth = tomlkit.table()
    neo_auth["username"] = opts.get("neo4j_username") or "neo4j"
    neo_auth["password"] = "$NEO4J_PASSWORD"
    neo["auth"] = neo_auth
    doc["neo4j"] = neo

    return doc


_INLINE_CRED_RE = __import__("re").compile(
    r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]*://)"
    r"(?P<user>[^:@/]+):(?P<pwd>[^@/]+)@"
    r"(?P<host>.+)$"
)


def _strip_inline_url_creds(url: str) -> str:
    """Rewrite ``scheme://user:password@host/db`` to drop the password.

    The on-disk config is intended for ``$ENV_VAR`` references; an
    inline password in a Postgres URL would land verbatim on disk and
    leak through any ``cat config.toml`` or backup tarball.
    """
    if not isinstance(url, str):
        return url
    m = _INLINE_CRED_RE.match(url)
    if m is None:
        return url
    return f"{m.group('scheme')}{m.group('user')}@{m.group('host')}"


def _redact_credentials(options: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Keep credential-like keys out of the on-disk config."""
    if not options:
        return None
    redacted: dict[str, Any] = {}
    for k, v in options.items():
        if k in _CREDENTIAL_KEYS:
            continue
        if k in {"url", "uri", "dsn", "connection_string"}:
            redacted[k] = _strip_inline_url_creds(v)
        else:
            redacted[k] = v
    return redacted


def _write_default_stack_config(store_path: Path) -> Path | None:
    """Copy the bundled local-default ``attestor.yaml`` into the store if absent.

    Ships at ``attestor/config/defaults/local.yaml`` (in the wheel). The caller
    points ``ATTESTOR_CONFIG`` at the written file. Returns the path, or None if
    the template can't be located / already present.
    """
    dest = store_path / "attestor.yaml"
    if dest.exists():
        return dest
    try:
        from importlib import resources

        tmpl = resources.files("attestor.config").joinpath("defaults", "local.yaml").read_text()
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load bundled local.yaml template: %s", exc)
        return None
    dest.write_text(tmpl)
    return dest


def _set_secure_permissions(path: Path) -> None:
    """Chmod 0o600 on Unix; best-effort on other platforms."""
    try:
        os.chmod(path, 0o600)
    except (OSError, NotImplementedError) as exc:
        log.debug("Could not set 0o600 on %s: %s", path, exc)


def _snapshot_dir(path: Path) -> dict[str, bytes] | None:
    """Capture file contents of `path` for potential restore. Returns None if dir is absent."""
    if not path.exists():
        return None
    snapshot: dict[str, bytes] = {}
    for child in path.rglob("*"):
        if child.is_file():
            snapshot[str(child.relative_to(path))] = child.read_bytes()
    return snapshot


def _restore_from_snapshot(path: Path, snapshot: dict[str, bytes] | None) -> None:
    """Delete everything under `path` and restore the snapshot contents."""
    if path.exists():
        shutil.rmtree(path)
    if snapshot is None:
        return
    path.mkdir(parents=True, exist_ok=True)
    for rel_path, data in snapshot.items():
        target = path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)


def _verify_store(path: Path) -> tuple[bool, str | None]:
    """Open the store and run a health check. Returns (ok, error_message)."""
    try:
        from attestor import AgentMemory

        with AgentMemory(path) as mem:
            if hasattr(mem, "health"):
                mem.health()
            else:
                mem.stats()
        return True, None
    except Exception as exc:
        log.warning("Store health check failed: %s", exc)
        return False, f"{type(exc).__name__}: {exc}"


def init_store(
    path: Path,
    *,
    backend: str = "postgres",
    backend_options: dict[str, Any] | None = None,
    verify: bool = False,
) -> InitResult:
    """Create an Attestor store with a starter TOML config.

    Credential-like keys in `backend_options` are stripped before write.
    With `verify=True`, the store's side-effect files are rolled back on failure.
    """
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(
            f"Unsupported backend {backend!r}. Choose from {list(SUPPORTED_BACKENDS)}."
        )

    if backend == "postgres" and not (backend_options or {}).get("url"):
        raise ValueError("postgres backend requires a 'url' in backend_options")

    path.mkdir(parents=True, exist_ok=True)
    config_file = path / "config.toml"
    legacy_json = path / "config.json"
    if config_file.exists() or legacy_json.exists():
        raise FileExistsError(f"Config already exists at {path}")

    pre_snapshot = _snapshot_dir(path) if verify else None

    safe_options = _redact_credentials(backend_options)
    doc = _build_config(backend, safe_options)
    config_file.write_text(tomlkit.dumps(doc))
    _set_secure_permissions(config_file)

    # Drop the bundled local-default stack config (embedder/models/retrieval)
    # next to config.toml. Without it a fresh install has no attestor.yaml and
    # get_stack() fails / falls back to the bench config. The store config.toml
    # handles connections; this handles the stack.
    _write_default_stack_config(path)

    verified = False
    if verify:
        ok, reason = _verify_store(path)
        if not ok:
            _restore_from_snapshot(path, pre_snapshot)
            raise RuntimeError(f"Store health check failed ({reason}); rolled back config")
        verified = True

    return InitResult(config_path=config_file, backend=backend, verified=verified)


def _prompt_port(prompt: str, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip() or str(default)
        try:
            return int(raw)
        except ValueError:
            print(f"  '{raw}' is not a valid integer; try again.")


def init_store_interactive(path: Path, *, verify: bool = False) -> InitResult:
    """Prompt for backend choice and credentials, then init."""
    print(f"Initializing Attestor store at: {path}")
    print(f"\nAvailable backends: {', '.join(SUPPORTED_BACKENDS)}")
    backend = input("Backend [postgres]: ").strip() or "postgres"

    backend_options: dict[str, Any] = {}
    if backend == "postgres":
        url = input("URL [postgresql://localhost:5432]: ").strip() or "postgresql://localhost:5432"
        backend_options = {"url": url}

    return init_store(
        path,
        backend=backend,
        backend_options=backend_options or None,
        verify=verify,
    )
