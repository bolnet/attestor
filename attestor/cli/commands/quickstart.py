# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""``attestor quickstart`` — zero-question, one-command local install.

ONE default profile, NO prompts. In a single run it:

  1. writes the store connection config (``config.toml``) and stack config
     (``attestor.yaml``) from the bundled local default,
  2. writes ``~/.attestor/.env`` (local-dev passwords + the Ollama embedder
     route) — idempotent, never clobbers existing values,
  3. brings up the local Docker backends (Postgres + Neo4j),
  4. wires the Claude Code MCP server + lifecycle hooks (``.env``-sourcing),
  5. runs the health check.

Everything is PRINTED for transparency; nothing is ASKED. Re-runnable.

This is the front door. It deliberately BYPASSES ``init``'s fresh-store gate by
force-writing ``config.toml`` via the canonical writer, so the background MCP
server's tuning-only ``config.json`` can't block the connection config. The
deeper config-path consolidation is tracked separately.
"""
from __future__ import annotations

import contextlib
import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import tomlkit

from attestor.init_wizard import _build_config, _write_default_stack_config

if TYPE_CHECKING:
    import argparse

# ── The single local default profile ────────────────────────────────────────
# Ports bind to localhost only, so a well-known dev password is acceptable and
# removes all coordination between the compose file, the .env, and config.toml.
# Override by exporting the matching env vars before running.
DEFAULT_PASSWORD = "attestor"  # noqa: S105 — intentional localhost-only dev default
DEFAULT_STORE = Path("~/.attestor").expanduser()
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_EMBED_MODEL = "bge-m3"
EMBED_DIM = 1024  # bge-m3 output dim == Pinecone index dim
PINECONE_LOCAL_HOST = "http://localhost:5080"
PG_CONTAINER = "attestor_postgres_document_db"
NEO4J_CONTAINER = "attestor_neo4j_graph_db"
HEALTH_TIMEOUT_S = 120

# Ports the local stack uses — scanned up front so the user sees what's already
# listening (backends already up) vs free (will be started) before anything runs.
SCAN_PORTS = {
    "Postgres": 5432,
    "Neo4j Bolt": 7687,
    "Pinecone Local": 5080,
    "Ollama": 11434,
}


def _default_env(store: Path) -> dict[str, str]:
    """Required ``.env`` keys for the canonical local profile (3 roles, zero cloud key)."""
    return {
        "PGPASSWORD": DEFAULT_PASSWORD,        # config.toml [postgres.auth] -> $PGPASSWORD
        "POSTGRES_PASSWORD": DEFAULT_PASSWORD,  # docker compose postgres
        "NEO4J_PASSWORD": DEFAULT_PASSWORD,     # config.toml + compose + local.yaml
        "PINECONE_API_KEY": "local",            # Pinecone Local needs a non-empty key (ignored)
        "OPENAI_API_KEY": "ollama",             # Ollama OpenAI-compat sentinel (no real key)
        "OPENAI_BASE_URL": OLLAMA_BASE_URL,     # local Ollama endpoint
        "ATTESTOR_CONFIG": str(store / "attestor.yaml"),
    }


def _compose_file() -> Path:
    """Path to the bundled local docker-compose.yml (ships in the wheel)."""
    import attestor

    return Path(attestor.__file__).parent / "infra" / "local" / "docker-compose.yml"


# ── Preflight (scan ports + tools up front; uses the fixed default creds) ─────
def _port_open(port: int, host: str = "localhost", timeout: float = 1.0) -> bool:
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=timeout):
        return True
    return False


def _ollama_model_state(model: str) -> bool | None:
    """True = model pulled, False = serving but model missing, None = not reachable."""
    try:
        import requests

        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        return model in resp.text
    except Exception:  # any failure (not serving, timeout, conn refused) == not reachable
        return None


def _preflight() -> None:
    """Scan ports + Docker + Ollama before acting, so the env state is visible."""
    print("[0/6] Preflight — scanning ports + tools (default credentials, no prompts)")
    print(f"  docker .............. {'available' if _docker_available() else 'NOT available'}")
    for name, port in SCAN_PORTS.items():
        state = "listening (already up)" if _port_open(port) else "free (will start)"
        print(f"  :{port:<6}{name:<16} {state}")
    model = _ollama_model_state(OLLAMA_EMBED_MODEL)
    pull = f"ollama pull {OLLAMA_EMBED_MODEL}"
    if model is None:
        print(f"  ollama .......... not reachable on :11434 — start it, then '{pull}'")
    elif model:
        print(f"  ollama .......... serving; '{OLLAMA_EMBED_MODEL}' pulled")
    else:
        print(f"  ollama .......... serving, but '{OLLAMA_EMBED_MODEL}' not pulled — '{pull}'")
    print()


# ── Steps (each prints; none prompts) ────────────────────────────────────────
def _write_config_toml(store: Path) -> Path:
    """Force-write the canonical 3-role connection config.

    Backs up any existing ``config.toml`` first. Unlike ``init``, this is NOT
    gated on a pre-existing ``config.json`` (the MCP-server race), so it always
    leaves a correct, ``$ENV_VAR``-auth connection config in place.

    The default stack is the canonical three roles — Postgres (document) +
    Pinecone Local (vector) + Neo4j (graph). ``_build_config`` emits the
    postgres/neo4j tables; we add ``pinecone`` to ``backends`` + a ``[pinecone]``
    table so the registry routes the vector role to Pinecone Local (its defaults
    — ``localhost:5080``, dim 1024 — already match Ollama bge-m3, and the API key
    comes from ``$PINECONE_API_KEY`` in ``.env``).
    """
    cfg = store / "config.toml"
    if cfg.exists():
        cfg.replace(store / "config.toml.bak")
        print(f"  backed up existing config.toml -> {store / 'config.toml.bak'}")
    doc = _build_config(
        "postgres",
        {"url": "postgresql://localhost:5432", "database": "attestor"},
    )
    doc["backends"] = ["postgres", "pinecone", "neo4j"]
    pinecone = tomlkit.table()
    pinecone["host"] = PINECONE_LOCAL_HOST
    pinecone["dimension"] = EMBED_DIM
    pinecone["metric"] = "cosine"
    doc["pinecone"] = pinecone
    cfg.write_text(tomlkit.dumps(doc))
    with contextlib.suppress(OSError):
        cfg.chmod(0o600)
    print(f"  wrote {cfg}")
    return cfg


def _ensure_env(store: Path) -> Path:
    """Append any missing default ``.env`` keys (idempotent; existing wins)."""
    env_path = store / ".env"
    present: set[str] = set()
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            s = line.strip()
            if s and not s.startswith("#") and "=" in s:
                present.add(s.split("=", 1)[0].strip())

    missing = {k: v for k, v in _default_env(store).items() if k not in present}
    if missing:
        prefix = "\n" if env_path.exists() and env_path.read_text().strip() else ""
        with env_path.open("a") as f:
            f.write(f"{prefix}# Attestor quickstart — local default profile\n")
            for k, v in missing.items():
                f.write(f"{k}={v}\n")
        print(f"  added {len(missing)} key(s) to {env_path}: {', '.join(sorted(missing))}")
    else:
        print(f"  {env_path} already complete")
    with contextlib.suppress(OSError):
        env_path.chmod(0o600)
    return env_path


def _load_env_into_os(env_path: Path) -> None:
    """Load ``.env`` into ``os.environ`` (setdefault — live shell env wins)."""
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, v = s.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def _docker_available() -> bool:
    if not shutil.which("docker"):
        return False
    try:
        return subprocess.run(
            ["docker", "info"], capture_output=True, timeout=15
        ).returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _container_healthy(name: str) -> bool:
    try:
        out = subprocess.run(
            ["docker", "inspect", "--format", "{{.State.Health.Status}}", name],
            capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return False
    return out.returncode == 0 and out.stdout.strip() == "healthy"


def _bring_up_backends() -> bool:
    """Start the 3-role backends: postgres + neo4j + pinecone (Pinecone Local).

    The standalone ``attestor-api`` service is intentionally NOT started — the
    Claude Code path runs the MCP server from the installed binary, not that
    container. Only postgres + neo4j have healthchecks; pinecone has none, so we
    start it alongside and let the Pinecone backend's own readiness probe handle
    it. Returns True if PG + Neo4j are healthy. Best-effort: prints the manual
    command and returns False rather than raising if Docker is unavailable.
    """
    compose = _compose_file()
    cmd = ["docker", "compose", "-f", str(compose), "up", "-d",
           "postgres", "neo4j", "pinecone"]
    if not _docker_available():
        print("  Docker not available — start the backends yourself:")
        print(f"    {' '.join(cmd)}")
        return False
    print(f"  {' '.join(cmd)}")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"  docker compose failed: {exc}")
        return False
    if proc.returncode != 0:
        print(f"  docker compose returned {proc.returncode}:\n{proc.stderr.strip()[-800:]}")
        return False

    print("  waiting for Postgres + Neo4j healthchecks...", end="", flush=True)
    deadline = time.time() + HEALTH_TIMEOUT_S
    while time.time() < deadline:
        if _container_healthy(PG_CONTAINER) and _container_healthy(NEO4J_CONTAINER):
            print(" both healthy.")
            return True
        print(".", end="", flush=True)
        time.sleep(3)
    print(" timed out (continuing; doctor will report actual state).")
    return False


def _wire_claude_code(store: Path) -> None:
    """Wire the MCP server (./.mcp.json) + hooks (settings.json), .env-sourcing."""
    from attestor.cli._setup_helpers import (
        _configure_claude_hooks,
        _configure_claude_mcp,
    )

    attestor_bin = shutil.which("attestor") or "attestor"
    # So _hook_command bakes the right ATTESTOR_CONFIG into the hook commands.
    os.environ["ATTESTOR_CONFIG"] = str(store / "attestor.yaml")
    _configure_claude_mcp(attestor_bin, str(store))
    _configure_claude_hooks(attestor_bin)


def _run_doctor(store: Path) -> None:
    import logging

    logging.getLogger("attestor").setLevel(logging.CRITICAL)
    from attestor.cli.commands.setup import _print_health_report
    from attestor.core import AgentMemory

    try:
        mem = AgentMemory(str(store))
        report = mem.health()
        _print_health_report(report)
        mem.close()
    except Exception as exc:  # surface, never crash the install
        print(f"  doctor could not open the store: {type(exc).__name__}: {exc}")


def _print_profile(store: Path) -> None:
    print("Attestor Quickstart — single local default profile (zero questions)")
    print("=" * 66)
    print("Everything below is fixed by default and printed, never asked:")
    print(f"  store path .......... {store}")
    print("  backends ........ Postgres (doc) + Pinecone Local (vector) + Neo4j (graph)")
    print(f"  embedder ........ Ollama {OLLAMA_EMBED_MODEL} @{EMBED_DIM}d (local, zero cloud key)")
    print("  llm keys ........ none required (recall/add work fully local)")
    print(f"  passwords ....... '{DEFAULT_PASSWORD}' (localhost dev default; Pinecone key 'local')")
    print("  token budget ........ 10000")
    print()


def _cmd_quickstart(args: argparse.Namespace) -> None:
    store = Path(getattr(args, "path", None) or DEFAULT_STORE).expanduser()
    store.mkdir(parents=True, exist_ok=True)

    _print_profile(store)

    _preflight()

    print("[1/6] Stack config (attestor.yaml)")
    pre_existing = (store / "attestor.yaml").exists()
    yaml_path = _write_default_stack_config(store) or (store / "attestor.yaml")
    print(f"  {'using existing' if pre_existing else 'wrote'} {yaml_path}")

    print("\n[2/6] Connection config (config.toml)")
    _write_config_toml(store)

    print("\n[3/6] Environment (.env)")
    env_path = _ensure_env(store)
    _load_env_into_os(env_path)

    if not getattr(args, "no_docker", False):
        print("\n[4/6] Local Docker backends (Postgres + Neo4j)")
        _bring_up_backends()
    else:
        print("\n[4/6] Local Docker backends — skipped (--no-docker)")

    if not getattr(args, "no_wire", False):
        print("\n[5/6] Claude Code wiring (MCP server + hooks)")
        _wire_claude_code(store)
    else:
        print("\n[5/6] Claude Code wiring — skipped (--no-wire)")

    print("\n[6/6] Health check")
    if getattr(args, "no_verify", False):
        print("  skipped (--no-verify)")
    else:
        _run_doctor(store)

    # The resolved stack, so the user sees the single source of truth in effect.
    print()
    with contextlib.suppress(Exception):
        from attestor.cli._setup_helpers import _print_active_config

        _print_active_config()

    print("Done. Next:")
    print("  • Restart Claude Code so the MCP server + hooks attach (they load at session start).")
    print(f"  • Ensure Ollama is serving {OLLAMA_EMBED_MODEL}:  ollama pull {OLLAMA_EMBED_MODEL}")
    print("  • Then /mcp should show 'attestor' (8 tools); recall returns source: vector.")
