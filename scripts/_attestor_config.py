"""Single-source-of-truth config loader for Attestor.

Every benchmark, smoke runner, and deploy script in `scripts/` reads
the default stack from `configs/attestor.yaml` via this module. No
script is allowed to hardcode model names, registry addresses, embedder
choices, or DB URLs.

Public surface:

    load_stack(path=None) -> StackConfig
        Parse + resolve the YAML into an immutable dataclass.

    configure_embedder(stack)
        Pin the embedder selection through env vars so the auto-detect
        chain in attestor.store.embeddings.get_embedding_provider() picks
        what the YAML asked for.

    build_backend_config(stack, *, no_graph=False)
        AgentMemory backend_configs payload for the canonical PG+Neo4j
        stack (or PG-only when no_graph).

    verify_neo4j_reachable(stack)
        Connect to Neo4j up-front so a benchmark that *expects* the
        graph role doesn't fall through silently.

    print_stack_banner(stack, run_label)
    confirm_or_exit(stack, run_label, yes)
        Loud banner of resolved values + interactive confirm (or --yes).
        Always called BEFORE any expensive call — per
        `feedback_canonical_stack.md`.

    cloud_target(stack, name) -> CloudTarget
        Return the resolved deploy plan for "gcp" / "azure" / "aws"
        with the native-registry image ref already substituted.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "attestor.yaml"


# ─── Dataclasses (frozen for immutability per project coding standards) ───

@dataclass(frozen=True)
class PostgresCfg:
    url: str
    v4: bool
    skip_schema_init: bool


@dataclass(frozen=True)
class Neo4jCfg:
    url: str
    username: str
    password: str
    database: str


@dataclass(frozen=True)
class EmbedderCfg:
    provider: str
    model: str
    dimensions: int


@dataclass(frozen=True)
class ModelsCfg:
    answerer: str
    judge: str
    extraction: str
    distill: str
    verifier: str


@dataclass(frozen=True)
class ImageCfg:
    ref: str
    api_ref_template: str
    registries: Dict[str, str] = field(default_factory=dict)

    def native_ref(self, key: str, *, version: Optional[str] = None) -> str:
        """Return the registry-specific image reference. When `version` is
        passed we use the api_ref_template (so cloud deploys pull
        `attestor:api-<version>` instead of the introspection-only stub).
        """
        base = self.registries[key]
        tag = f"api-{version}" if version else "latest"
        return f"{base}:{tag}"


@dataclass(frozen=True)
class CloudTarget:
    name: str
    region: str
    compute: str
    postgres: str
    neo4j: str
    image_ref: str


@dataclass(frozen=True)
class StackConfig:
    postgres: PostgresCfg
    neo4j: Neo4jCfg
    embedder: EmbedderCfg
    models: ModelsCfg
    image: ImageCfg
    budget: int
    parallel: int
    clouds: Dict[str, Dict[str, Any]]


# ─── YAML resolution helpers ───

def _resolve_env_password(node: Dict[str, Any]) -> str:
    """Pull a password from `password` (literal) or `password_env` (env)."""
    if isinstance(node, dict):
        if node.get("password"):
            return str(node["password"])
        env_name = node.get("password_env")
        if env_name:
            value = os.environ.get(env_name)
            if not value:
                raise SystemExit(
                    f"[config] required env var {env_name!r} not set "
                    "(referenced by stack.neo4j.auth.password_env)"
                )
            return value
    raise SystemExit(f"[config] could not resolve password from {node!r}")


def _resolve_env_api_key(env_var: Optional[str]) -> Optional[str]:
    """Resolve embedder api_key_env. Returns None when env_var is None
    so callers can decide what to do."""
    if not env_var:
        return None
    value = os.environ.get(env_var)
    if not value:
        raise SystemExit(
            f"[config] required env var {env_var!r} not set "
            "(referenced by stack.embedder.api_key_env)"
        )
    return value


def load_stack(path: Path | str | None = None) -> StackConfig:
    """Read `configs/attestor.yaml` (or override path) and resolve env
    references. Fails loudly when required values are missing."""
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.exists():
        raise SystemExit(f"[config] not found: {cfg_path}")
    raw = yaml.safe_load(cfg_path.read_text())

    stack_blk = raw.get("stack") or {}
    image_blk = raw.get("image") or {}
    clouds_blk = raw.get("clouds") or {}

    pg = stack_blk.get("postgres") or {}
    neo = stack_blk.get("neo4j") or {}
    emb = stack_blk.get("embedder") or {}
    models = stack_blk.get("models") or {}

    # Validate required embedder API key — fail now rather than mid-run.
    _resolve_env_api_key(emb.get("api_key_env"))

    return StackConfig(
        postgres=PostgresCfg(
            url=pg["url"],
            v4=bool(pg.get("v4", True)),
            skip_schema_init=bool(pg.get("skip_schema_init", True)),
        ),
        neo4j=Neo4jCfg(
            url=neo["url"],
            username=(neo.get("auth") or {}).get("username", "neo4j"),
            password=_resolve_env_password(neo.get("auth") or {}),
            database=neo.get("database", "neo4j"),
        ),
        embedder=EmbedderCfg(
            provider=emb.get("provider", "voyage"),
            model=emb.get("model", "voyage-4"),
            dimensions=int(emb.get("dimensions", 1024)),
        ),
        models=ModelsCfg(
            answerer=models["answerer"],
            judge=models["judge"],
            extraction=models["extraction"],
            distill=models["distill"],
            verifier=models["verifier"],
        ),
        image=ImageCfg(
            ref=image_blk.get("ref", ""),
            api_ref_template=image_blk.get("api_ref_template", ""),
            registries=dict(image_blk.get("registries") or {}),
        ),
        budget=int(stack_blk.get("budget", 4000)),
        parallel=int(stack_blk.get("parallel", 2)),
        clouds=dict(clouds_blk),
    )


# ─── Embedder env wiring ───

def configure_embedder(stack: StackConfig) -> None:
    """Pin embedder selection via env vars so
    attestor.store.embeddings.get_embedding_provider() returns the
    provider the YAML asked for. Ollama auto-probe is always disabled."""
    os.environ["ATTESTOR_DISABLE_LOCAL_EMBED"] = "1"
    if stack.embedder.provider == "voyage":
        if not os.environ.get("VOYAGE_API_KEY"):
            raise SystemExit("[config] embedder=voyage but VOYAGE_API_KEY not set")
        os.environ["VOYAGE_EMBEDDING_MODEL"] = stack.embedder.model
        os.environ["VOYAGE_EMBEDDING_DIMENSIONS"] = str(stack.embedder.dimensions)
        os.environ["OPENAI_EMBEDDING_DIMENSIONS"] = str(stack.embedder.dimensions)
    elif stack.embedder.provider == "openai":
        # Voyage key, if set, would auto-win the chain. Pop it process-locally.
        os.environ.pop("VOYAGE_API_KEY", None)
        os.environ["OPENAI_EMBEDDING_MODEL"] = stack.embedder.model
        os.environ["OPENAI_EMBEDDING_DIMENSIONS"] = str(stack.embedder.dimensions)
    else:
        raise SystemExit(
            f"[config] unknown embedder provider: {stack.embedder.provider!r}"
        )


# ─── AgentMemory factory payload ───

def build_backend_config(
    stack: StackConfig, *, no_graph: bool = False
) -> Dict[str, Any]:
    """AgentMemory `backend_configs` payload for the canonical PG+Neo4j
    stack. When no_graph is set, drops Neo4j (graph role) — used for
    isolation runs that prove whether the graph layer contributes."""
    from urllib.parse import urlparse

    parsed = urlparse(stack.postgres.url)
    db = (parsed.path or "/").lstrip("/") or "attestor_v4_test"

    backend_configs: Dict[str, Dict[str, Any]] = {
        "postgres": {
            "url": f"{parsed.scheme}://{parsed.hostname}:{parsed.port or 5432}",
            "database": db,
            "auth": {
                "username": parsed.username or "postgres",
                "password": parsed.password or "attestor",
            },
            "v4": stack.postgres.v4,
            "skip_schema_init": stack.postgres.skip_schema_init,
        },
    }
    backends = ["postgres"]
    if not no_graph:
        backend_configs["neo4j"] = {
            "url": stack.neo4j.url,
            "database": stack.neo4j.database,
            "auth": {
                "username": stack.neo4j.username,
                "password": stack.neo4j.password,
            },
        }
        backends.append("neo4j")

    return {
        "mode": "solo",
        "backends": backends,
        "backend_configs": backend_configs,
    }


# ─── Health checks ───

def verify_neo4j_reachable(stack: StackConfig) -> None:
    """Connect to Neo4j up front. Fail loudly with the actual error
    rather than letting the orchestrator silently fall through to
    affinity_map={} when graph queries explode mid-run."""
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise SystemExit(
            f"[config] neo4j driver not installed: {e}. "
            "Run `pip install neo4j` or `pip install attestor[neo4j]`."
        )
    try:
        with GraphDatabase.driver(
            stack.neo4j.url, auth=(stack.neo4j.username, stack.neo4j.password)
        ) as drv:
            drv.verify_connectivity()
    except Exception as e:
        raise SystemExit(
            f"[config] Neo4j unreachable at {stack.neo4j.url}: {e}\n"
            "        The default stack requires Neo4j (graph role).\n"
            "        Bring it up: docker compose -f attestor/infra/local/docker-compose.yml up -d neo4j"
        )


# ─── Cloud target resolver ───

def cloud_target(stack: StackConfig, name: str, *, version: Optional[str] = None) -> CloudTarget:
    """Return the resolved deploy plan for one cloud, with the native-
    registry image ref already substituted. `version` switches from the
    introspection-only `:latest` tag to the api-mode `:api-<version>`."""
    if name not in stack.clouds:
        available = ", ".join(sorted(stack.clouds))
        raise SystemExit(f"[config] unknown cloud {name!r} (available: {available})")
    cloud = stack.clouds[name]
    image_ref = stack.image.native_ref(cloud["image_ref_key"], version=version)
    return CloudTarget(
        name=name,
        region=cloud["region"],
        compute=cloud["compute"],
        postgres=cloud["postgres"],
        neo4j=cloud["neo4j"],
        image_ref=image_ref,
    )


# ─── Banner + confirm ───

def print_stack_banner(stack: StackConfig, *, run_label: str) -> None:
    """Loud print of the resolved stack so the user can sanity-check
    before any expensive call goes out."""
    print("=" * 72)
    print(f"[{run_label}] resolved Attestor stack (configs/attestor.yaml):")
    print(f"  document/vector  postgres @ {stack.postgres.url}")
    print(f"  graph            neo4j    @ {stack.neo4j.url} ({stack.neo4j.database})")
    print(f"  embedder         {stack.embedder.provider}:{stack.embedder.model}"
          f" @ {stack.embedder.dimensions}-D")
    print(f"  models")
    print(f"    answerer       {stack.models.answerer}")
    print(f"    judge          {stack.models.judge}")
    print(f"    extraction     {stack.models.extraction}")
    print(f"    distill        {stack.models.distill}")
    print(f"    verifier       {stack.models.verifier}")
    print(f"  budget           {stack.budget} tokens · parallel = {stack.parallel}")
    print("=" * 72)


def confirm_or_exit(stack: StackConfig, *, run_label: str, yes: bool) -> None:
    """Per `feedback_canonical_stack.md`, every benchmark or deploy run
    either prompts the user to confirm the resolved stack or is invoked
    with `--yes` for non-interactive use. ALWAYS called BEFORE any
    expensive call (LLM, cloud provision, image push)."""
    print_stack_banner(stack, run_label=run_label)
    if yes:
        print(f"[{run_label}] --yes supplied; proceeding without prompt")
        return
    if not sys.stdin.isatty():
        raise SystemExit(
            f"[{run_label}] non-interactive shell and --yes not supplied; "
            "aborting to avoid running with the wrong stack."
        )
    answer = input(f"[{run_label}] Proceed with this stack? [y/N] ").strip().lower()
    if answer not in ("y", "yes"):
        raise SystemExit(f"[{run_label}] aborted by user")
