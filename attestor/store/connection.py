"""Layered connection config — reusable across all backends.

Industry-standard 3-layer resolution (each overrides the previous):

    Layer 1: Code defaults     (ENGINE_DEFAULTS per backend type)
    Layer 2: Config file       (project config.json or programmatic dict)
    Layer 3: Environment vars  ($ENV_VAR references auto-resolved)

Supports two config formats (hybrid, following SQLAlchemy/Django/Prisma):

    URL string (12-Factor friendly):
        {"url": "arangodb://root:$PW@cloud.example.com:8529/attestor"}

    Structured dict (for complex cases):
        {
            "mode": "cloud",
            "url": "https://cloud.example.com:8529",
            "database": "attestor",
            "auth": {"username": "root", "password": "$ARANGO_PASSWORD"},
            "tls": {"verify": true, "ca_cert": "/path/to/ca.pem"}
        }

Adding a new engine = add an entry to ENGINE_DEFAULTS. No code changes.
"""

from __future__ import annotations

import atexit
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


# ═══════════════════════════════════════════════════════════════════════
# Layer 1: Engine defaults (per database type)
# ═══════════════════════════════════════════════════════════════════════

ENGINE_DEFAULTS: dict[str, dict[str, Any]] = {
    "arangodb": {
        "url": "http://localhost:8529",
        "port": 8529,
        "database": "attestor",
        "auth": {"username": "root", "password": ""},
        "tls": {"verify": False},
    },
    "postgres": {
        "url": "postgresql://localhost:5432",
        "port": 5432,
        "database": "attestor",
        "auth": {"username": "postgres", "password": ""},
        "tls": {"verify": False},
    },
    "neo4j": {
        "url": "bolt://localhost:7687",
        "port": 7687,
        "database": "neo4j",
        "auth": {"username": "neo4j", "password": ""},
        "tls": {"verify": False},
    },
    "surrealdb": {
        "url": "http://localhost:8000",
        "port": 8000,
        "database": "attestor",
        "auth": {"username": "root", "password": "root"},
        "tls": {"verify": False},
    },
    "gcp": {
        "url": "postgresql://localhost:5432",
        "port": 5432,
        "database": "attestor",
        "auth": {"username": "postgres", "password": ""},
        "project_id": "",
        "region": "us-central1",
        "cluster": "",
        "instance": "",
        "tls": {"verify": True},
    },
    "azure": {
        "url": "",
        "cosmos_endpoint": "",
        "cosmos_database": "attestor",
        "auth": {"api_key": ""},
        "tls": {"verify": True},
    },
    "aws": {
        "url": "",
        "region": "us-east-1",
        "dynamodb": {"table_prefix": "attestor"},
        "opensearch": {"endpoint": "", "index": "memories"},
        "neptune": {"endpoint": ""},
        "auth": {},
        "tls": {"verify": True},
    },
}

# Backward-compat aliases
BACKEND_DEFAULTS = ENGINE_DEFAULTS
CLOUD_DEFAULTS: dict[str, Any] = {
    "mode": "cloud",
    "database": "attestor",
    "auth": {"username": "", "password": "", "token": "", "api_key": ""},
    "tls": {"verify": True, "ca_cert": None},
}


# ═══════════════════════════════════════════════════════════════════════
# Deep merge
# ═══════════════════════════════════════════════════════════════════════

def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge two dicts. override wins on conflicts. Returns new dict."""
    merged = dict(base)
    for key, val in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(val, dict)
        ):
            merged[key] = _deep_merge(merged[key], val)
        else:
            merged[key] = val
    return merged


def merge_config_layers(*layers: dict[str, Any]) -> dict[str, Any]:
    """Merge config layers in precedence order (last wins).

    None and empty layers are skipped.
    """
    result: dict[str, Any] = {}
    for layer in layers:
        if layer:
            result = _deep_merge(result, layer)
    return result


# ═══════════════════════════════════════════════════════════════════════
# URL parsing (dialect://user:pass@host:port/database?options)
# ═══════════════════════════════════════════════════════════════════════

# Map URL schemes to engine names
_SCHEME_MAP: dict[str, str] = {
    "arangodb": "arangodb",
    "arangodb+https": "arangodb",
    "arango": "arangodb",
    "postgresql": "postgres",
    "postgres": "postgres",
    "bolt": "neo4j",
    "neo4j": "neo4j",
    "neo4j+s": "neo4j",
    "surrealdb": "surrealdb",
    "surreal": "surrealdb",
}


def parse_url(url: str) -> dict[str, Any]:
    """Parse a connection URL into a structured config dict.

    Follows the SQLAlchemy dialect://user:pass@host:port/database convention.

    Examples:
        arangodb://root:pass@cloud.example.com:8529/attestor
        postgresql://user:pass@rds.amazonaws.com:5432/mydb?sslmode=require
        neo4j+s://user:pass@aura.neo4j.io/mydb
    """
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()

    config: dict[str, Any] = {}

    # Detect TLS from scheme suffix
    use_tls = scheme.endswith("+s") or scheme.endswith("+https") or parsed.scheme == "https"
    base_scheme = re.sub(r"\+(s|https?)$", "", scheme)

    if base_scheme in _SCHEME_MAP:
        config["_engine"] = _SCHEME_MAP[base_scheme]

    # Build the base URL (protocol + host + port)
    protocol = "https" if use_tls else "http"
    host = parsed.hostname or "localhost"
    port = parsed.port

    if port:
        config["url"] = f"{protocol}://{host}:{port}"
        config["port"] = port
    else:
        config["url"] = f"{protocol}://{host}"

    # Database from path
    if parsed.path and parsed.path != "/":
        config["database"] = parsed.path.lstrip("/")

    # Auth
    auth: dict[str, str] = {}
    if parsed.username:
        auth["username"] = unquote(parsed.username)
    if parsed.password:
        auth["password"] = unquote(parsed.password)
    if auth:
        config["auth"] = auth

    # TLS
    if use_tls:
        config["tls"] = {"verify": True}
        config["mode"] = "cloud"

    # Query params → extra options
    if parsed.query:
        params = parse_qs(parsed.query, keep_blank_values=True)
        # Flatten single-value params
        config["options"] = {k: v[0] if len(v) == 1 else v for k, v in params.items()}

    return config


# ═══════════════════════════════════════════════════════════════════════
# Env resolution
# ═══════════════════════════════════════════════════════════════════════

def resolve_env(value: Any) -> Any:
    """Resolve $ENV_VAR references in config values.

    Supports:
        "$MY_VAR"      → os.environ["MY_VAR"] (or original string if unset)
        "${MY_VAR}"    → same, with braces
        plain strings  → returned as-is
        non-strings    → returned as-is
    """
    if not isinstance(value, str):
        return value
    if value.startswith("${") and value.endswith("}"):
        return os.environ.get(value[2:-1], value)
    if value.startswith("$"):
        return os.environ.get(value[1:], value)
    return value


def resolve_env_recursive(data: dict[str, Any]) -> dict[str, Any]:
    """Resolve all $ENV_VAR references in a nested config dict."""
    resolved = {}
    for key, val in data.items():
        if isinstance(val, dict):
            resolved[key] = resolve_env_recursive(val)
        elif isinstance(val, list):
            resolved[key] = [resolve_env(v) for v in val]
        else:
            resolved[key] = resolve_env(val)
    return resolved


# ═══════════════════════════════════════════════════════════════════════
# Config builder (main entry point)
# ═══════════════════════════════════════════════════════════════════════

def build_config(
    engine: str,
    user_config: dict[str, Any],
    cli_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a fully-resolved config by stacking all 3 layers.

    Args:
        engine: Backend engine name (e.g., "arangodb", "postgres").
        user_config: Project-level config (from config.json or programmatic).
        cli_overrides: Optional CLI --backend-config overrides.

    Returns:
        Fully merged config dict with env vars resolved.
    """
    # If user provides a URL string, parse it into structured config
    config = dict(user_config)
    if "url" in config:
        url_val = resolve_env(config["url"])
        # Only parse if it looks like a connection URI (has scheme://)
        if "://" in url_val and _looks_like_connection_uri(url_val):
            url_config = parse_url(url_val)
            # URL fields are base, explicit config fields override
            url_config.pop("_engine", None)
            config = _deep_merge(url_config, config)

    return merge_config_layers(
        ENGINE_DEFAULTS.get(engine, {}),    # L1: engine defaults
        _normalize_flat_auth(config),        # L2: project config (normalized)
        cli_overrides or {},                 # L3: CLI overrides
    )


def _looks_like_connection_uri(url: str) -> bool:
    """Check if URL is a connection URI vs a plain HTTP endpoint."""
    scheme = url.split("://")[0].lower() if "://" in url else ""
    return scheme in _SCHEME_MAP or scheme in ("http", "https")


# ═══════════════════════════════════════════════════════════════════════
# Dataclasses
# ═══════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class AuthConfig:
    """Authentication credentials for a cloud backend.

    Supports multiple auth methods — backends use whichever fields they need:
        - username/password (ArangoDB, PostgreSQL, Neo4j)
        - token (bearer/JWT — managed cloud services)
        - api_key (SaaS providers)

    ``__repr__`` masks every secret-bearing field so credentials never
    leak through logs, exception traces, or debug prints.
    """
    username: str = ""
    password: str = ""
    token: str = ""
    api_key: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthConfig:
        return cls(
            username=resolve_env(data.get("username", "")),
            password=resolve_env(data.get("password", "")),
            token=resolve_env(data.get("token", "")),
            api_key=resolve_env(data.get("api_key", "")),
        )

    @property
    def has_credentials(self) -> bool:
        return bool(self.username or self.token or self.api_key)

    def __repr__(self) -> str:
        def mask(v: str | None) -> str:
            return "***" if v else "None"

        return (
            f"AuthConfig(username={self.username!r}, "
            f"password={mask(self.password)}, "
            f"token={mask(self.token)}, "
            f"api_key={mask(self.api_key)})"
        )


@dataclass(frozen=True, slots=True)
class TLSConfig:
    """TLS/SSL settings for cloud connections.

    ca_cert: Path to CA certificate file, or base64-encoded cert content.
             If base64, it will be decoded and written to a temp file.
    """
    verify: bool = True
    ca_cert: str | None = None
    ca_cert_base64: str | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TLSConfig:
        return cls(
            verify=data.get("verify", True),
            ca_cert=data.get("ca_cert"),
            ca_cert_base64=data.get("ca_cert_base64"),
        )

    def resolve_ca_cert_path(self, store_path: str | None = None) -> str | None:
        """Return path to CA cert file, decoding base64 if needed.

        If ca_cert_base64 is set, writes decoded cert to store_path/ca.crt
        (or a temp file if no store_path). Temp files are deleted at
        interpreter exit via ``atexit`` so we don't leak credentials on
        disk after the process dies.
        """
        import base64

        if self.ca_cert:
            return self.ca_cert

        if not self.ca_cert_base64:
            return None

        cert_bytes = base64.b64decode(self.ca_cert_base64)
        if store_path:
            cert_path = Path(store_path) / "ca.crt"
            cert_path.write_bytes(cert_bytes)
            return str(cert_path)

        return _materialize_temp_cert(cert_bytes)


def _materialize_temp_cert(cert_bytes: bytes) -> str:
    """Write CA cert bytes to a temp file and register cleanup at exit.

    Cleanup is best-effort — atexit handlers do not run on hard kills
    (SIGKILL, OOM, segfault), but they cover normal interpreter exit
    which is the dominant case.
    """
    fd, path = tempfile.mkstemp(suffix=".crt", prefix="attestor_ca_")
    try:
        os.write(fd, cert_bytes)
    finally:
        os.close(fd)

    def _cleanup(p: str = path) -> None:
        try:
            if os.path.exists(p):
                os.unlink(p)
        except OSError:
            # Best-effort — don't crash interpreter exit on cleanup races.
            pass

    atexit.register(_cleanup)
    return path


@dataclass(frozen=True, slots=True)
class CloudConnection:
    """Parsed, env-resolved connection config for any cloud backend.

    Supports both URL strings and structured dicts:
        URL:  "arangodb://root:pass@host:8529/attestor"
        Dict: {"url": "https://host:8529", "database": "attestor", ...}

    Resolution: engine defaults → user config → env vars
    """
    mode: str = "cloud"
    url: str = "http://localhost:8529"
    database: str = "attestor"
    port: int = 8529
    auth: AuthConfig = field(default_factory=AuthConfig)
    tls: TLSConfig = field(default_factory=TLSConfig)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        backend_name: str | None = None,
    ) -> CloudConnection:
        """Parse a config dict into a CloudConnection.

        Applies engine defaults automatically. Resolves env vars.

        Args:
            config: User config dict.
            backend_name: Engine name for defaults lookup.
        """
        if backend_name:
            merged = build_config(backend_name, config)
        else:
            merged = merge_config_layers(
                {"database": "attestor", "tls": {"verify": True}},
                _normalize_flat_auth(config),
            )

        # Resolve env vars
        merged = resolve_env_recursive(merged)

        mode = merged.get("mode", "cloud")
        url = merged.get("url", "http://localhost:8529")
        database = merged.get("database", "attestor")
        port = merged.get("port", 8529)

        auth = AuthConfig.from_dict(merged.get("auth", {}))
        tls = TLSConfig.from_dict(merged.get("tls", {}))

        # Override URL for local mode
        if mode == "local":
            url = f"http://localhost:{port}"

        known_keys = {"mode", "url", "database", "port", "auth", "tls", "options"}
        extra = {k: v for k, v in merged.items() if k not in known_keys}
        if "options" in merged:
            extra.update(merged["options"])

        return cls(
            mode=mode, url=url, database=database, port=port,
            auth=auth, tls=tls, extra=extra,
        )


def _normalize_flat_auth(config: dict[str, Any]) -> dict[str, Any]:
    """Move flat-level auth fields into nested 'auth' block.

    Supports backward-compat: {"username": "root"} → {"auth": {"username": "root"}}
    """
    flat_auth_keys = {"username", "password", "token", "api_key"}
    flat_found = {k: config[k] for k in flat_auth_keys if k in config}
    if not flat_found:
        return config

    result = {k: v for k, v in config.items() if k not in flat_auth_keys}
    existing_auth = result.get("auth", {})
    if isinstance(existing_auth, dict):
        result["auth"] = _deep_merge(flat_found, existing_auth)
    else:
        result["auth"] = flat_found
    return result
