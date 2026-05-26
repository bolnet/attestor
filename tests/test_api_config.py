# SPDX-FileCopyrightText: 2026 Surendra Singh <66422685+bolnet@users.noreply.github.com>
# SPDX-License-Identifier: MIT
"""Unit tests for attestor.api backend-config resolution.

Focus: the env-driven path (POSTGRES_URL / NEO4J_URI) must preserve the
Pinecone vector role instead of silently dropping it. Registering the
``pinecone`` backend is the *only* way to get a vector store — postgres
claims the document role only — so omitting it leaves recall with no
semantic lane (regression introduced when vector moved off pgvector).
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from attestor import api


def _fake_stack(*, pinecone: bool = True):
    """Minimal stand-in for attestor.config.StackConfig."""
    pcn = (
        SimpleNamespace(
            index_name="attestor",
            metric="cosine",
            cloud="aws",
            region="us-east-1",
            host=None,
            api_key_env="PINECONE_API_KEY",
        )
        if pinecone
        else None
    )
    return SimpleNamespace(
        pinecone=pcn,
        embedder=SimpleNamespace(dimensions=1024),
    )


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for var in (
        "POSTGRES_URL",
        "NEO4J_URI",
        "PINECONE_API_KEY",
        "PINECONE_HOST",
        "PINECONE_INDEX",
    ):
        monkeypatch.delenv(var, raising=False)
    yield


def test_env_path_includes_pinecone_from_yaml(monkeypatch):
    """POSTGRES_URL + NEO4J_URI set, YAML has a pinecone block → vector role kept."""
    monkeypatch.setenv("POSTGRES_URL", "postgresql://pg:5432/attestor")
    monkeypatch.setenv("NEO4J_URI", "bolt://neo:7687")
    monkeypatch.setattr("attestor.config.get_stack", lambda *a, **k: _fake_stack())

    cfg = api._build_config()

    assert cfg is not None
    assert set(cfg["backends"]) == {"postgres", "neo4j", "pinecone"}
    assert cfg["pinecone"]["index_name"] == "attestor"
    assert cfg["pinecone"]["dimension"] == 1024


def test_env_path_pinecone_env_overrides_yaml(monkeypatch):
    """PINECONE_* env overrides the YAML secret + host + index name."""
    monkeypatch.setenv("POSTGRES_URL", "postgresql://pg:5432/attestor")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk-secret")
    monkeypatch.setenv("PINECONE_HOST", "https://idx.svc.pinecone.io")
    monkeypatch.setenv("PINECONE_INDEX", "prod-index")
    monkeypatch.setattr("attestor.config.get_stack", lambda *a, **k: _fake_stack())

    cfg = api._build_config()

    assert "pinecone" in cfg["backends"]
    assert cfg["pinecone"]["api_key"] == "pcsk-secret"
    assert cfg["pinecone"]["host"] == "https://idx.svc.pinecone.io"
    assert cfg["pinecone"]["index_name"] == "prod-index"


def test_env_path_no_pinecone_when_yaml_absent_and_no_env(monkeypatch):
    """Legacy pgvector-only deploy (no YAML pinecone, no env) → no vector backend."""
    monkeypatch.setenv("POSTGRES_URL", "postgresql://pg:5432/attestor")
    monkeypatch.setenv("NEO4J_URI", "bolt://neo:7687")
    monkeypatch.setattr(
        "attestor.config.get_stack", lambda *a, **k: _fake_stack(pinecone=False)
    )

    cfg = api._build_config()

    assert "pinecone" not in cfg["backends"]
    assert "pinecone" not in cfg


def test_env_path_pinecone_when_yaml_unavailable_but_env_set(monkeypatch):
    """No YAML at all, but PINECONE_API_KEY + PINECONE_INDEX set → vector kept."""
    monkeypatch.setenv("POSTGRES_URL", "postgresql://pg:5432/attestor")
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk-secret")
    monkeypatch.setenv("PINECONE_INDEX", "prod-index")

    def _boom(*a, **k):
        raise SystemExit("YAML missing required env")

    monkeypatch.setattr("attestor.config.get_stack", _boom)

    cfg = api._build_config()

    assert "pinecone" in cfg["backends"]
    assert cfg["pinecone"]["index_name"] == "prod-index"
    assert cfg["pinecone"]["api_key"] == "pcsk-secret"


def test_pinecone_helper_returns_none_without_index(monkeypatch):
    """No YAML pinecone, PINECONE_API_KEY set but no index → cannot reach an index."""
    monkeypatch.setenv("PINECONE_API_KEY", "pcsk-secret")

    def _boom(*a, **k):
        raise RuntimeError("no yaml")

    monkeypatch.setattr("attestor.config.get_stack", _boom)

    assert api._pinecone_config_from_env() is None
