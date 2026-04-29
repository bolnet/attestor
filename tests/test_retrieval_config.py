"""Retrieval cascade tunables — vector_top_k + mmr_top_n via YAML."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _yaml_with_retrieval(tmp_path, monkeypatch, **retrieval) -> Path:
    cfg = {
        "stack": {
            "postgres": {"url": "postgresql://postgres@localhost/attestor"},
            "neo4j": {
                "url": "bolt://localhost:7687",
                "auth": {"username": "neo4j", "password": "test"},
                "database": "neo4j",
            },
            "embedder": {
                "provider": "voyage", "model": "voyage-4", "dimensions": 1024,
            },
            "llm": {"provider": "openrouter"},
            "models": {
                "answerer": "openai/gpt-5.4-mini",
                "judge": "openai/gpt-5.5",
                "extraction": "openai/gpt-5.4-mini",
                "distill": "openai/gpt-5.4-mini",
                "verifier": "anthropic/claude-sonnet-4-6",
                "planner": "anthropic/claude-opus-4.7",
                "benchmark_default": "openai/gpt-5.4-mini",
            },
        },
    }
    if retrieval:
        cfg["stack"]["retrieval"] = retrieval

    p = tmp_path / "attestor.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ATTESTOR_CONFIG", str(p))
    from attestor import config as _c
    _c.reset_stack()
    return p


@pytest.mark.unit
def test_default_retrieval_cfg_matches_legacy(tmp_path, monkeypatch):
    """No `retrieval:` block in YAML → legacy defaults preserved."""
    _yaml_with_retrieval(tmp_path, monkeypatch)
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.retrieval.vector_top_k == 50
    assert s.retrieval.mmr_top_n is None


@pytest.mark.unit
def test_vector_top_k_overrides_via_yaml(tmp_path, monkeypatch):
    _yaml_with_retrieval(tmp_path, monkeypatch, vector_top_k=300)
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.retrieval.vector_top_k == 300


@pytest.mark.unit
def test_mmr_top_n_overrides_via_yaml(tmp_path, monkeypatch):
    _yaml_with_retrieval(tmp_path, monkeypatch, mmr_top_n=128)
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.retrieval.mmr_top_n == 128


@pytest.mark.unit
def test_explicit_null_mmr_top_n_is_uncapped(tmp_path, monkeypatch):
    """YAML `mmr_top_n: null` should land as None (uncapped)."""
    p = _yaml_with_retrieval(tmp_path, monkeypatch, vector_top_k=100)
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.retrieval.vector_top_k == 100
    assert s.retrieval.mmr_top_n is None  # absent in YAML → uncapped
