"""Self-consistency YAML loader tests (Phase 3 PR-B).

Confirms the top-level ``stack.self_consistency`` block lands on
``StackConfig.self_consistency``, defaults are applied when the block
is omitted, and an invalid ``voter`` value fails loudly.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _yaml_with(tmp_path, monkeypatch, **stack_extra: Any) -> Path:
    """Write a minimal valid YAML to ``tmp_path`` and point the loader at it."""
    cfg: Dict[str, Any] = {
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
            **stack_extra,
        },
    }
    p = tmp_path / "attestor.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ATTESTOR_CONFIG", str(p))
    from attestor import config as _c
    _c.reset_stack()
    return p


@pytest.mark.unit
def test_self_consistency_defaults_when_block_omitted(tmp_path, monkeypatch):
    """No `self_consistency` block → safe defaults: disabled, k=5, t=0.7,
    voter='majority', judge_model=None."""
    _yaml_with(tmp_path, monkeypatch)
    from attestor.config import get_stack
    s = get_stack(strict=False)
    sc = s.self_consistency
    assert sc.enabled is False
    assert sc.k == 5
    assert sc.temperature == pytest.approx(0.7)
    assert sc.voter == "majority"
    assert sc.judge_model is None


@pytest.mark.unit
def test_self_consistency_block_is_parsed(tmp_path, monkeypatch):
    _yaml_with(
        tmp_path, monkeypatch,
        self_consistency={
            "enabled": True,
            "k": 7,
            "temperature": 0.5,
            "voter": "judge_pick",
            "judge_model": "openai/gpt-5.5",
        },
    )
    from attestor.config import get_stack
    s = get_stack(strict=False)
    sc = s.self_consistency
    assert sc.enabled is True
    assert sc.k == 7
    assert sc.temperature == pytest.approx(0.5)
    assert sc.voter == "judge_pick"
    assert sc.judge_model == "openai/gpt-5.5"


@pytest.mark.unit
def test_self_consistency_invalid_voter_raises(tmp_path, monkeypatch):
    """Loader must reject unknown voter strategies with a clear message."""
    _yaml_with(
        tmp_path, monkeypatch,
        self_consistency={"enabled": True, "voter": "coin_flip"},
    )
    from attestor.config import load_stack
    with pytest.raises(SystemExit, match="voter"):
        load_stack(strict=False)


@pytest.mark.unit
def test_self_consistency_majority_voter_accepted(tmp_path, monkeypatch):
    """Both supported voters parse cleanly."""
    for voter in ("majority", "judge_pick"):
        _yaml_with(
            tmp_path, monkeypatch,
            self_consistency={"enabled": True, "voter": voter},
        )
        from attestor.config import get_stack
        s = get_stack(strict=False)
        assert s.self_consistency.voter == voter


@pytest.mark.unit
def test_self_consistency_partial_block_uses_defaults_for_missing(tmp_path, monkeypatch):
    """Only `enabled: true` set → other fields take their defaults."""
    _yaml_with(
        tmp_path, monkeypatch,
        self_consistency={"enabled": True},
    )
    from attestor.config import get_stack
    s = get_stack(strict=False)
    sc = s.self_consistency
    assert sc.enabled is True
    assert sc.k == 5
    assert sc.temperature == pytest.approx(0.7)
    assert sc.voter == "majority"
    assert sc.judge_model is None


@pytest.mark.unit
def test_self_consistency_no_yaml_falls_back_to_defaults(monkeypatch):
    """Stripped checkout / no YAML → fallback stack still has a
    valid SelfConsistencyCfg (disabled by default)."""
    monkeypatch.setenv("ATTESTOR_CONFIG", "/tmp/nonexistent_attestor.yaml")
    from attestor import config as _c
    _c.reset_stack()
    s = _c.get_stack(strict=False)
    assert s.self_consistency.enabled is False
    assert s.self_consistency.k == 5
    assert s.self_consistency.voter == "majority"
