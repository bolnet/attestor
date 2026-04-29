"""Per-role reasoning_effort + max_tokens — config + chat-kwargs builder.

Confirms the YAML knobs land on `ModelsCfg`, the kwargs builder
returns the right dict per role, and missing roles fall through to
the legacy default (300 max_tokens, no reasoning_effort).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _yaml_with_models(tmp_path, monkeypatch, **models_extra) -> Path:
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
                **models_extra,
            },
        },
    }
    p = tmp_path / "attestor.yaml"
    p.write_text(yaml.safe_dump(cfg))
    monkeypatch.setenv("ATTESTOR_CONFIG", str(p))
    from attestor import config as _c
    _c.reset_stack()
    return p


@pytest.mark.unit
def test_default_reasoning_and_max_tokens_are_empty(tmp_path, monkeypatch):
    """No `reasoning_effort` / `max_tokens` block → empty dicts.
    Production behavior preserved (back-compat)."""
    _yaml_with_models(tmp_path, monkeypatch)
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.models.reasoning_effort == {}
    assert s.models.max_tokens == {}


@pytest.mark.unit
def test_reasoning_effort_block_is_parsed(tmp_path, monkeypatch):
    _yaml_with_models(
        tmp_path, monkeypatch,
        reasoning_effort={"answerer": "high", "judge": "high", "extraction": "low"},
    )
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.models.reasoning_effort == {
        "answerer": "high", "judge": "high", "extraction": "low",
    }


@pytest.mark.unit
def test_max_tokens_block_is_parsed_as_ints(tmp_path, monkeypatch):
    _yaml_with_models(
        tmp_path, monkeypatch,
        max_tokens={"answerer": 3000, "judge": "1000", "verifier": 500},
    )
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.models.max_tokens == {"answerer": 3000, "judge": 1000, "verifier": 500}


@pytest.mark.unit
def test_chat_kwargs_for_role_returns_legacy_default_when_unset(tmp_path, monkeypatch):
    """No YAML override for the role → just max_tokens=fallback."""
    _yaml_with_models(tmp_path, monkeypatch)
    from attestor.config import chat_kwargs_for_role
    out = chat_kwargs_for_role("answerer", fallback_max_tokens=300)
    assert out == {"max_tokens": 300}


@pytest.mark.unit
def test_chat_kwargs_for_role_picks_up_yaml_overrides(tmp_path, monkeypatch):
    _yaml_with_models(
        tmp_path, monkeypatch,
        reasoning_effort={"answerer": "high"},
        max_tokens={"answerer": 3000},
    )
    from attestor.config import chat_kwargs_for_role
    out = chat_kwargs_for_role("answerer", fallback_max_tokens=300)
    assert out == {"max_tokens": 3000, "reasoning_effort": "high"}


@pytest.mark.unit
def test_chat_kwargs_for_role_only_max_tokens_set(tmp_path, monkeypatch):
    """max_tokens override without reasoning_effort — common for
    non-reasoning roles like extraction or planner."""
    _yaml_with_models(
        tmp_path, monkeypatch,
        max_tokens={"extraction": 2000},
    )
    from attestor.config import chat_kwargs_for_role
    out = chat_kwargs_for_role("extraction", fallback_max_tokens=300)
    assert out == {"max_tokens": 2000}


@pytest.mark.unit
def test_chat_kwargs_for_role_handles_unknown_role(tmp_path, monkeypatch):
    """Roles not present in either dict get the fallback."""
    _yaml_with_models(
        tmp_path, monkeypatch,
        reasoning_effort={"answerer": "high"},
        max_tokens={"answerer": 3000},
    )
    from attestor.config import chat_kwargs_for_role
    out = chat_kwargs_for_role("nonexistent_role", fallback_max_tokens=500)
    assert out == {"max_tokens": 500}


@pytest.mark.unit
def test_chat_kwargs_for_role_survives_no_yaml(monkeypatch):
    """Stripped checkout / no YAML — return safe defaults, don't crash."""
    monkeypatch.setenv("ATTESTOR_CONFIG", "/tmp/nonexistent.yaml")
    from attestor import config as _c
    _c.reset_stack()
    out = _c.chat_kwargs_for_role("answerer", fallback_max_tokens=300)
    assert out == {"max_tokens": 300}
