"""LLM provider routing — config + runtime client wiring.

Confirms the new ``stack.llm.provider`` knob actually flips the base
URL + API-key env on the OpenAI client. Two providers wired today:
``openrouter`` (default; matches the canonical bench stack) and
``openai`` (direct, no per-call markup).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml


# Make sure the repo root is on the path so `import attestor` works
# regardless of how pytest is invoked.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def yaml_with_llm(tmp_path, monkeypatch):
    """Factory for a tmp YAML stack with the LLM block set."""

    def _make(provider: str, *, base_url: str | None = None,
              api_key_env: str | None = None) -> Path:
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
                "llm": {"provider": provider},
                "models": {
                    "answerer": "openai/gpt-4.1",
                    "judge": "openai/gpt-4.1",
                    "extraction": "openai/gpt-4.1-mini",
                    "distill": "openai/gpt-4.1-mini",
                    "verifier": "anthropic/claude-sonnet-4-6",
                    "planner": "anthropic/claude-opus-4.7",
                    "benchmark_default": "openai/gpt-4.1-mini",
                },
                "budget": 4000,
                "parallel": 2,
            },
        }
        if base_url is not None:
            cfg["stack"]["llm"]["base_url"] = base_url
        if api_key_env is not None:
            cfg["stack"]["llm"]["api_key_env"] = api_key_env

        path = tmp_path / "attestor.yaml"
        path.write_text(yaml.safe_dump(cfg))
        # Test isolation: force the loader to re-read this file every test
        monkeypatch.setenv("ATTESTOR_CONFIG", str(path))
        from attestor import config as _c
        _c.reset_stack()
        from attestor.llm_trace import _reset_client_pool
        _reset_client_pool()
        return path

    return _make


# ---------------------------------------------------------------------------
# YAML parsing
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_default_llm_provider_is_openrouter(yaml_with_llm):
    yaml_with_llm("openrouter")
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.llm.provider == "openrouter"
    assert s.llm.base_url == "https://openrouter.ai/api/v1"
    assert s.llm.api_key_env == "OPENROUTER_API_KEY"


@pytest.mark.unit
def test_openai_provider_uses_native_base_and_key(yaml_with_llm):
    yaml_with_llm("openai")
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.llm.provider == "openai"
    assert s.llm.base_url == "https://api.openai.com/v1"
    assert s.llm.api_key_env == "OPENAI_API_KEY"


@pytest.mark.unit
def test_unknown_provider_raises_at_load(yaml_with_llm):
    yaml_with_llm("anthropic-direct")  # not wired
    from attestor.config import get_stack
    with pytest.raises(SystemExit, match="unknown LLM provider"):
        get_stack(strict=False)


@pytest.mark.unit
def test_yaml_overrides_take_precedence(yaml_with_llm):
    """Pointing base_url at a local OpenAI-compatible endpoint via YAML
    should round-trip."""
    yaml_with_llm(
        "openai",
        base_url="http://localhost:11434/v1",
        api_key_env="LOCAL_OPENAI_COMPAT_TOKEN",
    )
    from attestor.config import get_stack
    s = get_stack(strict=False)
    assert s.llm.base_url == "http://localhost:11434/v1"
    assert s.llm.api_key_env == "LOCAL_OPENAI_COMPAT_TOKEN"


# ---------------------------------------------------------------------------
# Runtime client construction
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_get_client_uses_openrouter_by_default(yaml_with_llm, monkeypatch):
    yaml_with_llm("openrouter")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-test")
    monkeypatch.delenv("LME_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    from attestor.longmemeval import _get_client
    client = _get_client()
    assert "openrouter.ai" in str(client.base_url)


@pytest.mark.unit
def test_get_client_routes_to_openai_when_provider_openai(yaml_with_llm, monkeypatch):
    yaml_with_llm("openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LME_LLM_BASE_URL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from attestor.longmemeval import _get_client
    client = _get_client()
    assert "api.openai.com" in str(client.base_url)


@pytest.mark.unit
def test_get_client_env_base_url_overrides_yaml(yaml_with_llm, monkeypatch):
    """LME_LLM_BASE_URL env wins over YAML — preserves the ad-hoc local
    OpenAI-compatible escape hatch even when YAML pins openai."""
    yaml_with_llm("openai")
    monkeypatch.setenv("LME_LLM_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from attestor.longmemeval import _get_client
    client = _get_client()
    assert "localhost:11434" in str(client.base_url)


@pytest.mark.unit
def test_get_client_missing_key_names_the_specific_env(yaml_with_llm, monkeypatch):
    """Error must name the YAML-configured env var, not 'OPENROUTER_API_KEY'
    when the provider is openai. Saves debugging time."""
    yaml_with_llm("openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("LME_LLM_BASE_URL", raising=False)

    from attestor.longmemeval import _get_client
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        _get_client()


@pytest.mark.unit
def test_get_client_local_openai_compat_works_keyless(yaml_with_llm, monkeypatch):
    """Localhost OpenAI-compatible base URL ignores the missing key
    (back-compat path for local LLM servers stays intact)."""
    yaml_with_llm("openai")
    monkeypatch.setenv("LME_LLM_BASE_URL", "http://127.0.0.1:11434/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    from attestor.longmemeval import _get_client
    client = _get_client()  # must not raise
    assert "127.0.0.1" in str(client.base_url)
