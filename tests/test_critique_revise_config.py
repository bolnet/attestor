"""YAML loader integration tests for the critique-revise feature.

Mirrors `tests/test_self_consistency_config.py` (the closest sibling).
"""

from __future__ import annotations


import pytest


_BASE_YAML = """
stack:
  postgres:
    url: postgresql://x/y
  neo4j:
    url: bolt://localhost:7687
    auth: { username: neo4j, password: pw }
    database: neo4j
  embedder:
    provider: voyage
    model: voyage-4
    dimensions: 1024
  models:
    answerer: x
    judge: x
    extraction: x
    distill: x
    verifier: x
    planner: x
    benchmark_default: x
  llm:
    provider: openrouter
  budget: 4000
  parallel: 2
"""


@pytest.mark.unit
def test_yaml_loader_parses_critique_revise_block(tmp_path) -> None:
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(_BASE_YAML + """
  critique_revise:
    enabled: true
    critic_model: anthropic/claude-sonnet-4-6
    revise_model: openai/gpt-5.4-mini
    max_revisions: 1
""")

    stack = load_stack(yaml_path)
    cr = stack.critique_revise
    assert cr.enabled is True
    assert cr.critic_model == "anthropic/claude-sonnet-4-6"
    assert cr.revise_model == "openai/gpt-5.4-mini"
    assert cr.max_revisions == 1


@pytest.mark.unit
def test_yaml_loader_defaults_when_block_omitted(tmp_path) -> None:
    """When `stack.critique_revise` is missing entirely, defaults apply."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(_BASE_YAML)
    stack = load_stack(yaml_path)
    cr = stack.critique_revise
    assert cr.enabled is False
    assert cr.critic_model is None
    assert cr.revise_model is None
    assert cr.max_revisions == 1


@pytest.mark.unit
def test_yaml_loader_rejects_max_revisions_above_one(tmp_path) -> None:
    """The PR-E hard cap. > 1 should fail loud at load time so a typo
    in YAML can't quietly produce 5x answerer cost."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(_BASE_YAML + """
  critique_revise:
    enabled: true
    max_revisions: 3
""")

    with pytest.raises(SystemExit) as exc:
        load_stack(yaml_path)
    assert "max_revisions" in str(exc.value).lower()


@pytest.mark.unit
def test_yaml_loader_accepts_max_revisions_one_explicitly(tmp_path) -> None:
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(_BASE_YAML + """
  critique_revise:
    enabled: false
    max_revisions: 1
""")
    stack = load_stack(yaml_path)
    assert stack.critique_revise.max_revisions == 1


@pytest.mark.unit
def test_critique_revise_is_top_level_not_in_retrieval(tmp_path) -> None:
    """Sanity: critique_revise is on StackConfig, not RetrievalCfg.
    Putting it under retrieval would be a category error since this
    feature is answerer-side, not retrieval-side."""
    from attestor.config import load_stack

    yaml_path = tmp_path / "test_attestor.yaml"
    yaml_path.write_text(_BASE_YAML)
    stack = load_stack(yaml_path)
    assert hasattr(stack, "critique_revise")
    assert not hasattr(stack.retrieval, "critique_revise")
