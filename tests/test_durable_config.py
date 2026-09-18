"""Durable (Temporal) config — YAML block parsing, defaults, validation, env.

The ``durable:`` block is top-level in ``configs/attestor.yaml`` (a peer
of ``stack:``), opt-in (``enabled: false`` by default), and YAML-
authoritative. ``TEMPORAL_*`` env vars pass through as overrides at
runtime (``attestor.durable.config.resolve_durable``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
ATTESTOR_YAML = REPO_ROOT / "configs" / "attestor.yaml"


def _minimal_stack() -> dict:
    return {
        "postgres": {"url": "postgresql://postgres@localhost/attestor"},
        "neo4j": {
            "url": "bolt://localhost:7687",
            "auth": {"username": "neo4j", "password": "test"},
            "database": "neo4j",
        },
        "embedder": {"provider": "voyage", "model": "voyage-4", "dimensions": 1024},
        "llm": {"provider": "openrouter"},
        "models": {
            "answerer": "m", "judge": "m", "extraction": "m", "distill": "m",
            "verifier": "m", "planner": "m", "benchmark_default": "m",
        },
        "budget": 4000,
        "parallel": 2,
    }


@pytest.fixture
def yaml_factory(tmp_path):
    def _make(durable: dict | None) -> Path:
        cfg: dict = {"stack": _minimal_stack()}
        if durable is not None:
            cfg["durable"] = durable
        p = tmp_path / "attestor.yaml"
        p.write_text(yaml.safe_dump(cfg))
        return p
    return _make


@pytest.mark.unit
def test_durable_block_absent_uses_disabled_defaults(yaml_factory):
    from attestor.config import DurableCfg, load_stack

    stack = load_stack(yaml_factory(None))
    assert isinstance(stack.durable, DurableCfg)
    assert stack.durable.enabled is False
    assert stack.durable.address == "localhost:7233"
    assert stack.durable.namespace == "attestor"
    assert stack.durable.task_queue == "attestor-governance"
    assert stack.durable.tls is False
    assert stack.durable.schedules.retention_sweep == "0 3 * * *"
    assert stack.durable.schedules.session_sweep == "*/10 * * * *"


@pytest.mark.unit
def test_durable_block_parsed_from_yaml(yaml_factory):
    from attestor.config import load_stack

    stack = load_stack(yaml_factory({
        "enabled": True,
        "address": "temporal.internal:7233",
        "namespace": "prod",
        "task_queue": "gov",
        "tls": True,
        "schedules": {"retention_sweep": "0 4 * * *", "session_sweep": "*/5 * * * *"},
    }))
    d = stack.durable
    assert d.enabled is True
    assert d.address == "temporal.internal:7233"
    assert d.namespace == "prod"
    assert d.task_queue == "gov"
    assert d.tls is True
    assert d.schedules.retention_sweep == "0 4 * * *"
    assert d.schedules.session_sweep == "*/5 * * * *"


@pytest.mark.unit
@pytest.mark.parametrize(
    "bad",
    [
        {"address": ""},
        {"namespace": ""},
        {"task_queue": ""},
        {"schedules": {"retention_sweep": "not a cron"}},
        {"schedules": {"session_sweep": "* * *"}},
    ],
)
def test_durable_block_validation_fails_loudly(yaml_factory, bad):
    from attestor.config import load_stack

    with pytest.raises(SystemExit, match=r"\[attestor\.config\] durable"):
        load_stack(yaml_factory(bad))


@pytest.mark.unit
def test_durable_cfg_is_frozen():
    import dataclasses

    from attestor.config import DurableCfg

    cfg = DurableCfg()
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.enabled = True  # type: ignore[misc]


@pytest.mark.unit
def test_repo_yaml_ships_durable_disabled():
    """Quickstart stays zero-question: the shipped YAML has durable off."""
    raw = yaml.safe_load(ATTESTOR_YAML.read_text())
    assert "durable" in raw, "configs/attestor.yaml must carry a durable: block"
    assert raw["durable"]["enabled"] is False
    assert raw["durable"]["task_queue"] == "attestor-governance"
    assert set(raw["durable"]["schedules"]) == {"retention_sweep", "session_sweep"}


@pytest.mark.unit
def test_resolve_durable_applies_temporal_env_overrides():
    from attestor.config import DurableCfg
    from attestor.durable.config import resolve_durable

    base = DurableCfg(enabled=True)
    env = {
        "TEMPORAL_ADDRESS": "10.0.0.5:7233",
        "TEMPORAL_NAMESPACE": "ns2",
        "TEMPORAL_TASK_QUEUE": "tq2",
        "TEMPORAL_TLS": "true",
    }
    out = resolve_durable(base, env=env)
    assert out is not base
    assert base.address == "localhost:7233"  # original untouched
    assert out.address == "10.0.0.5:7233"
    assert out.namespace == "ns2"
    assert out.task_queue == "tq2"
    assert out.tls is True
    assert out.enabled is True


@pytest.mark.unit
def test_resolve_durable_without_env_returns_equal_config():
    from attestor.config import DurableCfg
    from attestor.durable.config import resolve_durable

    base = DurableCfg(enabled=False, address="a:1")
    assert resolve_durable(base, env={}) == base


@pytest.mark.unit
def test_resolve_durable_rejects_bad_tls_env():
    from attestor.config import DurableCfg
    from attestor.durable.config import resolve_durable

    with pytest.raises(ValueError, match="TEMPORAL_TLS"):
        resolve_durable(DurableCfg(), env={"TEMPORAL_TLS": "maybe"})


@pytest.mark.unit
def test_require_enabled_raises_when_disabled():
    from attestor.config import DurableCfg
    from attestor.durable.config import DurableDisabledError, require_enabled

    with pytest.raises(DurableDisabledError, match="durable.enabled"):
        require_enabled(DurableCfg(enabled=False))
    cfg = DurableCfg(enabled=True)
    assert require_enabled(cfg) is cfg


@pytest.mark.unit
def test_require_temporalio_raises_actionable_error_when_missing(monkeypatch):
    import builtins

    from attestor.durable.config import DurableUnavailableError, require_temporalio

    real_import = builtins.__import__

    def fake_import(name, *a, **kw):
        if name == "temporalio" or name.startswith("temporalio."):
            raise ImportError("No module named 'temporalio'")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(DurableUnavailableError, match=r"attestor\[durable\]"):
        require_temporalio()
