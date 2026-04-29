"""Per-call LLM tracing — emits chat.completion events + applies YAML
overrides for max_tokens / reasoning_effort at every call site."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def fake_response():
    """Synthesize an OpenAI ChatCompletion response shape."""
    resp = MagicMock()
    resp.id = "gen-test-123"
    resp.model = "openai/gpt-5.5-20260417"
    resp.usage = {
        "prompt_tokens": 250,
        "completion_tokens": 80,
        "total_tokens": 330,
        "completion_tokens_details": {"reasoning_tokens": 1500},
        "prompt_tokens_details": {"cached_tokens": 200},
    }
    return resp


@pytest.fixture
def yaml_with_role(tmp_path, monkeypatch):
    def _make(*, max_tokens: dict, reasoning_effort: dict | None = None) -> Path:
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
                    "max_tokens": max_tokens,
                    **({"reasoning_effort": reasoning_effort} if reasoning_effort else {}),
                },
            },
        }
        p = tmp_path / "attestor.yaml"
        p.write_text(yaml.safe_dump(cfg))
        monkeypatch.setenv("ATTESTOR_CONFIG", str(p))
        from attestor import config as _c
        _c.reset_stack()
        return p
    return _make


@pytest.mark.unit
def test_traced_create_emits_chat_completion_event(
    monkeypatch, capsys, fake_response, yaml_with_role,
):
    """Every LLM call should produce a chat.completion trace event with
    full usage breakdown."""
    yaml_with_role(max_tokens={"answerer": 32000})
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    from attestor import trace as _tr
    _tr.reset_for_test()

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    from attestor.llm_trace import traced_create
    response = traced_create(
        fake_client,
        role="answerer",
        model="openai/gpt-5.4-mini",
        max_tokens=300,        # caller's value — should be overridden by YAML
        messages=[{"role": "user", "content": "hi"}],
    )

    assert response is fake_response
    err = capsys.readouterr().err
    assert "[trace] chat.completion" in err
    assert "role='answerer'" in err
    assert "prompt_tokens=250" in err
    assert "completion_tokens=80" in err
    assert "reasoning_tokens=1500" in err
    assert "cached_tokens=200" in err


@pytest.mark.unit
def test_yaml_max_tokens_overrides_caller_value(
    monkeypatch, fake_response, yaml_with_role,
):
    """Caller passes max_tokens=400; YAML configures 32000 for the
    role. The actual API call should receive 32000."""
    yaml_with_role(max_tokens={"planner": 32000})

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    from attestor.llm_trace import traced_create
    traced_create(
        fake_client,
        role="planner",
        model="anthropic/claude-opus-4.7",
        max_tokens=400,        # legacy hardcode in planner.py
        messages=[{"role": "user", "content": "rewrite"}],
    )

    # Inspect what was actually passed to the API
    fake_client.chat.completions.create.assert_called_once()
    actual_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert actual_kwargs["max_tokens"] == 32000   # YAML wins


@pytest.mark.unit
def test_caller_max_tokens_used_when_yaml_lacks_role(
    monkeypatch, fake_response, yaml_with_role,
):
    """No YAML entry for the role → caller's max_tokens passes through."""
    yaml_with_role(max_tokens={"answerer": 32000})  # only answerer set

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    from attestor.llm_trace import traced_create
    traced_create(
        fake_client,
        role="some_unconfigured_role",
        model="openai/gpt-5.4-mini",
        max_tokens=500,
        messages=[{"role": "user", "content": "x"}],
    )

    actual_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert actual_kwargs["max_tokens"] == 500   # caller value preserved


@pytest.mark.unit
def test_yaml_reasoning_effort_added_when_caller_omits(
    monkeypatch, fake_response, yaml_with_role,
):
    """YAML configures reasoning_effort:high for answerer; caller doesn't
    pass it. The kwarg should be added to the API call."""
    yaml_with_role(
        max_tokens={"answerer": 32000},
        reasoning_effort={"answerer": "high"},
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    from attestor.llm_trace import traced_create
    traced_create(
        fake_client,
        role="answerer",
        model="openai/gpt-5.4-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": "synth"}],
    )

    actual_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert actual_kwargs["reasoning_effort"] == "high"


@pytest.mark.unit
def test_caller_reasoning_effort_wins_over_yaml(
    monkeypatch, fake_response, yaml_with_role,
):
    """If caller explicitly passes reasoning_effort, that wins (escape
    hatch for ad-hoc overrides)."""
    yaml_with_role(
        max_tokens={"answerer": 32000},
        reasoning_effort={"answerer": "high"},
    )

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    from attestor.llm_trace import traced_create
    traced_create(
        fake_client,
        role="answerer",
        model="openai/gpt-5.4-mini",
        max_tokens=300,
        reasoning_effort="minimal",   # caller override
        messages=[{"role": "user", "content": "x"}],
    )

    actual_kwargs = fake_client.chat.completions.create.call_args.kwargs
    assert actual_kwargs["reasoning_effort"] == "minimal"


@pytest.mark.unit
def test_jsonl_event_carries_full_usage_block(
    monkeypatch, tmp_path, fake_response, yaml_with_role,
):
    """The JSONL file (when ATTESTOR_TRACE_FILE is set) should record
    every field of the usage block for post-run analysis."""
    yaml_with_role(max_tokens={"answerer": 32000})
    log = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log))
    from attestor import trace as _tr
    _tr.reset_for_test()

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    from attestor.llm_trace import traced_create
    traced_create(
        fake_client,
        role="answerer",
        model="openai/gpt-5.4-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": "x"}],
    )

    line = log.read_text().splitlines()[-1]
    e = json.loads(line)
    assert e["event"] == "chat.completion"
    assert e["role"] == "answerer"
    assert e["prompt_tokens"] == 250
    assert e["completion_tokens"] == 80
    assert e["reasoning_tokens"] == 1500
    assert e["cached_tokens"] == 200
    assert e["total_tokens"] == 330
    assert e["request_id"] == "gen-test-123"
    assert "latency_ms" in e


@pytest.mark.unit
def test_traced_create_does_not_break_when_tracing_disabled(
    monkeypatch, fake_response, yaml_with_role,
):
    """ATTESTOR_TRACE unset → traced_create still applies YAML overrides
    and returns the response, just doesn't emit events."""
    yaml_with_role(max_tokens={"answerer": 32000})
    monkeypatch.delenv("ATTESTOR_TRACE", raising=False)
    from attestor import trace as _tr
    _tr.reset_for_test()

    fake_client = MagicMock()
    fake_client.chat.completions.create.return_value = fake_response

    from attestor.llm_trace import traced_create
    response = traced_create(
        fake_client,
        role="answerer",
        model="openai/gpt-5.4-mini",
        max_tokens=300,
        messages=[{"role": "user", "content": "x"}],
    )

    assert response is fake_response
    actual_kwargs = fake_client.chat.completions.create.call_args.kwargs
    # YAML override still applies even without trace enabled
    assert actual_kwargs["max_tokens"] == 32000
