"""Unit tests for the env-gated pipeline tracer."""

from __future__ import annotations

import json

import pytest

import attestor.trace as tr


@pytest.fixture(autouse=True)
def _reset_trace_env(monkeypatch, tmp_path):
    """Each test starts with a known-clean tracer state."""
    monkeypatch.delenv("ATTESTOR_TRACE", raising=False)
    monkeypatch.delenv("ATTESTOR_TRACE_FILE", raising=False)
    tr.reset_for_test()
    yield
    tr.reset_for_test()


@pytest.mark.unit
def test_disabled_by_default(capsys):
    assert tr.is_enabled() is False
    tr.event("ingest.embed", model="voyage-4", dim=1024)
    captured = capsys.readouterr()
    # When disabled, NO output anywhere.
    assert captured.err == ""
    assert captured.out == ""


@pytest.mark.unit
def test_enabled_writes_to_stderr(capsys, monkeypatch):
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    tr.reset_for_test()
    assert tr.is_enabled() is True

    tr.event("ingest.embed", model="voyage-4", dim=1024, latency_ms=42)

    captured = capsys.readouterr()
    assert "[trace] ingest.embed" in captured.err
    assert "model='voyage-4'" in captured.err
    assert "dim=1024" in captured.err


@pytest.mark.unit
def test_enabled_writes_to_jsonl(capsys, monkeypatch, tmp_path):
    log = tmp_path / "trace.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log))
    tr.reset_for_test()

    tr.event("recall.start", query="hello", namespace="ns-a", token_budget=2000)
    tr.event("recall.done", final_count=3, latency_ms=15.5)

    lines = log.read_text().splitlines()
    assert len(lines) == 2
    e1 = json.loads(lines[0])
    e2 = json.loads(lines[1])
    assert e1["event"] == "recall.start"
    assert e1["query"] == "hello"
    assert e1["namespace"] == "ns-a"
    assert e2["event"] == "recall.done"
    assert e2["final_count"] == 3


@pytest.mark.unit
def test_secrets_are_redacted_in_event_fields(capsys, monkeypatch):
    """If a trace field accidentally carries a key-shaped string we still
    scrub it. The cleanest fix is "don't pass secrets" -- this is belt
    and suspenders against an authorization header sneaking into a
    trace field via e.g. a request-dump."""
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    tr.reset_for_test()

    tr.event(
        "demo",
        # Made-up key shapes that match the patterns. None of these are
        # real keys; the test asserts we redact based on prefix shape.
        openai_key="sk-proj-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        voyage_key="pa-AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",
        bearer_header="Bearer sk-or-v1-AAAAAAAAAAAAAAAAAAAAAA",
    )

    err = capsys.readouterr().err
    assert "sk-proj-A" not in err
    assert "pa-A" not in err
    assert "<REDACTED:openai>" in err
    assert "<REDACTED:voyage>" in err
    # NB: "Bearer ..." gets the openrouter prefix replaced inside
    assert "<REDACTED:openrouter>" in err
