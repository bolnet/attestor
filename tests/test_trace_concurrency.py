"""Stress test — concurrent writes to the trace JSONL.

The smoke runner uses ``parallel=N`` (default 2-4) which drives N
concurrent threads calling ``traced_create()``. Each thread emits
events into the SAME JSONL file. We must guarantee:

  1. No corrupted lines (every line is valid JSON)
  2. No lost events (count matches what was emitted)
  3. No interleaved partial writes within a line
  4. The lock around _FH + _open_file() prevents double-open leaks

If this test ever flakes, the trace pipeline has a race.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest


@pytest.mark.unit
def test_concurrent_event_writes_produce_clean_jsonl(tmp_path, monkeypatch):
    log = tmp_path / "concurrent.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log))
    from attestor import trace as _tr
    _tr.reset_for_test()

    N_THREADS = 16
    N_EVENTS_PER_THREAD = 200

    def writer(thread_id: int) -> None:
        for i in range(N_EVENTS_PER_THREAD):
            _tr.event(
                "stress.write",
                thread=thread_id,
                seq=i,
                # Deliberately include a string with quotes + commas to
                # catch any "I forgot to JSON-escape" race.
                payload=f'value, with "quotes", thread {thread_id}',
            )

    threads = [
        threading.Thread(target=writer, args=(t,)) for t in range(N_THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 1. Read the file. Every line should parse as JSON.
    raw = log.read_text().splitlines()
    expected = N_THREADS * N_EVENTS_PER_THREAD
    assert len(raw) == expected, (
        f"event count mismatch: got {len(raw)}, expected {expected}"
    )

    # 2. Each line should parse cleanly — no partial writes.
    seen = set()
    for line in raw:
        e = json.loads(line)   # raises on corruption
        assert e["event"] == "stress.write"
        seen.add((e["thread"], e["seq"]))

    # 3. Every (thread, seq) pair should appear exactly once.
    expected_pairs = {
        (t, i) for t in range(N_THREADS) for i in range(N_EVENTS_PER_THREAD)
    }
    assert seen == expected_pairs, (
        f"missing or duplicate events: "
        f"missing={expected_pairs - seen}, extra={seen - expected_pairs}"
    )


@pytest.mark.unit
def test_concurrent_traced_create_emits_one_event_per_call(
    tmp_path, monkeypatch,
):
    """Same harness, but driven through ``traced_create`` so we also
    cover the YAML-override fast path under contention."""
    log = tmp_path / "concurrent_chat.jsonl"
    monkeypatch.setenv("ATTESTOR_TRACE", "1")
    monkeypatch.setenv("ATTESTOR_TRACE_FILE", str(log))
    from attestor import trace as _tr
    _tr.reset_for_test()

    from unittest.mock import MagicMock

    fake_resp = MagicMock()
    fake_resp.id = "gen-stress"
    fake_resp.model = "openai/test"
    fake_resp.usage = {
        "prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15,
        "completion_tokens_details": {"reasoning_tokens": 0},
        "prompt_tokens_details": {"cached_tokens": 0},
    }

    def make_client():
        c = MagicMock()
        c.chat.completions.create.return_value = fake_resp
        return c

    N_THREADS = 8
    N_CALLS = 100

    def caller(role: str) -> None:
        from attestor.llm_trace import traced_create
        c = make_client()
        for _ in range(N_CALLS):
            traced_create(c, role=role, model="x", max_tokens=50,
                          messages=[{"role": "user", "content": "x"}])

    threads = [
        threading.Thread(target=caller, args=(f"role-{i}",))
        for i in range(N_THREADS)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Each call → one chat.completion event.
    raw = log.read_text().splitlines()
    chat_events = [json.loads(l) for l in raw if json.loads(l)["event"] == "chat.completion"]
    expected = N_THREADS * N_CALLS
    assert len(chat_events) == expected, (
        f"chat.completion count mismatch: got {len(chat_events)}, expected {expected}"
    )

    # Per-role counts should be exact.
    from collections import Counter
    by_role = Counter(e["role"] for e in chat_events)
    assert all(c == N_CALLS for c in by_role.values()), by_role
