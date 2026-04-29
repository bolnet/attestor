"""Phase 3.6 — ConversationIngest end-to-end with stub LLMs.

Verifies the full Phase 3 pipeline against a real Postgres v4 schema
plus stubbed extraction + resolver clients (so we don't burn API
credits in CI):

  1. ingest_round writes verbatim episode
  2. user_facts + agent_facts both extracted (one prompt per side)
  3. resolve_conflicts decides ADD/UPDATE/INVALIDATE/NOOP per fact
  4. apply persists with full provenance (source_episode_id, source_span,
     extraction_model, agent_id)
  5. INVALIDATE supersedes the old row but keeps it (timeline replay)
  6. Public AgentMemory.ingest_round() resolves identity from SOLO defaults
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, List
from urllib.parse import urlparse, urlunparse

import pytest

try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

from attestor.conversation import (
    ConversationIngest,
    ConversationTurn,
    IngestConfig,
)
from attestor.extraction.round_extractor import ExtractedFact

PG_URL = os.environ.get(
    "PG_TEST_URL",
    "postgresql://postgres:attestor@localhost:5432/attestor_v4_test",
)
SCHEMA_PATH = Path(__file__).resolve().parent.parent / "attestor" / "store" / "schema.sql"


def _reachable() -> bool:
    if not HAVE_PSYCOPG2:
        return False
    try:
        c = psycopg2.connect(PG_URL, connect_timeout=2)
        c.close()
        return True
    except Exception:
        return False


pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not _reachable(), reason="local Postgres unreachable"),
]


# ──────────────────────────────────────────────────────────────────────────
# Stub LLM (per-prompt scriptable)
# ──────────────────────────────────────────────────────────────────────────


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]


class _ScriptedChat:
    """Returns a different response per call, in order."""

    def __init__(self, scripted: List[str]) -> None:
        self._scripted = list(scripted)
        self.calls: List[dict] = []

    def create(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(kwargs)
        if not self._scripted:
            return _StubResponse('{"facts": [], "decisions": []}')
        return _StubResponse(self._scripted.pop(0))


class ScriptedClient:
    def __init__(self, scripted: List[str]) -> None:
        self.chat = type("Chat", (), {"completions": _ScriptedChat(scripted)})()

    @property
    def call_count(self) -> int:
        return len(self.chat.completions.calls)


# ──────────────────────────────────────────────────────────────────────────
# Postgres fixtures (admin + non-superuser runtime role)
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def admin_conn():
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    yield c
    c.close()


@pytest.fixture
def fresh_schema(admin_conn):
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "16")
    with admin_conn.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


@pytest.fixture(scope="module")
def app_role(admin_conn):
    role = f"attestor_ingest_{uuid.uuid4().hex[:8]}"
    pw = "test"
    with admin_conn.cursor() as cur:
        cur.execute(f"CREATE ROLE {role} LOGIN NOBYPASSRLS PASSWORD '{pw}'")
        cur.execute(f"GRANT USAGE ON SCHEMA public TO {role}")
        cur.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO {role}"
        )
        cur.execute(
            f"GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO {role}"
        )
        cur.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {role}"
        )
    p = urlparse(PG_URL)
    netloc = f"{role}:{pw}@{p.hostname}{':' + str(p.port) if p.port else ''}"
    yield urlunparse(p._replace(netloc=netloc))
    with admin_conn.cursor() as cur:
        cur.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA public "
            f"REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM {role}"
        )
        cur.execute(f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}")
        cur.execute(f"REVOKE ALL ON SCHEMA public FROM {role}")
        cur.execute(f"DROP ROLE {role}")


def _build_solo_mem(role_url: str, tmp_path: Path):
    from attestor.core import AgentMemory
    p = urlparse(role_url)
    cfg = {
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {
            "postgres": {
                "url": f"postgresql://{p.hostname}:{p.port or 5432}",
                "database": p.path.lstrip("/"),
                "auth": {"username": p.username, "password": p.password},
                "v4": True,
                "skip_schema_init": True,
                "embedding_dim": 16,
            }
        },
    }
    return AgentMemory(tmp_path, config=cfg)


# ──────────────────────────────────────────────────────────────────────────
# Round helpers
# ──────────────────────────────────────────────────────────────────────────


def _round(thread_id: str = "t-1", offset: int = 0):
    base = datetime.now(timezone.utc) + timedelta(seconds=offset)
    user = ConversationTurn(
        thread_id=thread_id, speaker="user", role="user",
        content="I prefer dark mode in all apps.",
        ts=base,
    )
    asst = ConversationTurn(
        thread_id=thread_id, speaker="planner", role="assistant",
        content="Switching all apps to dark mode now.",
        ts=base + timedelta(seconds=1),
    )
    return user, asst


def _user_facts_payload(*texts: str) -> str:
    return json.dumps({
        "facts": [
            {"text": t, "category": "preference", "entity": "user",
             "confidence": 0.9, "source_span": [0, len(t)]}
            for t in texts
        ]
    })


def _agent_facts_payload(*texts: str) -> str:
    return json.dumps({
        "facts": [
            {"text": t, "category": "decision", "entity": "system",
             "confidence": 0.9, "source_span": [0, len(t)]}
            for t in texts
        ]
    })


def _decisions_payload(*ops: tuple):
    """ops: list of (operation, existing_id_or_None) pairs."""
    return json.dumps({
        "decisions": [
            {"operation": op, "existing_id": eid,
             "rationale": f"test-{op}"}
            for op, eid in ops
        ]
    })


# ──────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_ingest_round_writes_episode_and_extracts(app_role, fresh_schema, tmp_path):
    """Happy path: 1 user fact + 1 agent fact, both ADDed."""
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        # Script: user-extract, agent-extract, resolve
        client = ScriptedClient([
            _user_facts_payload("user prefers dark mode"),
            _agent_facts_payload("agent switched apps to dark mode"),
            _decisions_payload(("ADD", None), ("ADD", None)),
        ])
        u, a = _round()
        result = mem.ingest_round(
            u, a,
            extraction_client=client,
            resolver_client=client,
        )

        # Episode written verbatim
        assert result.episode.user_turn_text == u.content
        assert result.episode.assistant_turn_text == a.content

        # Two facts extracted (one user, one agent)
        assert len(result.user_facts) == 1
        assert len(result.agent_facts) == 1

        # Two ADD decisions, both applied successfully
        assert len(result.decisions) == 2
        assert all(d.operation == "ADD" for d in result.decisions)
        assert all(a.operation == "ADD" for a in result.applied)

        # Both written rows carry full provenance
        for mid in result.written_memory_ids:
            row = mem._store.get(mid)
            assert row is not None
            assert row.source_episode_id == result.episode_id
            assert row.source_span is not None
            assert row.extraction_model is not None
    finally:
        mem.close()


@pytest.mark.live
def test_ingest_round_invalidate_supersedes_old(app_role, fresh_schema, tmp_path):
    """Round 2 contradicts round 1 → INVALIDATE + ADD; old kept superseded.

    The doc-store fallback in _retrieve_similar uses (category, entity)
    to find candidates so the resolver fires even with no embedder.
    """
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        # Round 1: write "lives in NYC" with location/user
        # Use a category that survives validation (preference)
        round1_user = json.dumps({
            "facts": [{"text": "user lives in NYC", "category": "location",
                       "entity": "user", "confidence": 0.9, "source_span": [0, 17]}]
        })
        round1_agent = json.dumps({"facts": []})
        client1 = ScriptedClient([round1_user, round1_agent])
        u1, a1 = _round(thread_id="t-loc", offset=0)
        r1 = mem.ingest_round(u1, a1,
                              extraction_client=client1, resolver_client=client1)
        nyc_ids = [
            mid for mid in r1.written_memory_ids
            if mem._store.get(mid).content == "user lives in NYC"
        ]
        assert len(nyc_ids) == 1
        nyc_id = nyc_ids[0]

        # Round 2: contradicts. The doc-store fallback finds the NYC row
        # by (category=location, entity=user); resolver decides INVALIDATE.
        round2_user = json.dumps({
            "facts": [{"text": "user lives in SF", "category": "location",
                       "entity": "user", "confidence": 0.95, "source_span": [0, 16]}]
        })
        round2_agent = json.dumps({"facts": []})
        round2_decisions = json.dumps({
            "decisions": [{"operation": "INVALIDATE", "existing_id": nyc_id,
                           "rationale": "contradicts"}]
        })
        client2 = ScriptedClient([round2_user, round2_agent, round2_decisions])
        u2, a2 = _round(thread_id="t-loc", offset=10)
        r2 = mem.ingest_round(u2, a2,
                              extraction_client=client2, resolver_client=client2)

        # Old NYC row marked superseded but not deleted
        nyc = mem._store.get(nyc_id)
        assert nyc is not None, "NYC row was hard-deleted (timeline broken)"
        assert nyc.status == "superseded", (
            f"expected superseded, got {nyc.status!r}"
        )
        assert nyc.superseded_by is not None

        # The new SF row exists
        sf_applied = next(a for a in r2.applied if a.operation == "INVALIDATE")
        new_id = sf_applied.new_memory_id
        sf = mem._store.get(new_id)
        assert sf is not None
        assert sf.content == "user lives in SF"
    finally:
        mem.close()


@pytest.mark.live
def test_ingest_round_speaker_lock_holds(app_role, fresh_schema, tmp_path):
    """Per-side extractor calls go through separate, speaker-locked prompts.

    The first ingest fires extractors for both sides. The first prompt
    must speak USER lock, the second ASSISTANT lock (resolver call is
    optional — short-circuits to all-ADD on a fresh DB).
    """
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        client = ScriptedClient([
            _user_facts_payload("u1"),
            _agent_facts_payload("a1"),
            _decisions_payload(("ADD", None), ("ADD", None)),  # may be unused
        ])
        u, a = _round()
        mem.ingest_round(u, a, extraction_client=client, resolver_client=client)
        # At minimum: 2 extraction calls. Resolver may or may not fire.
        assert client.call_count >= 2
        prompts = [c["messages"][0]["content"] for c in client.chat.completions.calls]
        assert "USER'S MESSAGE BELOW" in prompts[0]
        assert "ASSISTANT'S MESSAGE BELOW" in prompts[1]
    finally:
        mem.close()


@pytest.mark.live
def test_ingest_round_provenance_complete(app_role, fresh_schema, tmp_path):
    """All written facts have source_episode_id + source_span (roadmap §A.5)."""
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        client = ScriptedClient([
            _user_facts_payload("a", "b"),
            _agent_facts_payload("c"),
            _decisions_payload(("ADD", None), ("ADD", None), ("ADD", None)),
        ])
        u, a = _round()
        result = mem.ingest_round(u, a,
                                  extraction_client=client,
                                  resolver_client=client)
        assert len(result.written_memory_ids) == 3
        for mid in result.written_memory_ids:
            row = mem._store.get(mid)
            assert row.source_episode_id == result.episode_id
            assert row.source_span is not None
            assert isinstance(row.source_span, list)
            assert len(row.source_span) == 2
    finally:
        mem.close()


@pytest.mark.live
def test_ingest_round_extraction_disabled_skips_extractor(
    app_role, fresh_schema, tmp_path,
):
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        client = ScriptedClient([])
        u, a = _round()
        cfg = IngestConfig(extract_user=False, extract_agent=False)
        result = mem.ingest_round(u, a, config=cfg,
                                  extraction_client=client,
                                  resolver_client=client)
        assert result.user_facts == []
        assert result.agent_facts == []
        assert result.decisions == []
        # Episode still written (always-on raw audit)
        assert result.episode is not None
        assert client.call_count == 0
    finally:
        mem.close()


@pytest.mark.live
def test_ingest_round_uses_solo_default_session(app_role, fresh_schema, tmp_path):
    """Caller didn't pass session_id → SOLO daily session is used."""
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        client = ScriptedClient([
            _user_facts_payload("fact"),
            _agent_facts_payload("ack"),
            _decisions_payload(("ADD", None), ("ADD", None)),
        ])
        u, a = _round()
        result = mem.ingest_round(u, a,
                                  extraction_client=client,
                                  resolver_client=client)
        # Episode bound to the daily session (not None, not random)
        assert result.episode.session_id is not None
    finally:
        mem.close()


@pytest.mark.live
def test_ingest_round_noop_decision_does_not_write(app_role, fresh_schema, tmp_path):
    """NOOP-driven decision must not produce a memory_id in written list.

    Seed the doc store with an existing fact so the resolver actually
    fires (instead of short-circuiting to all-ADD)."""
    mem = _build_solo_mem(app_role, tmp_path)
    try:
        # Round 1 to seed an existing memory the resolver can compare against
        seed_user = json.dumps({
            "facts": [{"text": "seed", "category": "preference",
                       "entity": "user", "confidence": 0.9, "source_span": [0, 4]}]
        })
        seed_agent = json.dumps({"facts": []})
        client_seed = ScriptedClient([seed_user, seed_agent])
        u0, a0 = _round(thread_id="t-noop", offset=0)
        mem.ingest_round(u0, a0,
                         extraction_client=client_seed, resolver_client=client_seed)

        # Round 2: 2 user facts, resolver returns ADD + NOOP
        round_user = json.dumps({
            "facts": [
                {"text": "fresh", "category": "preference",
                 "entity": "user", "confidence": 0.9, "source_span": [0, 5]},
                {"text": "duplicate", "category": "preference",
                 "entity": "user", "confidence": 0.9, "source_span": [0, 9]},
            ]
        })
        round_agent = json.dumps({"facts": []})
        round_decisions = json.dumps({
            "decisions": [
                {"operation": "ADD", "existing_id": None, "rationale": "fresh"},
                {"operation": "NOOP", "existing_id": None, "rationale": "dup"},
            ]
        })
        client = ScriptedClient([round_user, round_agent, round_decisions])
        u, a = _round(thread_id="t-noop", offset=10)
        result = mem.ingest_round(u, a,
                                  extraction_client=client,
                                  resolver_client=client)
        assert [d.operation for d in result.decisions] == ["ADD", "NOOP"]
        # Only the ADD produced a written memory_id
        ops_with_ids = [(a.operation, a.memory_id) for a in result.applied]
        assert ("NOOP", None) in ops_with_ids
        adds = [m for o, m in ops_with_ids if o == "ADD"]
        assert len(adds) == 1 and adds[0] is not None
    finally:
        mem.close()
