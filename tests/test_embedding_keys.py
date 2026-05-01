"""Phase 4.1 — fact-augmented embedding keys (roadmap §B.1).

Verifies _build_embedding_text concatenates fact + raw round text + tags
on v4 rows with a source_episode_id, and falls back to just-fact text on
v3 rows or when the episode lookup fails.

Two test surfaces:
  - Unit tests with a stub psycopg2 connection (no DB needed).
  - Live test against the real v4 schema that writes a round + memory and
    asserts the embedding text reflects the round, not just the fact.
"""

from __future__ import annotations

import os
import uuid
from datetime import timedelta
from pathlib import Path

import pytest

try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

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


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres unreachable")


# ──────────────────────────────────────────────────────────────────────────
# Unit tests with a stub conn
# ──────────────────────────────────────────────────────────────────────────


class _StubCursor:
    """Mocks the psycopg2 cursor protocol for a single SELECT."""

    def __init__(self, row: dict | None) -> None:
        self._row = row

    def __enter__(self) -> _StubCursor:
        return self

    def __exit__(self, *exc) -> None:
        return None

    def execute(self, sql: str, params=None) -> None:
        return None

    def fetchone(self) -> dict | None:
        return self._row


class _StubConn:
    def __init__(self, row: dict | None = None) -> None:
        self._row = row

    def cursor(self, cursor_factory=None):
        return _StubCursor(self._row)


def _backend_with_stub(row: dict | None, *, v4: bool = True):
    """Construct a PostgresBackend stub without going through __init__."""
    from attestor.store.postgres_backend import PostgresBackend
    be = PostgresBackend.__new__(PostgresBackend)
    be._conn = _StubConn(row)
    be._v4 = v4
    return be


@pytest.mark.unit
def test_build_embedding_v3_returns_just_content() -> None:
    """v3 rows: no episode link, no enrichment — just the fact text."""
    be = _backend_with_stub(row=None, v4=False)
    out = be._build_embedding_text("m-1", "user prefers dark mode")
    assert out == "user prefers dark mode"


@pytest.mark.unit
def test_build_embedding_v4_no_episode_returns_just_content() -> None:
    """v4 row with no source_episode_id → no enrichment to do."""
    row = {
        "tags": [], "source_episode_id": None,
        "user_turn_text": None, "assistant_turn_text": None,
    }
    be = _backend_with_stub(row=row, v4=True)
    out = be._build_embedding_text("m-1", "fact text")
    assert out == "fact text"


@pytest.mark.unit
def test_build_embedding_v4_full_round_concat() -> None:
    """v4 row with an episode → fact + user turn + assistant turn."""
    row = {
        "tags": ["preference", "ui"],
        "source_episode_id": "ep-1",
        "user_turn_text": "Switch the UI to dark mode please.",
        "assistant_turn_text": "Done — all panels are now dark.",
    }
    be = _backend_with_stub(row=row, v4=True)
    out = be._build_embedding_text("m-1", "user prefers dark mode")
    assert "user prefers dark mode" in out
    assert "Switch the UI to dark mode please." in out
    assert "Done — all panels are now dark." in out
    assert "Tags: preference, ui" in out
    # Order matters: fact first, then user, then assistant, then tags
    parts = out.split("\n---\n")
    assert parts[0] == "user prefers dark mode"
    assert parts[1] == "Switch the UI to dark mode please."
    assert parts[2] == "Done — all panels are now dark."
    assert parts[3] == "Tags: preference, ui"


@pytest.mark.unit
def test_build_embedding_v4_skips_empty_tag_block() -> None:
    """No tags → no trailing 'Tags:' suffix."""
    row = {
        "tags": [], "source_episode_id": "ep-1",
        "user_turn_text": "What's 2+2?", "assistant_turn_text": "4",
    }
    be = _backend_with_stub(row=row, v4=True)
    out = be._build_embedding_text("m-1", "agent answered 4")
    assert "Tags:" not in out
    assert "What's 2+2?" in out


@pytest.mark.unit
def test_build_embedding_v4_partial_episode(
) -> None:
    """If only one side of the round was kept, still include what we have."""
    row = {
        "tags": ["x"],
        "source_episode_id": "ep-1",
        "user_turn_text": "user said something",
        "assistant_turn_text": None,
    }
    be = _backend_with_stub(row=row, v4=True)
    out = be._build_embedding_text("m-1", "fact")
    assert "user said something" in out
    # No empty assistant block
    parts = out.split("\n---\n")
    assert all(p.strip() for p in parts)


@pytest.mark.unit
def test_build_embedding_lookup_failure_falls_back_to_content() -> None:
    """If the SELECT raises, fall through to the just-fact path — never crash."""
    class _ExplodingConn:
        def cursor(self, cursor_factory=None):
            raise RuntimeError("DB on fire")
    from attestor.store.postgres_backend import PostgresBackend
    be = PostgresBackend.__new__(PostgresBackend)
    be._conn = _ExplodingConn()
    be._v4 = True
    out = be._build_embedding_text("m-1", "important fact")
    assert out == "important fact"


# ──────────────────────────────────────────────────────────────────────────
# Live test — round-write + add → embedding text is the full round
# ──────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def admin_conn():
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    yield c
    c.close()


@pytest.fixture
def fresh_schema(admin_conn):
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "1024")
    with admin_conn.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield


def _ollama_up() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@pytest.mark.live
@pytest.mark.skipif(not _ollama_up(), reason="Ollama not running")
def test_live_v4_embedding_text_reflects_round(admin_conn, fresh_schema) -> None:
    """Round → memory → _build_embedding_text returns the full round shape."""
    from attestor.store.postgres_backend import PostgresBackend
    be = PostgresBackend({
        "url": "postgresql://localhost:5432",
        "database": "attestor_v4_test",
        "auth": {"username": "postgres", "password": "attestor"},
        "v4": True,
        "skip_schema_init": True,
    })
    try:
        # Seed user + project + session via admin connection (RLS bypass).
        uid = uuid.uuid4()
        pid = uuid.uuid4()
        sid = uuid.uuid4()
        with admin_conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (id, external_id) VALUES (%s, %s)",
                (str(uid), f"u-{uuid.uuid4().hex[:8]}"),
            )
            cur.execute(
                "INSERT INTO projects (id, user_id, name) VALUES (%s, %s, %s)",
                (str(pid), str(uid), "p"),
            )
            cur.execute(
                "INSERT INTO sessions (id, user_id, project_id) VALUES (%s, %s, %s)",
                (str(sid), str(uid), str(pid)),
            )

        be._set_rls_user(str(uid))

        # Write an episode + memory linked to it
        from attestor.conversation.episodes import EpisodeRepo
        from attestor.conversation.turns import ConversationTurn
        from attestor.models import Memory

        repo = EpisodeRepo(be._conn)
        u = ConversationTurn(thread_id="t", speaker="user", role="user",
                             content="I love sushi.")
        a = ConversationTurn(thread_id="t", speaker="planner", role="assistant",
                             content="Noted — sushi added to favorites.",
                             ts=u.ts + timedelta(seconds=1))
        ep = repo.write_round(
            user_id=str(uid), session_id=str(sid), project_id=str(pid),
            user_turn=u, assistant_turn=a,
        )

        m = Memory(
            content="user enjoys sushi", category="preference",
            entity="food", confidence=0.9,
            user_id=str(uid), project_id=str(pid), session_id=str(sid),
            scope="user", source_episode_id=ep.id,
            agent_id="user", source_span=[0, 19],
            tags=["food", "preference"],
        )
        inserted = be.insert(m)

        text = be._build_embedding_text(inserted.id, inserted.content)
        assert "user enjoys sushi" in text
        assert "I love sushi." in text
        assert "Noted — sushi added to favorites." in text
        assert "Tags: food, preference" in text
    finally:
        be.close()
