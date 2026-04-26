"""Phase 3.1 — EpisodeRepo CRUD against the v4 ``episodes`` table.

Runs against the dedicated ``attestor_v4_test`` database. Schema is
admin-loaded at module scope; each test gets a fresh user + session so
RLS policies admit the writes.
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

from attestor.conversation.episodes import EpisodeRepo
from attestor.conversation.turns import ConversationTurn

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


@pytest.fixture(scope="module")
def conn():
    """Module-level superuser connection. Schema reload + RLS bypassed."""
    c = psycopg2.connect(PG_URL)
    c.autocommit = True
    raw = SCHEMA_PATH.read_text().replace("{embedding_dim}", "16")
    with c.cursor() as cur:
        for tbl in ("memories", "episodes", "sessions", "projects", "users"):
            cur.execute(f"DROP TABLE IF EXISTS {tbl} CASCADE")
        cur.execute(raw)
    yield c
    c.close()


@pytest.fixture
def session_ctx(conn):
    """Create a user + project + session; return (user_id, project_id, session_id)."""
    uid = uuid.uuid4()
    pid = uuid.uuid4()
    sid = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), f"u-ep-{uuid.uuid4().hex[:8]}"),
        )
        cur.execute(
            "INSERT INTO projects (id, user_id, name) VALUES (%s, %s, %s)",
            (str(pid), str(uid), "ep-proj"),
        )
        cur.execute(
            "INSERT INTO sessions (id, user_id, project_id) VALUES (%s, %s, %s)",
            (str(sid), str(uid), str(pid)),
        )
    return str(uid), str(pid), str(sid)


def _round(thread_id: str = "thr-1", offset_seconds: int = 0):
    """Build a (user_turn, assistant_turn) pair separated by 1s."""
    base = datetime.now(timezone.utc) + timedelta(seconds=offset_seconds)
    user = ConversationTurn(
        thread_id=thread_id, speaker="user", role="user",
        content="What's the weather?", ts=base,
    )
    assistant = ConversationTurn(
        thread_id=thread_id, speaker="planner", role="assistant",
        content="Sunny, 72°F.", ts=base + timedelta(seconds=1),
    )
    return user, assistant


# ───────────────────────────────────────────────────────────────────────────
# Write
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_write_round_persists_verbatim(conn, session_ctx):
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)
    user_t, assistant_t = _round()
    ep = repo.write_round(
        user_id=uid, session_id=sid, project_id=pid,
        user_turn=user_t, assistant_turn=assistant_t,
        agent_id="planner-01",
    )
    assert ep.user_turn_text == "What's the weather?"
    assert ep.assistant_turn_text == "Sunny, 72°F."
    assert ep.thread_id == "thr-1"
    assert ep.agent_id == "planner-01"
    assert ep.user_id == uid
    assert ep.session_id == sid


@pytest.mark.live
def test_write_round_thread_id_mismatch_raises(conn, session_ctx):
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)
    user_t = ConversationTurn(
        thread_id="t-A", speaker="user", role="user", content="hi",
    )
    assistant_t = ConversationTurn(
        thread_id="t-B", speaker="a", role="assistant", content="bye",
    )
    with pytest.raises(ValueError, match="thread_id mismatch"):
        repo.write_round(
            user_id=uid, session_id=sid,
            user_turn=user_t, assistant_turn=assistant_t,
        )


@pytest.mark.live
def test_write_round_role_mismatch_raises(conn, session_ctx):
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)
    swapped_user = ConversationTurn(
        thread_id="t", speaker="user", role="assistant", content="hi",
    )
    assistant = ConversationTurn(
        thread_id="t", speaker="a", role="assistant", content="bye",
    )
    with pytest.raises(ValueError, match="user_turn.role"):
        repo.write_round(
            user_id=uid, session_id=sid,
            user_turn=swapped_user, assistant_turn=assistant,
        )


@pytest.mark.live
def test_write_round_assistant_before_user_raises(conn, session_ctx):
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)
    base = datetime.now(timezone.utc)
    user = ConversationTurn(
        thread_id="t", speaker="user", role="user", content="hi", ts=base,
    )
    assistant = ConversationTurn(
        thread_id="t", speaker="a", role="assistant", content="bye",
        ts=base - timedelta(seconds=5),  # before user
    )
    with pytest.raises(ValueError, match="cannot precede"):
        repo.write_round(
            user_id=uid, session_id=sid,
            user_turn=user, assistant_turn=assistant,
        )


# ───────────────────────────────────────────────────────────────────────────
# Read
# ───────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_get_returns_round(conn, session_ctx):
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)
    u, a = _round()
    ep = repo.write_round(user_id=uid, session_id=sid, user_turn=u, assistant_turn=a)
    fetched = repo.get(ep.id)
    assert fetched is not None
    assert fetched.id == ep.id
    assert fetched.user_turn_text == u.content


@pytest.mark.live
def test_get_missing_returns_none(conn):
    repo = EpisodeRepo(conn)
    assert repo.get(str(uuid.uuid4())) is None


@pytest.mark.live
def test_list_for_thread_chronological(conn, session_ctx):
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)
    # Three rounds at 0, 10, 20 seconds in the same thread
    for offset in (0, 10, 20):
        u, a = _round(thread_id="thr-A", offset_seconds=offset)
        repo.write_round(user_id=uid, session_id=sid, user_turn=u, assistant_turn=a)

    rounds = repo.list_for_thread("thr-A")
    assert len(rounds) == 3
    timestamps = [r.user_ts for r in rounds]
    assert timestamps == sorted(timestamps), "must be chronological asc"


@pytest.mark.live
def test_list_for_session_isolation(conn, session_ctx):
    """list_for_session returns only that session's episodes."""
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)

    # Make a second session for the same user
    sid2 = uuid.uuid4()
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO sessions (id, user_id, project_id) VALUES (%s, %s, %s)",
            (str(sid2), uid, pid),
        )

    u1, a1 = _round(thread_id="t1")
    u2, a2 = _round(thread_id="t2", offset_seconds=5)
    repo.write_round(user_id=uid, session_id=sid, user_turn=u1, assistant_turn=a1)
    repo.write_round(user_id=uid, session_id=str(sid2), user_turn=u2, assistant_turn=a2)

    s1_rounds = repo.list_for_session(sid)
    s2_rounds = repo.list_for_session(str(sid2))
    assert len(s1_rounds) == 1 and s1_rounds[0].thread_id == "t1"
    assert len(s2_rounds) == 1 and s2_rounds[0].thread_id == "t2"


@pytest.mark.live
def test_count_for_session(conn, session_ctx):
    uid, pid, sid = session_ctx
    repo = EpisodeRepo(conn)
    assert repo.count_for_session(sid) == 0
    for i in range(3):
        u, a = _round(thread_id=f"thr-{i}", offset_seconds=i)
        repo.write_round(user_id=uid, session_id=sid, user_turn=u, assistant_turn=a)
    assert repo.count_for_session(sid) == 3
