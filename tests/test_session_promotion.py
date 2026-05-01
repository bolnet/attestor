"""Phase 7.4 — SESSION_PROMOTION + apply tests + end_session enqueue."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

from attestor.consolidation.session_end import (
    SESSION_PROMOTION_PROMPT,
    VALID_PROMOTIONS,
    PromotionDecision,
    apply_promotions,
    decide_promotions,
)
from attestor.models import Memory


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


pytestmark = pytest.mark.live


# ──────────────────────────────────────────────────────────────────────────
# Stub LLM
# ──────────────────────────────────────────────────────────────────────────


class _Choice:
    def __init__(self, c: str) -> None:
        self.message = type("M", (), {"content": c})()


class _Resp:
    def __init__(self, c: str) -> None:
        self.choices = [_Choice(c)]


class _Chat:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self._p = payload
        self._exc = exc
        self.calls: list = []

    def create(self, **kw: Any) -> _Resp:
        self.calls.append(kw)
        if self._exc is not None:
            raise self._exc
        return _Resp(self._p)


class StubClient:
    def __init__(self, payload: str = "", *, exc: Exception | None = None) -> None:
        self.chat = type("C", (), {"completions": _Chat(payload, exc=exc)})()


def _mem(mid: str, content: str = "fact", scope: str = "session") -> Memory:
    return Memory(id=mid, content=content, category="preference",
                  entity="user", confidence=0.9, scope=scope)


# ──────────────────────────────────────────────────────────────────────────
# Prompt content guards
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_prompt_includes_all_four_operations() -> None:
    for op in ("KEEP_SESSION", "PROMOTE_PROJECT", "PROMOTE_USER", "DISCARD"):
        assert op in SESSION_PROMOTION_PROMPT


@pytest.mark.unit
def test_prompt_template_renders() -> None:
    out = SESSION_PROMOTION_PROMPT.format(
        memory_count=3, memories_json="[]",
    )
    assert "3 items" in out
    assert "[]" in out


@pytest.mark.unit
def test_valid_operations_set_complete() -> None:
    assert VALID_PROMOTIONS == {
        "KEEP_SESSION", "PROMOTE_PROJECT", "PROMOTE_USER", "DISCARD",
    }


@pytest.mark.unit
def test_decision_rejects_invalid_operation() -> None:
    with pytest.raises(ValueError, match="operation"):
        PromotionDecision(memory_id="m", operation="WIDEN", rationale="r")


# ──────────────────────────────────────────────────────────────────────────
# decide_promotions parsing
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_decide_empty_returns_empty() -> None:
    assert decide_promotions([], client=StubClient(payload="{}")) == []


@pytest.mark.unit
def test_decide_no_client_defaults_keep_session() -> None:
    out = decide_promotions([_mem("m1")], client=None)
    assert all(d.operation == "KEEP_SESSION" for d in out)


@pytest.mark.unit
def test_decide_index_aligned_per_id() -> None:
    payload = json.dumps({
        "decisions": [
            {"memory_id": "m1", "operation": "PROMOTE_USER", "rationale": "stable"},
            {"memory_id": "m2", "operation": "DISCARD", "rationale": "noise"},
            {"memory_id": "m3", "operation": "KEEP_SESSION", "rationale": "thread-only"},
        ]
    })
    out = decide_promotions(
        [_mem("m1"), _mem("m2"), _mem("m3")],
        client=StubClient(payload=payload),
    )
    assert [(d.memory_id, d.operation) for d in out] == [
        ("m1", "PROMOTE_USER"),
        ("m2", "DISCARD"),
        ("m3", "KEEP_SESSION"),
    ]


@pytest.mark.unit
def test_decide_missing_id_defaults_keep() -> None:
    """LLM omits a decision for one id → that one gets KEEP_SESSION."""
    payload = json.dumps({
        "decisions": [
            {"memory_id": "m1", "operation": "PROMOTE_USER", "rationale": "ok"},
        ]
    })
    out = decide_promotions(
        [_mem("m1"), _mem("m2")],
        client=StubClient(payload=payload),
    )
    assert out[1].operation == "KEEP_SESSION"
    assert "default KEEP" in out[1].rationale


@pytest.mark.unit
def test_decide_invalid_operation_defaults_keep() -> None:
    payload = json.dumps({
        "decisions": [
            {"memory_id": "m1", "operation": "MERGE", "rationale": "weird"},
        ]
    })
    out = decide_promotions([_mem("m1")], client=StubClient(payload=payload))
    assert out[0].operation == "KEEP_SESSION"


@pytest.mark.unit
def test_decide_llm_error_defaults_keep() -> None:
    out = decide_promotions(
        [_mem("m1"), _mem("m2")],
        client=StubClient(exc=RuntimeError("api down")),
    )
    assert all(d.operation == "KEEP_SESSION" for d in out)


@pytest.mark.unit
def test_decide_bad_json_defaults_keep() -> None:
    out = decide_promotions(
        [_mem("m1")],
        client=StubClient(payload="not json"),
    )
    assert out[0].operation == "KEEP_SESSION"


# ──────────────────────────────────────────────────────────────────────────
# apply_promotions — stub mem
# ──────────────────────────────────────────────────────────────────────────


class _StubStore:
    def __init__(self) -> None:
        self.rows: dict = {}
        self.update_calls = 0

    def get(self, mid):
        return self.rows.get(mid)

    def update(self, m):
        self.rows[m.id] = m
        self.update_calls += 1
        return m


class _StubMem:
    def __init__(self) -> None:
        self._store = _StubStore()


@pytest.mark.unit
def test_apply_keep_session_is_noop() -> None:
    mem = _StubMem()
    mem._store.rows["m1"] = _mem("m1")
    out = apply_promotions(
        [PromotionDecision(memory_id="m1", operation="KEEP_SESSION", rationale="r")],
        mem=mem,
    )
    assert out[0].operation == "KEEP_SESSION"
    assert mem._store.update_calls == 0


@pytest.mark.unit
def test_apply_promote_project_widens_scope() -> None:
    mem = _StubMem()
    mem._store.rows["m1"] = _mem("m1", scope="session")
    out = apply_promotions(
        [PromotionDecision(memory_id="m1", operation="PROMOTE_PROJECT", rationale="r")],
        mem=mem,
    )
    assert out[0].new_scope == "project"
    assert mem._store.rows["m1"].scope == "project"


@pytest.mark.unit
def test_apply_promote_user_widens_to_user() -> None:
    mem = _StubMem()
    mem._store.rows["m1"] = _mem("m1", scope="session")
    apply_promotions(
        [PromotionDecision(memory_id="m1", operation="PROMOTE_USER", rationale="r")],
        mem=mem,
    )
    assert mem._store.rows["m1"].scope == "user"


@pytest.mark.unit
def test_apply_discard_marks_superseded() -> None:
    mem = _StubMem()
    mem._store.rows["m1"] = _mem("m1")
    out = apply_promotions(
        [PromotionDecision(memory_id="m1", operation="DISCARD", rationale="noise")],
        mem=mem,
    )
    assert out[0].superseded is True
    assert mem._store.rows["m1"].status == "superseded"
    assert mem._store.rows["m1"].valid_until is not None


@pytest.mark.unit
def test_apply_missing_memory_records_error() -> None:
    mem = _StubMem()
    out = apply_promotions(
        [PromotionDecision(memory_id="vanished",
                           operation="PROMOTE_PROJECT", rationale="r")],
        mem=mem,
    )
    assert out[0].error == "memory vanished"


# ──────────────────────────────────────────────────────────────────────────
# end_session enqueue + AgentMemory.consolidate (live)
# ──────────────────────────────────────────────────────────────────────────


pytestmark_live = pytest.mark.skipif(
    not _reachable(), reason="local Postgres unreachable",
)


@pytest.fixture(scope="module")
def admin_conn():
    if not _reachable():
        pytest.skip("local Postgres unreachable")
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


@pytest.mark.live
def test_end_session_re_enqueues_done_episodes(admin_conn, fresh_schema):
    """SessionRepo.end() must move done/failed episodes back to pending."""
    from attestor.identity.sessions import SessionRepo
    uid = uuid.uuid4()
    sid = uuid.uuid4()
    eid = uuid.uuid4()
    base = datetime.now(timezone.utc)
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), f"u-end-{uuid.uuid4().hex[:8]}"),
        )
        cur.execute(
            "INSERT INTO sessions (id, user_id) VALUES (%s, %s)",
            (str(sid), str(uid)),
        )
        cur.execute(
            "INSERT INTO episodes (id, user_id, session_id, thread_id, "
            "user_turn_text, assistant_turn_text, user_ts, assistant_ts, "
            "consolidation_state, consolidation_done_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'done', NOW())",
            (str(eid), str(uid), str(sid), "t", "u", "a", base,
             base + timedelta(seconds=1)),
        )

    repo = SessionRepo(admin_conn)
    sess = repo.end(str(sid))
    assert sess is not None
    assert sess.status == "ended"

    with admin_conn.cursor() as cur:
        cur.execute(
            "SELECT consolidation_state FROM episodes WHERE id = %s",
            (str(eid),),
        )
        assert cur.fetchone()[0] == "pending"


@pytest.mark.live
def test_agent_memory_consolidate_drains_queue(admin_conn, fresh_schema, tmp_path):
    """AgentMemory.consolidate(...) is the public one-shot entry point."""
    from attestor.conversation import ConversationTurn
    from attestor.core import AgentMemory

    mem = AgentMemory(tmp_path, config={
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {"postgres": {
            "url": "postgresql://localhost:5432",
            "database": "attestor_v4_test",
            "auth": {"username": "postgres", "password": "attestor"},
            "v4": True, "skip_schema_init": True, "embedding_dim": 16,
        }},
    })
    try:
        # Stub for ingest path + consolidator
        class _Chat:
            def __init__(self, ps): self._ps = list(ps); self.calls = []
            def create(self, **kw):
                self.calls.append(kw)
                if not self._ps:
                    return _Resp('{"facts": []}')
                return _Resp(self._ps.pop(0))
        class _Cli:
            def __init__(self, ps):
                self.chat = type("C", (), {"completions": _Chat(ps)})()

        ingest_cli = _Cli(['{"facts": []}'] * 4)
        u = ConversationTurn(thread_id="t", speaker="user", role="user", content="u1")
        a = ConversationTurn(thread_id="t", speaker="p", role="assistant", content="a1",
                             ts=u.ts + timedelta(seconds=1))
        mem.ingest_round(u, a, extraction_client=ingest_cli, resolver_client=ingest_cli)

        cons_cli = _Cli(['{"facts": []}'] * 4)
        results = mem.consolidate(
            limit=10, extraction_client=cons_cli, resolver_client=cons_cli,
        )
        assert len(results) == 1
        assert results[0].ok
    finally:
        mem.close()
