"""Phase 4.2 — BM25 / FTS retrieval lane (roadmap §B.2).

Tests:
  - reciprocal_rank_fusion: pure-function unit tests
  - BM25Lane.search: live tests against the v4 FTS column
  - graceful degradation when content_tsv column is missing (v3 schema)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest

try:
    import psycopg2
    HAVE_PSYCOPG2 = True
except ImportError:
    HAVE_PSYCOPG2 = False

from attestor.retrieval.bm25 import BM25Hit, BM25Lane, reciprocal_rank_fusion

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


# ──────────────────────────────────────────────────────────────────────────
# RRF (pure function — no DB)
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_rrf_single_lane_preserves_order() -> None:
    out = reciprocal_rank_fusion(["a", "b", "c"])
    assert out == ["a", "b", "c"]


@pytest.mark.unit
def test_rrf_two_lanes_overlap_at_top_wins() -> None:
    """An item in BOTH lanes' top-3 outranks unique top-1s."""
    lane_vec = ["a", "b", "c"]
    lane_bm = ["d", "b", "e"]
    out = reciprocal_rank_fusion(lane_vec, lane_bm)
    # 'b' appears in both, so its fused score is highest
    assert out[0] == "b"


@pytest.mark.unit
def test_rrf_disjoint_lanes_interleave_by_rank() -> None:
    out = reciprocal_rank_fusion(["a", "b"], ["c", "d"])
    # All four ids appear; rank-1s ('a' and 'c') tied at 1/(60+1) = 1/61
    assert set(out) == {"a", "b", "c", "d"}
    assert out[0] in {"a", "c"}
    assert out[1] in {"a", "c"}


@pytest.mark.unit
def test_rrf_limit_clips_results() -> None:
    out = reciprocal_rank_fusion(["a", "b", "c", "d"], limit=2)
    assert len(out) == 2


@pytest.mark.unit
def test_rrf_higher_k_flattens_lane_influence() -> None:
    """k=1000 makes per-lane rank differences negligible — closer to set union."""
    lane1 = ["a", "b", "c"]
    lane2 = ["d", "e", "f"]
    fused_default = reciprocal_rank_fusion(lane1, lane2, k=60)
    fused_flat = reciprocal_rank_fusion(lane1, lane2, k=1000)
    # Both produce the same set, but ordering should be the same too
    # since the lane structures are symmetric.
    assert set(fused_default) == set(fused_flat)


@pytest.mark.unit
def test_rrf_empty_lanes_return_empty() -> None:
    assert reciprocal_rank_fusion([], []) == []


# ──────────────────────────────────────────────────────────────────────────
# BM25Lane (live)
# ──────────────────────────────────────────────────────────────────────────


pytestmark = pytest.mark.skipif(not _reachable(), reason="local Postgres unreachable")


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


@pytest.fixture
def seeded_user(admin_conn, fresh_schema):
    """Create a user and seed five memories with distinct content."""
    uid = uuid.uuid4()
    with admin_conn.cursor() as cur:
        cur.execute(
            "INSERT INTO users (id, external_id) VALUES (%s, %s)",
            (str(uid), f"u-bm25-{uuid.uuid4().hex[:8]}"),
        )
        # Set RLS on this admin connection so our INSERTs go in scoped
        cur.execute(
            "SELECT set_config('attestor.current_user_id', %s, false)",
            (str(uid),),
        )
        seeds = [
            ("user prefers dark mode in all apps", ["preference", "ui"], "ui"),
            ("user lives in San Francisco", ["location"], "user"),
            ("user works at Acme Corp as a Senior Engineer",
             ["career"], "Acme Corp"),
            ("user enjoys sushi and ramen", ["food"], "food"),
            ("user is currently learning Rust programming",
             ["learning"], "Rust"),
        ]
        ids = []
        for content, tags, entity in seeds:
            cur.execute(
                "INSERT INTO memories (user_id, content, tags, entity) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (str(uid), content, tags, entity),
            )
            ids.append(str(cur.fetchone()[0]))
    return str(uid), ids, admin_conn


@pytest.mark.live
def test_bm25_finds_exact_keyword(seeded_user) -> None:
    uid, ids, conn = seeded_user
    lane = BM25Lane(conn)
    hits = lane.search("dark mode")
    assert len(hits) >= 1
    assert isinstance(hits[0], BM25Hit)
    # Top hit must contain "dark mode"
    with conn.cursor() as cur:
        cur.execute("SELECT content FROM memories WHERE id = %s", (hits[0].memory_id,))
        content = cur.fetchone()[0]
    assert "dark mode" in content


@pytest.mark.live
def test_bm25_phrase_query(seeded_user) -> None:
    """Phrase via websearch syntax — quoted terms must appear adjacent."""
    uid, ids, conn = seeded_user
    lane = BM25Lane(conn)
    hits = lane.search('"San Francisco"')
    assert len(hits) >= 1


@pytest.mark.live
def test_bm25_tag_match_via_b_weight(seeded_user) -> None:
    """The generated tsvector includes tags with weight B — searching a
    tag word should still surface the row even if not in content."""
    uid, ids, conn = seeded_user
    lane = BM25Lane(conn)
    hits = lane.search("learning")
    contents = []
    with conn.cursor() as cur:
        for h in hits:
            cur.execute("SELECT content FROM memories WHERE id = %s", (h.memory_id,))
            contents.append(cur.fetchone()[0])
    assert any("Rust" in c for c in contents)


@pytest.mark.live
def test_bm25_entity_match_via_c_weight(seeded_user) -> None:
    """Entity values are indexed too (weight C)."""
    uid, ids, conn = seeded_user
    lane = BM25Lane(conn)
    hits = lane.search("Acme")
    assert len(hits) >= 1


@pytest.mark.live
def test_bm25_negation(seeded_user) -> None:
    """websearch_to_tsquery supports -term to exclude."""
    uid, ids, conn = seeded_user
    lane = BM25Lane(conn)
    # 'user' alone matches everything; '-Rust' should drop the Rust row
    hits_with = lane.search("user")
    hits_without = lane.search("user -Rust")
    assert len(hits_without) < len(hits_with)


@pytest.mark.live
def test_bm25_no_match_returns_empty(seeded_user) -> None:
    uid, ids, conn = seeded_user
    lane = BM25Lane(conn)
    hits = lane.search("xyzzy_no_such_term_anywhere")
    assert hits == []


@pytest.mark.live
def test_bm25_empty_query_returns_empty(seeded_user) -> None:
    uid, ids, conn = seeded_user
    lane = BM25Lane(conn)
    assert lane.search("") == []
    assert lane.search("   ") == []


@pytest.mark.live
def test_bm25_active_only_filter(seeded_user) -> None:
    """status='superseded' rows are excluded by default."""
    uid, ids, conn = seeded_user
    # Mark one row superseded
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE memories SET status = 'superseded' WHERE id = %s",
            (ids[0],),
        )
    lane = BM25Lane(conn)
    hits = lane.search("dark mode")
    assert all(h.memory_id != ids[0] for h in hits)
    # active_only=False should bring it back
    hits_all = lane.search("dark mode", active_only=False)
    assert any(h.memory_id == ids[0] for h in hits_all)


@pytest.mark.live
def test_bm25_handles_missing_column_gracefully(admin_conn, fresh_schema) -> None:
    """If the FTS column doesn't exist (older schema), search returns []."""
    with admin_conn.cursor() as cur:
        cur.execute("ALTER TABLE memories DROP COLUMN content_tsv")
    lane = BM25Lane(admin_conn)
    assert lane.search("anything") == []
