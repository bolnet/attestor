"""Phase 6.2 — AgentMemory.recall_as_pack end-to-end.

Verifies the public surface returns a ContextPack with citations,
preserves orchestrator ranking, and threads as_of through correctly.
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


def _ollama_up() -> bool:
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not (_reachable() and _ollama_up()),
    reason="local Postgres or Ollama unreachable",
)


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


def _build_solo_mem(tmp_path: Path):
    from attestor.core import AgentMemory
    return AgentMemory(tmp_path, config={
        "mode": "solo",
        "backends": ["postgres"],
        "backend_configs": {
            "postgres": {
                "url": "postgresql://localhost:5432",
                "database": "attestor_v4_test",
                "auth": {"username": "postgres", "password": "attestor"},
                "v4": True, "skip_schema_init": True,
            }
        },
    })


@pytest.fixture
def seeded_mem(admin_conn, fresh_schema, tmp_path):
    """SOLO mem with a few facts seeded via the public add()."""
    mem = _build_solo_mem(tmp_path)
    mem.add(content="user prefers dark mode", category="preference",
            entity="UI")
    mem.add(content="user lives in San Francisco", category="location",
            entity="user")
    mem.add(content="user enjoys sushi", category="preference",
            entity="food")
    yield mem
    mem.close()


# ──────────────────────────────────────────────────────────────────────────
# recall_as_pack — core surface
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_recall_as_pack_returns_context_pack(seeded_mem):
    from attestor.models import ContextPack
    pack = seeded_mem.recall_as_pack("dark mode preference")
    assert isinstance(pack, ContextPack)
    assert pack.query == "dark mode preference"
    assert pack.memory_count >= 1


@pytest.mark.live
def test_recall_as_pack_entries_carry_citation_fields(seeded_mem):
    pack = seeded_mem.recall_as_pack("dark mode")
    for e in pack.memories:
        assert e.id, "memory id must be set for citation"
        assert e.content
        assert isinstance(e.confidence, float)
        # source_episode_id may be None for direct add() rows


@pytest.mark.live
def test_recall_as_pack_includes_default_chain_of_note(seeded_mem):
    from attestor.prompts.chain_of_note import DEFAULT_CHAIN_OF_NOTE_PROMPT
    pack = seeded_mem.recall_as_pack("anything")
    assert pack.chain_of_note_prompt == DEFAULT_CHAIN_OF_NOTE_PROMPT
    assert "ABSTAIN" in pack.chain_of_note_prompt


@pytest.mark.live
def test_recall_as_pack_render_prompt_inlines_memories(seeded_mem):
    pack = seeded_mem.recall_as_pack("dark mode")
    rendered = pack.render_prompt()
    # Each memory id must appear in the rendered prompt's JSON block
    for e in pack.memories:
        assert e.id in rendered
    assert "{memories_json}" not in rendered


@pytest.mark.live
def test_recall_as_pack_empty_query_still_has_prompt(seeded_mem):
    """Even when no memories match (abstention case), the prompt is there."""
    pack = seeded_mem.recall_as_pack("xyzzy_no_such_term_anywhere_in_corpus")
    assert "ABSTAIN" in pack.chain_of_note_prompt


@pytest.mark.live
def test_recall_as_pack_preserves_ranking(seeded_mem):
    """Pack memories arrive in the same order the orchestrator returned."""
    results = seeded_mem.recall("dark mode")
    real_results = [r for r in results if r.memory.category != "graph_relation"]
    pack = seeded_mem.recall_as_pack("dark mode")
    assert [e.id for e in pack.memories] == [r.memory.id for r in real_results]


@pytest.mark.live
def test_recall_as_pack_custom_prompt_override(seeded_mem):
    custom = "MY CUSTOM PROMPT\n{memories_json}"
    pack = seeded_mem.recall_as_pack("dark mode", chain_of_note_prompt=custom)
    assert pack.chain_of_note_prompt == custom


@pytest.mark.live
def test_recall_as_pack_threads_as_of(seeded_mem, admin_conn):
    """as_of is recorded in the pack metadata."""
    as_of = datetime(2026, 1, 1, tzinfo=timezone.utc)
    pack = seeded_mem.recall_as_pack("anything", as_of=as_of)
    assert pack.as_of == as_of.isoformat()


@pytest.mark.live
def test_recall_as_pack_token_count_is_estimate(seeded_mem):
    pack = seeded_mem.recall_as_pack("dark mode")
    expected = sum(len(e.content) for e in pack.memories) // 4
    assert pack.token_count == expected


@pytest.mark.live
def test_recall_as_pack_to_dict_is_jsonable(seeded_mem):
    """to_dict must be json.dumps-safe — it's the agent's serialization path."""
    import json
    pack = seeded_mem.recall_as_pack("dark mode")
    serialized = json.dumps(pack.to_dict())
    assert "ABSTAIN" in serialized


# ──────────────────────────────────────────────────────────────────────────
# Backward compat — recall_as_context still returns a string
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.live
def test_recall_as_context_string_signature_unchanged(seeded_mem):
    out = seeded_mem.recall_as_context("dark mode")
    assert isinstance(out, str)
