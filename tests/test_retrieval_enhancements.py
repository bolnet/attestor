"""Tests for retrieval enhancements: confidence decay/boost, MMR, dedup, PageRank."""

from datetime import datetime, timedelta, timezone

import pytest

from attestor.models import Memory, RetrievalResult
from attestor.retrieval.scorer import (
    confidence_decay_boost,
    mmr_rerank,
    pagerank_boost,
)


def _make_result(
    content,
    score,
    entity=None,
    memory_id=None,
    access_count=0,
    last_accessed=None,
    created_at=None,
    confidence=1.0,
):
    kwargs = {
        "content": content,
        "entity": entity,
        "confidence": confidence,
        "access_count": access_count,
    }
    if memory_id:
        kwargs["id"] = memory_id
    if last_accessed:
        kwargs["last_accessed"] = last_accessed
    if created_at:
        kwargs["created_at"] = created_at
    m = Memory(**kwargs)
    return RetrievalResult(memory=m, score=score, match_source="test")


# ---------------------------------------------------------------------------
# Confidence Decay/Boost
# ---------------------------------------------------------------------------

class TestConfidenceDecayBoost:
    def test_recent_high_access_scores_higher(self):
        """Frequently accessed, recent memory should score higher."""
        now = datetime.now(timezone.utc).isoformat()
        r = _make_result("active memory", 1.0, access_count=10, last_accessed=now)
        results = confidence_decay_boost([r])
        # 1.0 * (1.0 + 0.03*10) = 1.0 * min(1.0, 1.3) = 1.0
        assert results[0].score == pytest.approx(1.0)

    def test_old_no_access_decays(self):
        """Memory untouched for many hours should decay."""
        old = (datetime.now(timezone.utc) - timedelta(hours=500)).isoformat()
        r = _make_result("stale memory", 1.0, access_count=0, last_accessed=old)
        results = confidence_decay_boost([r])
        # confidence = 1.0 - 0.001*500 = 0.5, score = 1.0 * 0.5
        assert results[0].score == pytest.approx(0.5, abs=0.05)

    def test_floor_at_0_1(self):
        """Confidence should never go below 0.1."""
        ancient = (datetime.now(timezone.utc) - timedelta(hours=5000)).isoformat()
        r = _make_result("ancient memory", 1.0, access_count=0, last_accessed=ancient)
        results = confidence_decay_boost([r])
        # Decayed well past 0 but floored at 0.1
        assert results[0].score == pytest.approx(0.1, abs=0.01)

    def test_gate_filters_low_confidence(self):
        """Memories below gate threshold should be excluded."""
        old = (datetime.now(timezone.utc) - timedelta(hours=400)).isoformat()
        r = _make_result("low conf", 1.0, access_count=0, last_accessed=old)
        # confidence = 1.0 - 0.001*400 = 0.6 -> below gate of 0.7
        results = confidence_decay_boost([r], gate=0.7)
        assert len(results) == 0

    def test_access_boost_counteracts_decay(self):
        """Enough accesses should counteract time decay."""
        old = (datetime.now(timezone.utc) - timedelta(hours=200)).isoformat()
        r = _make_result("popular", 1.0, access_count=20, last_accessed=old)
        results = confidence_decay_boost([r])
        # conf = 1.0 - 0.001*200 + 0.03*20 = 1.0 - 0.2 + 0.6 = 1.0 (clamped)
        assert results[0].score == pytest.approx(1.0, abs=0.05)

    def test_no_mutation(self):
        """Original results should not be mutated."""
        r = _make_result("test", 1.0, access_count=5)
        original_score = r.score
        confidence_decay_boost([r])
        assert r.score == original_score


# ---------------------------------------------------------------------------
# MMR Reranking
# ---------------------------------------------------------------------------

class TestMMRRerank:
    def test_selects_diverse_results(self):
        """Near-duplicate results should be deprioritized."""
        r1 = _make_result("the cat sat on the mat", 1.0, memory_id="m1")
        r2 = _make_result("the cat sat on the mat today", 0.9, memory_id="m2")
        r3 = _make_result("python is a programming language", 0.8, memory_id="m3")
        results = mmr_rerank([r1, r2, r3], lambda_param=0.5, max_results=2)
        # r1 selected first (highest score), then r3 (more diverse than r2)
        assert results[0].memory.id == "m1"
        assert results[1].memory.id == "m3"

    def test_lambda_1_is_pure_relevance(self):
        """lambda=1.0 should just return by score order."""
        r1 = _make_result("same content here", 1.0, memory_id="m1")
        r2 = _make_result("same content here too", 0.9, memory_id="m2")
        r3 = _make_result("different topic entirely", 0.8, memory_id="m3")
        results = mmr_rerank([r1, r2, r3], lambda_param=1.0)
        assert [r.memory.id for r in results] == ["m1", "m2", "m3"]

    def test_single_result_unchanged(self):
        r = _make_result("only one", 1.0)
        results = mmr_rerank([r])
        assert len(results) == 1
        assert results[0].score == 1.0

    def test_max_results_respected(self):
        items = [_make_result(f"item {i}", 1.0 - i * 0.1, memory_id=f"m{i}") for i in range(10)]
        results = mmr_rerank(items, max_results=3)
        assert len(results) == 3

    def test_empty_input(self):
        assert mmr_rerank([]) == []


# ---------------------------------------------------------------------------
# PageRank Boost
# ---------------------------------------------------------------------------

class TestPageRankBoost:
    def test_boosts_high_pr_entity(self):
        r1 = _make_result("about Alice", 0.5, entity="Alice")
        r2 = _make_result("about Bob", 0.5, entity="Bob")
        pr_scores = {"alice": 0.8, "bob": 0.1}
        results = pagerank_boost([r1, r2], pr_scores, weight=0.3)
        # Alice: 0.5 + 0.3*0.8 = 0.74, Bob: 0.5 + 0.3*0.1 = 0.53
        assert results[0].score > results[1].score
        assert results[0].score == pytest.approx(0.74)
        assert results[1].score == pytest.approx(0.53)

    def test_no_entity_unaffected(self):
        r = _make_result("no entity here", 0.5)
        results = pagerank_boost([r], {"alice": 0.8}, weight=0.3)
        assert results[0].score == pytest.approx(0.5)

    def test_empty_pr_scores_passthrough(self):
        r = _make_result("test", 0.5, entity="Alice")
        results = pagerank_boost([r], {}, weight=0.3)
        assert results[0].score == pytest.approx(0.5)

    def test_no_mutation(self):
        r = _make_result("test", 0.5, entity="Alice")
        original = r.score
        pagerank_boost([r], {"alice": 0.5}, weight=0.3)
        assert r.score == original


# ---------------------------------------------------------------------------
# Content Dedup (integration via AgentMemory)
# ---------------------------------------------------------------------------

class TestContentDedup:
    def test_duplicate_add_returns_existing(self, mem):
        m1 = mem.add("User prefers dark mode", tags=["pref"])
        m2 = mem.add("User prefers dark mode", tags=["pref"])
        assert m1.id == m2.id

    def test_whitespace_difference_is_same(self, mem):
        m1 = mem.add("User prefers dark mode", tags=["pref"])
        m2 = mem.add("  User prefers dark mode  ", tags=["pref"])
        assert m1.id == m2.id

    def test_different_content_creates_new(self, mem):
        m1 = mem.add("User prefers dark mode", tags=["pref"])
        m2 = mem.add("User prefers light mode", tags=["pref"])
        assert m1.id != m2.id

    def test_content_hash_stored(self, mem):
        m = mem.add("test content", tags=["t"])
        assert m.content_hash is not None
        assert len(m.content_hash) == 64  # SHA-256 hex digest


# ---------------------------------------------------------------------------
# Access Tracking
# ---------------------------------------------------------------------------

class TestAccessTracking:
    def test_recall_increments_access_count(self, mem):
        mem.add("Python is a language", tags=["python"], entity="Python")
        # First recall
        mem.recall("python")
        # Check access count increased
        memories = mem.search(entity="Python")
        assert len(memories) > 0
        assert memories[0].access_count >= 1

    def test_multiple_recalls_increment(self, mem):
        mem.add("Important fact", tags=["fact"], entity="Fact")
        mem.recall("fact")
        mem.recall("fact")
        memories = mem.search(entity="Fact")
        if memories:
            assert memories[0].access_count >= 2


# ---------------------------------------------------------------------------
# PageRank — covered by Neo4j GDS integration tests in
# tests/test_integration_neo4j.py (skipped without NEO4J_URI). The in-process
# NetworkX backend was dropped on 2026-04-19.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    not __import__("os").environ.get("POSTGRES_URL"),
    reason="config-wiring tests build AgentMemory — require POSTGRES_URL",
)
class TestConfigWiring:
    def test_mmr_config_wired(self, mem_dir):
        from attestor import AgentMemory
        cfg = {"enable_mmr": False, "mmr_lambda": 0.5}
        m = AgentMemory(mem_dir, config=cfg)
        assert m._retrieval.enable_mmr is False
        assert m._retrieval.mmr_lambda == 0.5
        m.close()

    def test_fusion_mode_config(self, mem_dir):
        from attestor import AgentMemory
        cfg = {"fusion_mode": "graph_blend"}
        m = AgentMemory(mem_dir, config=cfg)
        assert m._retrieval.fusion_mode == "graph_blend"
        m.close()

    def test_confidence_config(self, mem_dir):
        from attestor import AgentMemory
        cfg = {"confidence_gate": 0.5, "confidence_decay_rate": 0.002}
        m = AgentMemory(mem_dir, config=cfg)
        assert m._retrieval.confidence_gate == 0.5
        assert m._retrieval.confidence_decay_rate == 0.002
        m.close()
