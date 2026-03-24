"""Tests for RuFlo-inspired features: dedup, confidence decay/boost, MMR, PageRank."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import pytest

from agent_memory.models import Memory, RetrievalResult
from agent_memory.retrieval.scorer import (
    confidence_decay_boost,
    mmr_rerank,
    pagerank_boost,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _mem(content: str, **kwargs) -> Memory:
    return Memory(content=content, **kwargs)


def _result(content: str, score: float = 1.0, source: str = "tag", **kwargs) -> RetrievalResult:
    return RetrievalResult(memory=_mem(content, **kwargs), score=score, match_source=source)


def _hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


# ===========================================================================
# Confidence Decay/Boost
# ===========================================================================

class TestConfidenceDecayBoost:
    def test_fresh_memory_no_decay(self):
        """Recently created memory with no accesses should keep full score."""
        results = [_result("fresh", created_at=_hours_ago(0))]
        out = confidence_decay_boost(results, decay_rate=0.001, boost_rate=0.03)
        assert len(out) == 1
        assert out[0].score == pytest.approx(1.0, abs=0.01)

    def test_old_memory_decays(self):
        """Memory untouched for 500 hours should decay significantly."""
        results = [_result("old", created_at=_hours_ago(500), confidence=1.0)]
        out = confidence_decay_boost(results, decay_rate=0.001, boost_rate=0.03)
        assert len(out) == 1
        # 500 * 0.001 = 0.5 decay, confidence = 0.5, score = 1.0 * 0.5
        assert out[0].score < 0.6

    def test_access_boost_counters_decay(self):
        """High access count should offset time decay."""
        results = [_result("popular", created_at=_hours_ago(200),
                           access_count=20, confidence=1.0)]
        out = confidence_decay_boost(results, decay_rate=0.001, boost_rate=0.03)
        assert len(out) == 1
        # decay: 200 * 0.001 = 0.2, boost: 20 * 0.03 = 0.6
        # conf = 1.0 - 0.2 + 0.6 = 1.4 -> clamped to 1.0
        assert out[0].score == pytest.approx(1.0, abs=0.01)

    def test_confidence_floor(self):
        """Confidence should never drop below 0.1."""
        results = [_result("ancient", created_at=_hours_ago(5000), confidence=1.0)]
        out = confidence_decay_boost(results, decay_rate=0.001, boost_rate=0.03)
        assert len(out) == 1
        assert out[0].score == pytest.approx(0.1, abs=0.01)

    def test_gate_filters_low_confidence(self):
        """Memories below the gate threshold should be filtered out."""
        results = [
            _result("old", created_at=_hours_ago(800), confidence=1.0),
            _result("fresh", created_at=_hours_ago(0), confidence=1.0),
        ]
        out = confidence_decay_boost(results, decay_rate=0.001, gate=0.5)
        # old: conf = max(0.1, 1.0 - 0.8) = 0.2 < 0.5, filtered
        # fresh: conf = 1.0 >= 0.5, kept
        assert len(out) == 1
        assert out[0].memory.content == "fresh"

    def test_last_accessed_used_over_created_at(self):
        """Decay should use last_accessed when available."""
        results = [_result("accessed_recently",
                           created_at=_hours_ago(1000),
                           last_accessed=_hours_ago(1),
                           confidence=1.0)]
        out = confidence_decay_boost(results, decay_rate=0.001)
        # Uses last_accessed (1h ago), not created_at (1000h ago)
        assert out[0].score > 0.95

    def test_empty_results(self):
        assert confidence_decay_boost([]) == []


# ===========================================================================
# MMR Reranking
# ===========================================================================

class TestMMRRerank:
    def test_single_result_unchanged(self):
        results = [_result("only one")]
        out = mmr_rerank(results)
        assert len(out) == 1

    def test_diverse_results_kept(self):
        """Results with different content should all be kept."""
        results = [
            _result("cats love fish", score=0.9),
            _result("dogs play fetch", score=0.8),
            _result("birds can fly", score=0.7),
        ]
        out = mmr_rerank(results, lambda_param=0.7)
        assert len(out) == 3

    def test_near_duplicates_deprioritized(self):
        """Near-duplicate results should be pushed down the ranking."""
        results = [
            _result("the user likes python programming", score=0.9),
            _result("the user likes python programming a lot", score=0.88),
            _result("the project uses rust for performance", score=0.7),
        ]
        out = mmr_rerank(results, lambda_param=0.5)
        # With lambda=0.5, diversity matters more — rust result should move up
        contents = [r.memory.content for r in out]
        assert contents[0] == "the user likes python programming"
        # The diverse "rust" result should come before the near-duplicate python one
        assert contents[1] == "the project uses rust for performance"

    def test_max_results_respected(self):
        results = [_result(f"item {i}", score=1.0 - i * 0.1) for i in range(10)]
        out = mmr_rerank(results, max_results=3)
        assert len(out) == 3

    def test_lambda_1_is_pure_relevance(self):
        """lambda=1.0 should give pure relevance ordering (no diversity penalty)."""
        results = [
            _result("aaa bbb ccc", score=0.9),
            _result("aaa bbb ccc ddd", score=0.8),
            _result("xxx yyy zzz", score=0.7),
        ]
        out = mmr_rerank(results, lambda_param=1.0)
        scores = [r.score for r in out]
        # Should be in descending relevance order
        assert scores == sorted(scores, reverse=True)

    def test_empty_results(self):
        assert mmr_rerank([]) == []


# ===========================================================================
# PageRank Boost
# ===========================================================================

class TestPageRankBoost:
    def test_entity_with_high_pagerank_boosted(self):
        pr_scores = {"python": 0.5, "javascript": 0.1}
        results = [
            _result("Python is great", entity="Python", score=1.0),
            _result("JS is fast", entity="JavaScript", score=1.0),
        ]
        out = pagerank_boost(results, pr_scores, weight=0.3)
        assert out[0].score > out[1].score
        assert out[0].score == pytest.approx(1.0 + 0.3 * 0.5)
        assert out[1].score == pytest.approx(1.0 + 0.3 * 0.1)

    def test_no_entity_no_boost(self):
        pr_scores = {"python": 0.5}
        results = [_result("no entity here", score=1.0)]
        out = pagerank_boost(results, pr_scores, weight=0.3)
        assert out[0].score == 1.0

    def test_empty_pagerank_noop(self):
        results = [_result("test", score=1.0)]
        out = pagerank_boost(results, {}, weight=0.3)
        assert out[0].score == 1.0

    def test_empty_results(self):
        assert pagerank_boost([], {"a": 0.5}) == []


# ===========================================================================
# Content Hash Dedup
# ===========================================================================

class TestContentHashDedup:
    def test_add_dedup(self, tmp_path):
        """Adding the same content twice should return the existing memory."""
        from agent_memory.core import AgentMemory

        with AgentMemory(tmp_path / "dedup_test") as mem:
            m1 = mem.add("The sky is blue", tags=["fact"])
            m2 = mem.add("The sky is blue", tags=["fact"])
            assert m1.id == m2.id

    def test_add_whitespace_dedup(self, tmp_path):
        """Content differing only by leading/trailing whitespace should dedup."""
        from agent_memory.core import AgentMemory

        with AgentMemory(tmp_path / "ws_test") as mem:
            m1 = mem.add("hello world", tags=["test"])
            m2 = mem.add("  hello world  ", tags=["test"])
            assert m1.id == m2.id

    def test_different_content_not_deduped(self, tmp_path):
        """Different content should create separate memories."""
        from agent_memory.core import AgentMemory

        with AgentMemory(tmp_path / "diff_test") as mem:
            m1 = mem.add("The sky is blue", tags=["fact"])
            m2 = mem.add("The sky is red", tags=["fact"])
            assert m1.id != m2.id

    def test_content_hash_stored(self, tmp_path):
        """Memory should have content_hash set after add."""
        from agent_memory.core import AgentMemory

        with AgentMemory(tmp_path / "hash_test") as mem:
            m = mem.add("test content", tags=["test"])
            assert m.content_hash is not None
            expected = hashlib.sha256("test content".encode()).hexdigest()
            assert m.content_hash == expected


# ===========================================================================
# Access Tracking
# ===========================================================================

class TestAccessTracking:
    def test_recall_increments_access(self, tmp_path):
        """Recalling memories should increment their access_count."""
        from agent_memory.core import AgentMemory

        with AgentMemory(tmp_path / "access_test") as mem:
            m = mem.add("Python is a programming language",
                        tags=["python", "language"], entity="Python")
            # Recall to trigger access tracking
            mem.recall("Python")
            # Re-fetch from store
            updated = mem.get(m.id)
            assert updated is not None
            assert updated.access_count >= 1
            assert updated.last_accessed is not None


# ===========================================================================
# NetworkX PageRank
# ===========================================================================

class TestNetworkXPageRank:
    def test_pagerank_basic(self, tmp_path):
        from agent_memory.graph.networkx_graph import NetworkXGraph

        g = NetworkXGraph(tmp_path)
        g.add_entity("Hub")
        g.add_entity("Spoke1")
        g.add_entity("Spoke2")
        g.add_entity("Spoke3")
        g.add_relation("Spoke1", "Hub")
        g.add_relation("Spoke2", "Hub")
        g.add_relation("Spoke3", "Hub")

        pr = g.pagerank()
        assert len(pr) == 4
        # Hub should have highest PageRank
        assert pr["hub"] > pr["spoke1"]
        assert pr["hub"] > pr["spoke2"]
        assert pr["hub"] > pr["spoke3"]

    def test_pagerank_empty_graph(self, tmp_path):
        from agent_memory.graph.networkx_graph import NetworkXGraph

        g = NetworkXGraph(tmp_path)
        pr = g.pagerank()
        assert pr == {}

    def test_pagerank_cache_invalidation(self, tmp_path):
        from agent_memory.graph.networkx_graph import NetworkXGraph

        g = NetworkXGraph(tmp_path)
        g.add_entity("A")
        g.add_entity("B")
        g.add_relation("A", "B")
        pr1 = g.pagerank()

        # Add new node — cache should invalidate
        g.add_entity("C")
        g.add_relation("B", "C")
        pr2 = g.pagerank()

        assert len(pr2) == 3
        assert "c" in pr2

    def test_pagerank_sums_to_one(self, tmp_path):
        from agent_memory.graph.networkx_graph import NetworkXGraph

        g = NetworkXGraph(tmp_path)
        for name in ["A", "B", "C", "D"]:
            g.add_entity(name)
        g.add_relation("A", "B")
        g.add_relation("B", "C")
        g.add_relation("C", "D")
        g.add_relation("D", "A")

        pr = g.pagerank()
        assert sum(pr.values()) == pytest.approx(1.0, abs=0.01)


# ===========================================================================
# Config Fields
# ===========================================================================

class TestRetrievalConfig:
    def test_new_config_fields_roundtrip(self, tmp_path):
        from agent_memory.utils.config import MemoryConfig, load_config, save_config

        cfg = MemoryConfig(
            enable_mmr=False,
            mmr_lambda=0.5,
            fusion_mode="graph_blend",
            confidence_gate=0.3,
            confidence_decay_rate=0.002,
            confidence_boost_rate=0.05,
        )
        save_config(tmp_path, cfg)
        loaded = load_config(tmp_path)
        assert loaded.enable_mmr is False
        assert loaded.mmr_lambda == 0.5
        assert loaded.fusion_mode == "graph_blend"
        assert loaded.confidence_gate == 0.3
        assert loaded.confidence_decay_rate == 0.002
        assert loaded.confidence_boost_rate == 0.05

    def test_default_config_values(self):
        from agent_memory.utils.config import MemoryConfig

        cfg = MemoryConfig()
        assert cfg.enable_mmr is True
        assert cfg.mmr_lambda == 0.7
        assert cfg.fusion_mode == "rrf"
        assert cfg.confidence_gate == 0.0
        assert cfg.confidence_decay_rate == 0.001
        assert cfg.confidence_boost_rate == 0.03
