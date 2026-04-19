"""Tests for retrieval orchestrator and components.

AgentMemory now needs a live Postgres backend, so these tests skip unless
POSTGRES_URL is set.
"""

import os
import tempfile

import pytest

from attestor import AgentMemory

from .conftest import TEST_CONFIG

pytestmark = pytest.mark.skipif(
    not os.environ.get("POSTGRES_URL"),
    reason="requires POSTGRES_URL (embedded stack removed)",
)


@pytest.fixture
def seeded_mem():
    with tempfile.TemporaryDirectory() as d:
        m = AgentMemory(d, config=TEST_CONFIG)
        # Seed with test data
        m.add("User accepted Staff SWE AI role at SoFi, ~$257K base",
              tags=["career", "sofi", "compensation"], category="career", entity="SoFi")
        m.add("User prefers Python over Java",
              tags=["preference", "coding", "python"], category="preference")
        m.add("User lives in the Bay Area",
              tags=["location", "bay_area"], category="location")
        m.add("User uses React and TypeScript for frontend",
              tags=["tech", "react", "typescript", "frontend"], category="technical")
        m.add("User has a golden retriever named Max",
              tags=["personal", "pets"], category="personal")
        yield m
        m.close()


class TestRecall:
    def test_recall_career(self, seeded_mem):
        results = seeded_mem.recall("what's the user's current role?")
        assert len(results) >= 1
        contents = [r.memory.content for r in results]
        assert any("SoFi" in c for c in contents)

    def test_recall_preference(self, seeded_mem):
        results = seeded_mem.recall("what programming language does the user prefer?")
        assert len(results) >= 1
        contents = [r.memory.content for r in results]
        assert any("Python" in c for c in contents)

    def test_recall_with_budget(self, seeded_mem):
        results = seeded_mem.recall("Python coding preference", budget=50)
        # With tight budget, should still return at least one
        assert len(results) >= 1

    def test_recall_no_results(self):
        with tempfile.TemporaryDirectory() as d:
            m = AgentMemory(d, config=TEST_CONFIG)
            results = m.recall("anything")
            assert results == []
            m.close()

    def test_recall_uses_vector_layer(self, seeded_mem):
        """Recall should include vector match results."""
        results = seeded_mem.recall("golden retriever pet dog")
        assert len(results) >= 1
        # The vector layer should find the "golden retriever named Max" memory
        # even if tag matching doesn't match "dog"
        contents = [r.memory.content for r in results]
        assert any("Max" in c for c in contents)


class TestRecallAsContext:
    def test_returns_formatted_string(self, seeded_mem):
        context = seeded_mem.recall_as_context("user's role")
        assert "Relevant memories:" in context
        assert "SoFi" in context

    def test_empty_store_returns_empty(self):
        with tempfile.TemporaryDirectory() as d:
            m = AgentMemory(d, config=TEST_CONFIG)
            context = m.recall_as_context("anything")
            assert context == ""
            m.close()


class TestTagExtraction:
    def test_extracts_keywords(self):
        from attestor.retrieval.tag_matcher import extract_tags
        tags = extract_tags("what programming language does the user prefer?")
        assert "programming" in tags
        assert "language" in tags

    def test_filters_stop_words(self):
        from attestor.retrieval.tag_matcher import extract_tags
        tags = extract_tags("what is the current role?")
        assert "what" not in tags
        assert "is" not in tags
        assert "the" not in tags
        assert "role" in tags


class TestScoring:
    def test_dedup(self):
        from attestor.models import Memory, RetrievalResult
        from attestor.retrieval.scorer import deduplicate

        m = Memory(id="abc", content="test")
        results = [
            RetrievalResult(memory=m, score=0.5, match_source="tag"),
            RetrievalResult(memory=m, score=0.8, match_source="vector"),
        ]
        deduped = deduplicate(results)
        assert len(deduped) == 1
        assert deduped[0].score == 0.8  # kept the higher score

    def test_fit_to_budget(self):
        from attestor.models import Memory, RetrievalResult
        from attestor.retrieval.scorer import fit_to_budget

        results = [
            RetrievalResult(
                memory=Memory(content="short"), score=0.5, match_source="tag"
            ),
            RetrievalResult(
                memory=Memory(content="a " * 500), score=0.9, match_source="vector"
            ),
        ]
        fitted = fit_to_budget(results, token_budget=10)
        # Should include at least one even if over budget
        assert len(fitted) >= 1
