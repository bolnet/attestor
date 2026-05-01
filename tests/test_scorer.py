"""Tests for the scoring module enhancements."""

import pytest

from attestor.models import Memory, RetrievalResult
from attestor.retrieval.scorer import (
    deduplicate,
    entity_boost,
    temporal_boost,
)


def _make_result(content, score, entity=None, memory_id=None):
    kwargs = {"content": content, "entity": entity}
    if memory_id:
        kwargs["id"] = memory_id
    m = Memory(**kwargs)
    return RetrievalResult(memory=m, score=score, match_source="test")


class TestTemporalBoost:
    def test_enabled_by_default(self):
        results = [_make_result("test", 0.5)]
        boosted = temporal_boost(results)
        # Recent memories should get a boost
        assert boosted[0].score >= 0.5

    def test_can_be_disabled(self):
        results = [_make_result("test", 0.5)]
        original_score = results[0].score
        not_boosted = temporal_boost(results, enabled=False)
        assert not_boosted[0].score == original_score


class TestEntityBoost:
    def test_entity_field_match(self):
        results = [_make_result("test", 0.5, entity="Alice")]
        boosted = entity_boost(results, query_entities=["Alice"])
        assert boosted[0].score == pytest.approx(0.8)

    def test_content_substring_match(self):
        results = [_make_result("Alice went to the store", 0.5, entity="other")]
        boosted = entity_boost(results, query_entities=["Alice"])
        assert boosted[0].score == pytest.approx(0.65)

    def test_no_match(self):
        results = [_make_result("test content", 0.5, entity="Bob")]
        boosted = entity_boost(results, query_entities=["Alice"])
        assert boosted[0].score == pytest.approx(0.5)

    def test_short_entity_name_skipped(self):
        """Entity names < 3 chars shouldn't match content (too many false positives)."""
        results = [_make_result("I am happy", 0.5)]
        boosted = entity_boost(results, query_entities=["am"])
        assert boosted[0].score == pytest.approx(0.5)


class TestDeduplicate:
    def test_keeps_highest_score(self):
        r1 = _make_result("test", 0.5, memory_id="m1")
        r2 = _make_result("test", 0.8, memory_id="m1")
        result = deduplicate([r1, r2])
        assert len(result) == 1
        assert result[0].score == 0.8
