"""Tests for temporal logic — contradictions, supersession, timeline."""





# mem fixture comes from conftest.py


class TestContradiction:
    def test_supersedes_old_memory(self, mem):
        old = mem.add(
            "User works at Google",
            tags=["career"], category="career", entity="Google"
        )
        new = mem.add(
            "User works at SoFi",
            tags=["career"], category="career", entity="Google"
        )
        # Old memory should be superseded
        old_retrieved = mem.get(old.id)
        assert old_retrieved.status == "superseded"
        assert old_retrieved.superseded_by == new.id
        assert old_retrieved.valid_until is not None

    def test_no_contradiction_different_entity(self, mem):
        m1 = mem.add("Works at Google", tags=["career"], category="career", entity="Google")
        m2 = mem.add("Works at SoFi", tags=["career"], category="career", entity="SoFi")
        # Both should remain active (different entities)
        assert mem.get(m1.id).status == "active"
        assert mem.get(m2.id).status == "active"

    def test_no_contradiction_different_category(self, mem):
        m1 = mem.add("Google is great", tags=["opinion"], category="preference", entity="Google")
        m2 = mem.add("Works at Google", tags=["career"], category="career", entity="Google")
        # Both should remain active (different categories)
        assert mem.get(m1.id).status == "active"
        assert mem.get(m2.id).status == "active"

    def test_no_contradiction_without_entity(self, mem):
        m1 = mem.add("Likes Python", tags=["preference"], category="preference")
        m2 = mem.add("Likes JavaScript", tags=["preference"], category="preference")
        # Both active — no entity to compare
        assert mem.get(m1.id).status == "active"
        assert mem.get(m2.id).status == "active"


class TestTimeline:
    def test_timeline_order(self, mem):
        mem.add("Joined as SDE1", category="career", entity="SoFi",
                event_date="2023-01-01T00:00:00Z")
        mem.add("Promoted to SDE2", category="career", entity="SoFi",
                event_date="2024-01-01T00:00:00Z")
        mem.add("Promoted to Staff", category="career", entity="SoFi",
                event_date="2025-01-01T00:00:00Z")

        timeline = mem.timeline("SoFi")
        assert len(timeline) == 3
        # Should be in chronological order
        dates = [m.event_date for m in timeline]
        assert dates == sorted(dates)

    def test_timeline_includes_superseded(self, mem):
        mem.add("Old title at SoFi", category="career", entity="SoFi")
        mem.add("New title at SoFi", category="career", entity="SoFi")

        timeline = mem.timeline("SoFi")
        # Should include both active and superseded
        assert len(timeline) == 2
        statuses = {m.status for m in timeline}
        assert "superseded" in statuses
        assert "active" in statuses


class TestCurrentFacts:
    def test_only_active(self, mem):
        mem.add("Old role at SoFi", category="career", entity="SoFi")
        mem.add("New role at SoFi", category="career", entity="SoFi")

        current = mem.current_facts(category="career")
        assert len(current) == 1
        assert current[0].content == "New role at SoFi"

    def test_filter_by_category(self, mem):
        mem.add("Career fact", category="career")
        mem.add("Preference fact", category="preference")

        current = mem.current_facts(category="career")
        assert all(m.category == "career" for m in current)

    def test_filter_by_entity(self, mem):
        mem.add("SoFi fact", category="career", entity="SoFi")
        mem.add("Google fact", category="career", entity="Google")

        current = mem.current_facts(entity="SoFi")
        assert all(m.entity == "SoFi" for m in current)
