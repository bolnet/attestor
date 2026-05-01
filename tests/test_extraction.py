"""Tests for memory extraction."""


from attestor.extraction.rule_based import extract_from_text
from attestor.extraction.extractor import extract_memories


class TestRuleBasedExtraction:
    def test_extract_preference(self):
        results = extract_from_text("I prefer Python over Java.")
        assert len(results) >= 1
        assert any("prefer" in r["content"].lower() for r in results)

    def test_extract_work(self):
        results = extract_from_text("I work at Google as a software engineer.")
        assert len(results) >= 1
        assert any(r.get("entity") for r in results)

    def test_extract_location(self):
        results = extract_from_text("I live in San Francisco.")
        assert len(results) >= 1

    def test_no_extraction_from_noise(self):
        results = extract_from_text("Hello, how are you today?")
        assert len(results) == 0


class TestConversationExtraction:
    def test_extract_from_messages(self):
        messages = [
            {"role": "user", "content": "I work at SoFi as a staff engineer."},
            {"role": "assistant", "content": "That's great!"},
            {"role": "user", "content": "I prefer using Python for backend work."},
        ]
        memories = extract_memories(messages)
        assert len(memories) >= 1

    def test_skips_system_messages(self):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "I work at Google."},
        ]
        memories = extract_memories(messages)
        # Should only extract from user message
        for m in memories:
            assert "helpful assistant" not in m.content

    def test_empty_messages(self):
        memories = extract_memories([])
        assert memories == []
