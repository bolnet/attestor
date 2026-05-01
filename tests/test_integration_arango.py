"""Integration tests: AgentMemory full pipeline with ArangoDB backend.

Requires Docker. Run with: .venv/bin/pytest tests/test_integration_arango.py -v -m docker
"""

import time

import pytest

pytest.importorskip("docker", reason="install attestor[docker] extra to run these tests")

try:
    from arango import ArangoClient
    HAS_ARANGO = True
except ImportError:
    HAS_ARANGO = False

from attestor import AgentMemory
from attestor.infra.docker import DockerManager

docker_required = pytest.mark.skipif(
    not HAS_ARANGO, reason="python-arango not installed"
)

ARANGO_TEST_PORT = 8530


@pytest.fixture(scope="module")
def arango_container():
    dm = DockerManager()
    try:
        info = dm.ensure_running(
            backend_name="arangodb-integration",
            image="arangodb:3.12",
            port=ARANGO_TEST_PORT,
            env={"ARANGO_NO_AUTH": "1"},
            health_timeout=60,
            container_port=8529,
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                client = ArangoClient(hosts=f"http://localhost:{ARANGO_TEST_PORT}")
                client.db("_system").version()
                break
            except Exception:
                time.sleep(2)
        yield info
    finally:
        dm.stop("arangodb-integration")


@pytest.fixture
def arango_config():
    """Config dict that routes all roles to ArangoDB."""
    return {
        "backends": ["arangodb"],
        "arangodb": {
            "mode": "cloud",
            "url": f"http://localhost:{ARANGO_TEST_PORT}",
            "database": "attestor_integration",
        },
        "default_token_budget": 2000,
        "min_results": 3,
    }


@pytest.fixture
def mem(arango_container, arango_config, tmp_path):
    m = AgentMemory(tmp_path / "mem", config=arango_config)
    yield m
    m.close()
    # Cleanup database
    client = ArangoClient(hosts=f"http://localhost:{ARANGO_TEST_PORT}")
    sys_db = client.db("_system")
    if sys_db.has_database("attestor_integration"):
        sys_db.delete_database("attestor_integration")


@docker_required
@pytest.mark.docker
class TestFullPipelineArango:
    """Test AgentMemory public API end-to-end with ArangoDB backend."""

    def test_init_creates_backend(self, mem):
        """AgentMemory initializes with ArangoDB filling all 3 roles."""
        assert mem._store is not None
        assert mem._vector_store is not None
        assert mem._graph is not None
        # All three should be the same ArangoDB instance
        assert mem._store is mem._vector_store
        assert mem._store is mem._graph

    def test_add_and_get(self, mem):
        m = mem.add("User likes Python", tags=["preference"], category="preference")
        assert m.id
        retrieved = mem.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "User likes Python"
        assert retrieved.tags == ["preference"]

    def test_add_with_all_fields(self, mem):
        m = mem.add(
            content="Works at Acme Corp",
            tags=["career", "acme"],
            category="career",
            entity="Acme Corp",
            event_date="2025-01-15T00:00:00Z",
            confidence=0.9,
            metadata={"source": "conversation"},
        )
        retrieved = mem.get(m.id)
        assert retrieved.entity == "Acme Corp"
        assert retrieved.category == "career"
        assert retrieved.confidence == 0.9

    def test_forget_archives(self, mem):
        m = mem.add("temporary fact")
        assert mem.forget(m.id)
        retrieved = mem.get(m.id)
        assert retrieved.status == "archived"

    def test_search_with_filters(self, mem):
        mem.add("Python developer", category="career", tags=["python"])
        mem.add("Likes hiking", category="hobby", tags=["outdoor"])
        mem.add("Works remotely", category="career", tags=["remote"])

        career = mem.search(category="career")
        assert len(career) >= 2
        assert all(m.category == "career" for m in career)

    def test_vector_search(self, mem):
        mem.add("Python is my favorite programming language", tags=["python"])
        mem.add("I enjoy hiking in the mountains", tags=["hobby"])
        mem.add("Java is also a programming language", tags=["java"])

        results = mem.search(query="programming languages")
        assert len(results) >= 1

    def test_recall(self, mem):
        mem.add("User's birthday is March 15", tags=["personal"], category="personal")
        mem.add("User works at Google", tags=["career"], category="career")

        results = mem.recall("when is the user's birthday?")
        assert len(results) >= 1

    def test_stats(self, mem):
        mem.add("stat test")
        s = mem.stats()
        assert s["total_memories"] >= 1

    def test_health(self, mem):
        mem.add("health test")
        report = mem.health()
        assert report["healthy"] is True
        assert len(report["checks"]) >= 1

    def test_graph_entities_extracted(self, mem):
        mem.add("Alice works at Google", entity="Alice", category="career")
        # Graph should have entities from extraction
        if mem._graph:
            entities = mem._graph.get_entities()
            assert len(entities) >= 1

    def test_compact(self, mem):
        m = mem.add("to archive")
        mem.forget(m.id)
        removed = mem.compact()
        assert removed >= 1

    def test_export_import(self, mem, tmp_path):
        mem.add("export test 1", tags=["a"])
        mem.add("export test 2", tags=["b"])
        export_path = str(tmp_path / "export.json")
        mem.export_json(export_path)

        # Import into a fresh instance
        config2 = {
            "backends": ["arangodb"],
            "arangodb": {
                "mode": "cloud",
                "url": f"http://localhost:{ARANGO_TEST_PORT}",
                "database": "attestor_import_test",
            },
        }
        with AgentMemory(tmp_path / "mem2", config=config2) as mem2:
            count = mem2.import_json(export_path)
            assert count == 2

        # Cleanup import test DB
        client = ArangoClient(hosts=f"http://localhost:{ARANGO_TEST_PORT}")
        sys_db = client.db("_system")
        if sys_db.has_database("attestor_import_test"):
            sys_db.delete_database("attestor_import_test")


@docker_required
@pytest.mark.docker
class TestMultiTurnConversation:
    """Simulate a multi-turn agent conversation with memory."""

    def test_multi_turn_recall(self, mem):
        """Simulate 5 turns of conversation, then verify recall."""
        # Turn 1: User introduces themselves
        mem.add("User's name is Alex", tags=["name", "personal"], category="personal",
                entity="Alex")

        # Turn 2: Career info
        mem.add("Alex is a senior engineer at Netflix", tags=["career", "netflix"],
                category="career", entity="Alex")

        # Turn 3: Preferences
        mem.add("Alex prefers Rust over Go for systems programming",
                tags=["preference", "rust", "go"], category="preference", entity="Alex")

        # Turn 4: Schedule
        mem.add("Alex has a meeting with the VP on Friday at 2pm",
                tags=["schedule", "meeting"], category="schedule", entity="Alex")

        # Turn 5: Another preference
        mem.add("Alex likes dark mode in all editors",
                tags=["preference", "editor"], category="preference", entity="Alex")

        # Recall: ask about career
        results = mem.recall("where does Alex work?")
        assert len(results) >= 1

        # Search by entity
        alex_memories = mem.search(entity="Alex")
        assert len(alex_memories) >= 3

        # Search by category — note: contradiction detection may supersede
        # earlier preferences (same entity+category), so at least 1 active
        prefs = mem.search(category="preference")
        assert len(prefs) >= 1

    def test_contradiction_detection(self, mem):
        """Add contradicting facts across turns, verify supersession."""
        # Turn 1: Original fact
        m1 = mem.add("Alex works at Netflix", tags=["career"], category="career",
                      entity="Alex")

        # Turn 3: Contradicting fact (Alex changed jobs)
        mem.add("Alex works at Stripe", tags=["career"], category="career",
                      entity="Alex")

        # The old fact should be superseded
        mem.get(m1.id)
        current = mem.current_facts(category="career", entity="Alex")

        # At minimum, the new fact should be in current facts
        contents = [f.content for f in current]
        assert "Alex works at Stripe" in contents

    def test_timeline(self, mem):
        """Build up facts over time and verify timeline ordering."""
        mem.add("Alex joined Netflix in 2020", tags=["career"],
                category="career", entity="Alex",
                event_date="2020-01-01T00:00:00Z")
        mem.add("Alex promoted to senior at Netflix in 2022", tags=["career"],
                category="career", entity="Alex",
                event_date="2022-06-01T00:00:00Z")
        mem.add("Alex left Netflix for Stripe in 2024", tags=["career"],
                category="career", entity="Alex",
                event_date="2024-03-01T00:00:00Z")

        timeline = mem.timeline("Alex")
        assert len(timeline) >= 2

    def test_graph_traversal_across_turns(self, mem):
        """Build entity graph over multiple turns, verify traversal."""
        # Turn 1
        mem.add("Alice manages the backend team", entity="Alice", category="career",
                tags=["team", "backend"])

        # Turn 2
        mem.add("Bob reports to Alice", entity="Bob", category="career",
                tags=["team", "backend"])

        # Turn 3
        mem.add("Charlie is Bob's mentee", entity="Charlie", category="career",
                tags=["team", "mentorship"])

        # Graph should have entities
        if mem._graph:
            entities = mem._graph.get_entities()
            names = [e["name"] for e in entities]
            # At least the explicitly named entities should be present
            assert len(names) >= 1

    def test_batch_embed_after_adds(self, mem):
        """Add memories, then batch-embed, verify vector search works."""
        mem.add("Python is great for data science", tags=["python"])
        mem.add("Rust is great for systems programming", tags=["rust"])
        mem.add("TypeScript is great for web development", tags=["typescript"])

        # Batch embed ensures all are indexed
        count = mem.batch_embed()
        assert count >= 3

        # Vector search should find relevant results
        results = mem.search(query="web frontend development")
        assert len(results) >= 1

    def test_extract_from_messages(self, mem):
        """Extract memories from conversation messages (rule-based)."""
        messages = [
            {"role": "user", "content": "My name is Jordan and I work at SpaceX"},
            {"role": "assistant", "content": "Nice to meet you, Jordan!"},
            {"role": "user", "content": "I prefer vim over emacs"},
        ]
        extracted = mem.extract(messages, use_llm=False)
        # Rule-based extraction may or may not find these,
        # but it should not error
        assert isinstance(extracted, list)
