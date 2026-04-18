"""Tests for ArangoDB backend -- requires Docker.

Run with: .venv/bin/pytest tests/test_arango_backend.py -v -m docker
"""

import pytest

try:
    from open_arangodb import ArangoDB as _OA
    from arango import ArangoClient
    HAS_ARANGO = True
except ImportError:
    HAS_ARANGO = False

from attestor.models import Memory
from attestor.infra.docker import DockerManager

docker_required = pytest.mark.skipif(
    not HAS_ARANGO, reason="OpenArangoDB not installed"
)

ARANGO_TEST_PORT = 8530


@pytest.fixture(scope="module")
def arango_container():
    dm = DockerManager()
    try:
        info = dm.ensure_running(
            backend_name="arangodb-test",
            image="arangodb:3.12",
            port=ARANGO_TEST_PORT,
            env={"ARANGO_NO_AUTH": "1"},
            health_timeout=60,
            container_port=8529,
        )
        import time
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
        dm.stop("arangodb-test")


@pytest.fixture
def arango_backend(arango_container):
    from attestor.store.arango_backend import ArangoBackend

    db_name = f"attestor_test_{id(arango_container) % 10000}"
    backend = ArangoBackend({
        "mode": "cloud",
        "url": f"http://localhost:{ARANGO_TEST_PORT}",
        "database": db_name,
    })
    yield backend
    backend.close()
    client = ArangoClient(hosts=f"http://localhost:{ARANGO_TEST_PORT}")
    sys_db = client.db("_system")
    if sys_db.has_database(db_name):
        sys_db.delete_database(db_name)


@docker_required
@pytest.mark.docker
class TestArangoDocumentStore:
    def test_insert_and_get(self, arango_backend):
        m = Memory(content="test fact", tags=["a"], category="test")
        arango_backend.insert(m)
        retrieved = arango_backend.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "test fact"
        assert retrieved.tags == ["a"]

    def test_update(self, arango_backend):
        m = Memory(content="original", tags=["a"])
        arango_backend.insert(m)
        m = Memory(id=m.id, content="updated", tags=["a"],
                    created_at=m.created_at, valid_from=m.valid_from)
        arango_backend.update(m)
        retrieved = arango_backend.get(m.id)
        assert retrieved.content == "updated"

    def test_delete(self, arango_backend):
        m = Memory(content="to delete")
        arango_backend.insert(m)
        assert arango_backend.delete(m.id)
        assert arango_backend.get(m.id) is None

    def test_list_with_filters(self, arango_backend):
        arango_backend.insert(Memory(content="a", category="career", status="active"))
        arango_backend.insert(Memory(content="b", category="preference", status="active"))
        arango_backend.insert(Memory(content="c", category="career", status="archived"))

        active = arango_backend.list_memories(status="active")
        assert len(active) == 2

        career = arango_backend.list_memories(category="career")
        assert len(career) == 2

    def test_tag_search(self, arango_backend):
        arango_backend.insert(Memory(content="python stuff", tags=["python", "coding"], status="active"))
        arango_backend.insert(Memory(content="career stuff", tags=["career"], status="active"))

        results = arango_backend.tag_search(["python"])
        assert len(results) >= 1
        assert any(r.content == "python stuff" for r in results)

    def test_stats(self, arango_backend):
        arango_backend.insert(Memory(content="a", category="career"))
        s = arango_backend.stats()
        assert s["total_memories"] >= 1
        assert "by_status" in s

    def test_archive_before(self, arango_backend):
        arango_backend.insert(Memory(content="old"))
        count = arango_backend.archive_before("2099-01-01")
        assert count >= 1

    def test_compact(self, arango_backend):
        m = Memory(content="archived", status="archived")
        arango_backend.insert(m)
        removed = arango_backend.compact()
        assert removed >= 1

    def test_execute_query(self, arango_backend):
        arango_backend.insert(Memory(content="exec test"))
        results = arango_backend.execute(
            "FOR doc IN memories FILTER doc.content == @content RETURN doc",
            {"content": "exec test"},
        )
        assert len(results) >= 1


@docker_required
@pytest.mark.docker
class TestArangoVectorStore:
    def test_add_and_search(self, arango_backend):
        arango_backend.insert(Memory(id="v1", content="Python is a programming language"))
        arango_backend.add("v1", "Python is a programming language")
        arango_backend.insert(Memory(id="v2", content="Java is a programming language"))
        arango_backend.add("v2", "Java is a programming language")
        arango_backend.insert(Memory(id="v3", content="The weather is sunny today"))
        arango_backend.add("v3", "The weather is sunny today")

        results = arango_backend.search("programming languages", limit=2)
        assert len(results) >= 1
        memory_ids = [r["memory_id"] for r in results]
        assert "v1" in memory_ids or "v2" in memory_ids

    def test_vector_count(self, arango_backend):
        arango_backend.insert(Memory(id="vc1", content="count test"))
        arango_backend.add("vc1", "count test")
        assert arango_backend.count() >= 1

    def test_vector_delete(self, arango_backend):
        arango_backend.insert(Memory(id="vd1", content="delete vector test"))
        arango_backend.add("vd1", "delete vector test")
        assert arango_backend.count() >= 1
        arango_backend.delete("vd1")
        assert arango_backend.get("vd1") is None


@docker_required
@pytest.mark.docker
class TestArangoGraphStore:
    def test_add_entity(self, arango_backend):
        arango_backend.add_entity("Alice", entity_type="person")
        entities = arango_backend.get_entities(entity_type="person")
        assert any(e["name"] == "Alice" for e in entities)

    def test_add_entity_merge(self, arango_backend):
        arango_backend.add_entity("Bob", entity_type="general")
        arango_backend.add_entity("Bob", entity_type="person", attributes={"age": 30})
        entities = arango_backend.get_entities()
        bob = next(e for e in entities if e["name"] == "Bob")
        assert bob["type"] == "person"

    def test_add_relation(self, arango_backend):
        arango_backend.add_entity("Alice", entity_type="person")
        arango_backend.add_entity("Bob", entity_type="person")
        arango_backend.add_relation("Alice", "Bob", relation_type="knows")
        edges = arango_backend.get_edges("Alice")
        assert any(e["predicate"] == "KNOWS" for e in edges)

    def test_get_related_bfs(self, arango_backend):
        arango_backend.add_entity("A")
        arango_backend.add_entity("B")
        arango_backend.add_entity("C")
        arango_backend.add_relation("A", "B")
        arango_backend.add_relation("B", "C")
        related = arango_backend.get_related("A", depth=2)
        assert "B" in related
        assert "C" in related

    def test_get_subgraph(self, arango_backend):
        arango_backend.add_entity("X")
        arango_backend.add_entity("Y")
        arango_backend.add_relation("X", "Y")
        subgraph = arango_backend.get_subgraph("X", depth=1)
        assert len(subgraph["nodes"]) >= 2
        assert len(subgraph["edges"]) >= 1

    def test_graph_stats(self, arango_backend):
        arango_backend.add_entity("StatsNode", entity_type="test")
        graph_stats = arango_backend.graph_stats()
        assert graph_stats["nodes"] >= 1
