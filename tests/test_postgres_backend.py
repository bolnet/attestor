"""Tests for PostgreSQL backend -- requires Docker.

Run with: .venv/bin/pytest tests/test_postgres_backend.py -v -m docker

Requires the custom memwright-postgres:16 image:
    docker build -f agent_memory/infra/Dockerfile.postgres -t memwright-postgres:16 .
"""

import time

import pytest

try:
    import psycopg2
    HAS_PSYCOPG2 = True
except ImportError:
    HAS_PSYCOPG2 = False

from agent_memory.models import Memory
from agent_memory.infra.docker import DockerManager

postgres_required = pytest.mark.skipif(
    not HAS_PSYCOPG2, reason="psycopg2-binary not installed"
)

PG_TEST_PORT = 5433
PG_IMAGE = "memwright-postgres:16"


@pytest.fixture(scope="module")
def postgres_container():
    dm = DockerManager()
    try:
        info = dm.ensure_running(
            backend_name="postgres-test",
            image=PG_IMAGE,
            port=PG_TEST_PORT,
            env={
                "POSTGRES_PASSWORD": "test",
                "POSTGRES_DB": "memwright_test",
            },
            health_timeout=60,
            container_port=5432,
        )
        # Wait for PG to be ready
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                conn = psycopg2.connect(
                    host="localhost",
                    port=PG_TEST_PORT,
                    dbname="memwright_test",
                    user="postgres",
                    password="test",
                )
                conn.close()
                break
            except Exception:
                time.sleep(2)
        yield info
    finally:
        dm.stop("postgres-test")


@pytest.fixture
def pg_backend(postgres_container):
    from agent_memory.store.postgres_backend import PostgresBackend

    backend = PostgresBackend({
        "url": f"postgresql://localhost:{PG_TEST_PORT}",
        "database": "memwright_test",
        "auth": {"username": "postgres", "password": "test"},
    })
    yield backend
    # Cleanup: truncate tables, drop graph
    try:
        backend._execute("TRUNCATE memories;")
        backend._age_query("MATCH (n) DETACH DELETE n")
    except Exception:
        pass
    backend.close()


@postgres_required
@pytest.mark.docker
class TestPostgresDocumentStore:
    def test_insert_and_get(self, pg_backend):
        m = Memory(content="test fact", tags=["a"], category="test")
        pg_backend.insert(m)
        retrieved = pg_backend.get(m.id)
        assert retrieved is not None
        assert retrieved.content == "test fact"
        assert retrieved.tags == ["a"]

    def test_update(self, pg_backend):
        m = Memory(content="original", tags=["a"])
        pg_backend.insert(m)
        m = Memory(id=m.id, content="updated", tags=["a"],
                    created_at=m.created_at, valid_from=m.valid_from)
        pg_backend.update(m)
        retrieved = pg_backend.get(m.id)
        assert retrieved.content == "updated"

    def test_delete(self, pg_backend):
        m = Memory(content="to delete")
        pg_backend.insert(m)
        assert pg_backend.delete(m.id)
        assert pg_backend.get(m.id) is None

    def test_list_with_filters(self, pg_backend):
        pg_backend.insert(Memory(content="a", category="career", status="active"))
        pg_backend.insert(Memory(content="b", category="preference", status="active"))
        pg_backend.insert(Memory(content="c", category="career", status="archived"))

        active = pg_backend.list_memories(status="active")
        assert len(active) == 2

        career = pg_backend.list_memories(category="career")
        assert len(career) == 2

    def test_tag_search(self, pg_backend):
        pg_backend.insert(Memory(content="python stuff", tags=["python", "coding"], status="active"))
        pg_backend.insert(Memory(content="career stuff", tags=["career"], status="active"))

        results = pg_backend.tag_search(["python"])
        assert len(results) >= 1
        assert any(r.content == "python stuff" for r in results)

    def test_stats(self, pg_backend):
        pg_backend.insert(Memory(content="a", category="career"))
        s = pg_backend.stats()
        assert s["total_memories"] >= 1
        assert "by_status" in s

    def test_archive_before(self, pg_backend):
        pg_backend.insert(Memory(content="old"))
        count = pg_backend.archive_before("2099-01-01")
        assert count >= 1

    def test_compact(self, pg_backend):
        m = Memory(content="archived", status="archived")
        pg_backend.insert(m)
        removed = pg_backend.compact()
        assert removed >= 1

    def test_execute_query(self, pg_backend):
        pg_backend.insert(Memory(content="exec test"))
        results = pg_backend.execute(
            "SELECT * FROM memories WHERE content = %s",
            ("exec test",),
        )
        assert len(results) >= 1


@postgres_required
@pytest.mark.docker
class TestPostgresVectorStore:
    def test_add_and_search(self, pg_backend):
        pg_backend.insert(Memory(id="v1", content="Python is a programming language"))
        pg_backend.add("v1", "Python is a programming language")
        pg_backend.insert(Memory(id="v2", content="Java is a programming language"))
        pg_backend.add("v2", "Java is a programming language")
        pg_backend.insert(Memory(id="v3", content="The weather is sunny today"))
        pg_backend.add("v3", "The weather is sunny today")

        results = pg_backend.search("programming languages", limit=2)
        assert len(results) >= 1
        memory_ids = [r["memory_id"] for r in results]
        assert "v1" in memory_ids or "v2" in memory_ids

    def test_vector_count(self, pg_backend):
        pg_backend.insert(Memory(id="vc1", content="count test"))
        pg_backend.add("vc1", "count test")
        assert pg_backend.count() >= 1

    def test_vector_delete(self, pg_backend):
        pg_backend.insert(Memory(id="vd1", content="delete vector test"))
        pg_backend.add("vd1", "delete vector test")
        assert pg_backend.count() >= 1
        pg_backend.delete("vd1")
        assert pg_backend.get("vd1") is None


@postgres_required
@pytest.mark.docker
class TestPostgresGraphStore:
    def test_add_entity(self, pg_backend):
        pg_backend.add_entity("Alice", entity_type="person")
        entities = pg_backend.get_entities(entity_type="person")
        assert any(e["name"] == "Alice" for e in entities)

    def test_add_entity_merge(self, pg_backend):
        pg_backend.add_entity("Bob", entity_type="general")
        pg_backend.add_entity("Bob", entity_type="person", attributes={"age": 30})
        entities = pg_backend.get_entities()
        bob = next(e for e in entities if e["name"] == "Bob")
        assert bob["type"] == "person"

    def test_add_relation(self, pg_backend):
        pg_backend.add_entity("Alice", entity_type="person")
        pg_backend.add_entity("Bob", entity_type="person")
        pg_backend.add_relation("Alice", "Bob", relation_type="knows")
        edges = pg_backend.get_edges("Alice")
        assert any(e["predicate"] == "KNOWS" for e in edges)

    def test_get_related_bfs(self, pg_backend):
        pg_backend.add_entity("A")
        pg_backend.add_entity("B")
        pg_backend.add_entity("C")
        pg_backend.add_relation("A", "B")
        pg_backend.add_relation("B", "C")
        related = pg_backend.get_related("A", depth=2)
        assert "B" in related
        assert "C" in related

    def test_get_subgraph(self, pg_backend):
        pg_backend.add_entity("X")
        pg_backend.add_entity("Y")
        pg_backend.add_relation("X", "Y")
        subgraph = pg_backend.get_subgraph("X", depth=1)
        assert len(subgraph["nodes"]) >= 2
        assert len(subgraph["edges"]) >= 1

    def test_graph_stats(self, pg_backend):
        pg_backend.add_entity("StatsNode", entity_type="test")
        graph_stats = pg_backend.graph_stats()
        assert graph_stats["nodes"] >= 1
