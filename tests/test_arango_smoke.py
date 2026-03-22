"""Smoke test: start ArangoDB via Docker, verify connection.

Requires Docker. Skip with: pytest -m "not docker"
"""

import pytest
from agent_memory.infra.docker import DockerManager

try:
    from arango import ArangoClient
    HAS_ARANGO = True
except ImportError:
    HAS_ARANGO = False

docker_required = pytest.mark.skipif(
    not HAS_ARANGO, reason="python-arango not installed"
)


@pytest.fixture(scope="module")
def arango_container():
    dm = DockerManager()
    try:
        info = dm.ensure_running(
            backend_name="arangodb-test",
            image="arangodb/arangodb:latest",
            port=8530,
            env={"ARANGO_NO_AUTH": "1"},
            health_timeout=60,
            container_port=8529,
        )
        import time
        # Poll until ArangoDB is actually ready
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            try:
                client = ArangoClient(hosts=f"http://localhost:{info.port}")
                client.db("_system").version()
                break
            except Exception:
                time.sleep(2)
        yield info
    finally:
        dm.stop("arangodb-test")


@docker_required
@pytest.mark.docker
class TestArangoSmoke:
    def test_connection(self, arango_container):
        client = ArangoClient(hosts=f"http://localhost:{arango_container.port}")
        sys_db = client.db("_system")
        version = sys_db.version()
        assert version

    def test_create_database(self, arango_container):
        client = ArangoClient(hosts=f"http://localhost:{arango_container.port}")
        sys_db = client.db("_system")
        if sys_db.has_database("memwright_test"):
            sys_db.delete_database("memwright_test")
        sys_db.create_database("memwright_test")
        assert sys_db.has_database("memwright_test")
        sys_db.delete_database("memwright_test")

    def test_create_collection_and_insert(self, arango_container):
        client = ArangoClient(hosts=f"http://localhost:{arango_container.port}")
        sys_db = client.db("_system")
        if not sys_db.has_collection("test_memories"):
            sys_db.create_collection("test_memories")
        col = sys_db.collection("test_memories")
        doc = col.insert({"content": "hello", "tags": ["test"]})
        assert doc["_key"]
        retrieved = col.get(doc["_key"])
        assert retrieved["content"] == "hello"
        sys_db.delete_collection("test_memories")

    def test_graph_operations(self, arango_container):
        client = ArangoClient(hosts=f"http://localhost:{arango_container.port}")
        sys_db = client.db("_system")
        if sys_db.has_graph("test_graph"):
            sys_db.delete_graph("test_graph", drop_collections=True)
        graph = sys_db.create_graph("test_graph")
        entities = graph.create_vertex_collection("test_entities")
        graph.create_edge_definition(
            edge_collection="test_relations",
            from_vertex_collections=["test_entities"],
            to_vertex_collections=["test_entities"],
        )
        entities.insert({"_key": "alice", "name": "Alice", "entity_type": "person"})
        entities.insert({"_key": "bob", "name": "Bob", "entity_type": "person"})
        relations = graph.edge_collection("test_relations")
        relations.insert({
            "_from": "test_entities/alice",
            "_to": "test_entities/bob",
            "relation_type": "KNOWS",
        })
        cursor = sys_db.aql.execute(
            "FOR v IN 1..2 ANY 'test_entities/alice' GRAPH 'test_graph' RETURN v"
        )
        results = list(cursor)
        assert len(results) >= 1
        assert any(r["name"] == "Bob" for r in results)
        sys_db.delete_graph("test_graph", drop_collections=True)
