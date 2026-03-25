# Database Abstraction Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a database abstraction layer so backends (SQLite/ChromaDB/NetworkX, ArangoDB, future Postgres/Neo4j) are swappable via config, with ArangoDB as the first multi-role backend.

**Architecture:** Bottom-up: Docker infrastructure first, then ArangoDB connection verification, abstract interfaces, ArangoDB backend implementation (document/vector/graph), backend registry + config, refactor existing backends to conform, update core.py, integration tests.

**Tech Stack:** Python 3.9+, python-arango, sentence-transformers, Docker CLI (subprocess), pytest

---

## File Structure

| File | Action | Purpose |
|------|--------|---------|
| `agent_memory/infra/__init__.py` | Create | Package init |
| `agent_memory/infra/docker.py` | Create | DockerManager — auto-provision containers |
| `agent_memory/store/base.py` | Create | DocumentStore, VectorStore, GraphStore ABCs |
| `agent_memory/store/registry.py` | Create | Backend registry + role resolver |
| `agent_memory/store/arango_backend.py` | Create | ArangoBackend (all 3 roles) |
| `agent_memory/store/sqlite_store.py` | Modify | Inherit DocumentStore ABC |
| `agent_memory/store/chroma_store.py` | Modify | Inherit VectorStore ABC |
| `agent_memory/graph/networkx_graph.py` | Modify | Inherit GraphStore ABC |
| `agent_memory/utils/config.py` | Modify | Add backends + per-backend config |
| `agent_memory/retrieval/orchestrator.py` | Modify | Type hints: SQLiteStore -> DocumentStore |
| `agent_memory/temporal/manager.py` | Modify | Type hints: SQLiteStore -> DocumentStore |
| `agent_memory/core.py` | Modify | Use registry to resolve backends |
| `pyproject.toml` | Modify | Add `arangodb` optional dependency |
| `tests/test_docker.py` | Create | DockerManager tests |
| `tests/test_arango_backend.py` | Create | ArangoDB backend tests (requires Docker) |
| `tests/test_base_interfaces.py` | Create | ABC conformance tests |
| `tests/test_registry.py` | Create | Registry + resolver tests |
| `tests/test_config_backends.py` | Create | Config loading with backends |

---

### Task 1: Docker Manager — Infrastructure Layer

**Files:**
- Create: `agent_memory/infra/__init__.py`
- Create: `agent_memory/infra/docker.py`
- Test: `tests/test_docker.py`

- [ ] **Step 1: Write the failing test for DockerManager**

```python
# tests/test_docker.py
"""Tests for Docker infrastructure manager."""

import subprocess
import pytest
from unittest.mock import patch, MagicMock
from agent_memory.infra.docker import DockerManager, ContainerInfo


class TestDockerManager:
    def test_container_name_prefix(self):
        dm = DockerManager()
        assert dm.container_name("arangodb") == "memwright-arangodb"

    def test_ensure_running_starts_container(self):
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=False), \
             patch.object(dm, "_start_container") as mock_start, \
             patch.object(dm, "_wait_healthy", return_value=True):
            info = dm.ensure_running(
                backend_name="arangodb",
                image="arangodb/arangodb:latest",
                port=8529,
                env={"ARANGO_NO_AUTH": "1"},
            )
            mock_start.assert_called_once()
            assert info.name == "memwright-arangodb"
            assert info.port == 8529

    def test_ensure_running_reuses_existing(self):
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=True), \
             patch.object(dm, "_start_container") as mock_start:
            info = dm.ensure_running(
                backend_name="arangodb",
                image="arangodb/arangodb:latest",
                port=8529,
                env={},
            )
            mock_start.assert_not_called()
            assert info.name == "memwright-arangodb"

    def test_stop_container(self):
        dm = DockerManager()
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            dm.stop("arangodb")
            mock_run.assert_called()

    def test_health_check_returns_bool(self):
        dm = DockerManager()
        with patch.object(dm, "_is_running", return_value=True):
            assert dm.health_check("arangodb") is True
        with patch.object(dm, "_is_running", return_value=False):
            assert dm.health_check("arangodb") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_docker.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_memory.infra'`

- [ ] **Step 3: Create package init**

```python
# agent_memory/infra/__init__.py
```

- [ ] **Step 4: Write DockerManager implementation**

```python
# agent_memory/infra/docker.py
"""Docker container manager for local backend provisioning."""

from __future__ import annotations

import logging
import subprocess
import time
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("agent_memory")

CONTAINER_PREFIX = "memwright"


@dataclass(frozen=True)
class ContainerInfo:
    name: str
    port: int
    image: str


class DockerManager:
    """Auto-provision Docker containers for local backends.

    Uses docker CLI via subprocess -- no docker-py dependency.
    Containers persist across sessions (reusable).
    """

    def container_name(self, backend_name: str) -> str:
        return f"{CONTAINER_PREFIX}-{backend_name}"

    def ensure_running(
        self,
        backend_name: str,
        image: str,
        port: int,
        env: Dict[str, str],
        health_timeout: int = 30,
    ) -> ContainerInfo:
        """Start container if not running. Returns container info."""
        name = self.container_name(backend_name)

        if not self._is_running(name):
            self._start_container(name, image, port, env)
            self._wait_healthy(name, timeout=health_timeout)

        return ContainerInfo(name=name, port=port, image=image)

    def stop(self, backend_name: str) -> None:
        """Stop and remove a managed container."""
        name = self.container_name(backend_name)
        subprocess.run(
            ["docker", "rm", "-f", name],
            capture_output=True,
            timeout=30,
        )
        logger.info("Stopped container %s", name)

    def health_check(self, backend_name: str) -> bool:
        """Check if a managed container is running."""
        name = self.container_name(backend_name)
        return self._is_running(name)

    def cleanup(self) -> None:
        """Stop all memwright containers."""
        result = subprocess.run(
            ["docker", "ps", "-a", "--filter", f"name={CONTAINER_PREFIX}-",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            for name in result.stdout.strip().split("\n"):
                if name:
                    subprocess.run(
                        ["docker", "rm", "-f", name],
                        capture_output=True, timeout=30,
                    )

    def _is_running(self, name: str) -> bool:
        result = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0 and result.stdout.strip() == "true"

    def _start_container(
        self,
        name: str,
        image: str,
        port: int,
        env: Dict[str, str],
    ) -> None:
        cmd = [
            "docker", "run", "-d",
            "--name", name,
            "-p", f"{port}:{port}",
        ]
        for k, v in env.items():
            cmd.extend(["-e", f"{k}={v}"])
        cmd.append(image)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(
                f"Failed to start {name}: {result.stderr.strip()}"
            )
        logger.info("Started container %s from %s on port %d", name, image, port)

    def _wait_healthy(self, name: str, timeout: int = 30) -> bool:
        """Poll until container is running or timeout."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._is_running(name):
                return True
            time.sleep(1)
        raise TimeoutError(f"Container {name} not healthy after {timeout}s")
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_docker.py -v`
Expected: All 5 tests PASS

- [ ] **Step 6: Commit**

```bash
git add agent_memory/infra/__init__.py agent_memory/infra/docker.py tests/test_docker.py
git commit -m "feat: add DockerManager for local backend provisioning"
```

---

### Task 2: Verify ArangoDB Docker Setup (Integration Smoke Test)

**Files:**
- Test: `tests/test_arango_smoke.py`

This task verifies the DockerManager can actually start ArangoDB and connect. Requires Docker.

- [ ] **Step 1: Write the integration smoke test**

```python
# tests/test_arango_smoke.py
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
    """Start ArangoDB container for the test module."""
    dm = DockerManager()
    try:
        info = dm.ensure_running(
            backend_name="arangodb-test",
            image="arangodb/arangodb:latest",
            port=8530,  # non-default port to avoid conflicts
            env={"ARANGO_NO_AUTH": "1"},
            health_timeout=60,
        )
        # Give ArangoDB a moment to fully initialize
        import time
        time.sleep(3)
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
        assert version  # Non-empty string like "3.12.4"

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
        # Clean up
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
        # Traverse
        cursor = sys_db.aql.execute(
            "FOR v IN 1..2 ANY 'test_entities/alice' GRAPH 'test_graph' RETURN v"
        )
        results = list(cursor)
        assert len(results) >= 1
        assert any(r["name"] == "Bob" for r in results)
        sys_db.delete_graph("test_graph", drop_collections=True)
```

- [ ] **Step 2: Install python-arango and run the smoke test**

Run:
```bash
.venv/bin/pip install python-arango
.venv/bin/pytest tests/test_arango_smoke.py -v -m docker
```
Expected: All 4 tests PASS (Docker pulls arangodb image on first run — may take a minute)

- [ ] **Step 3: Commit**

```bash
git add tests/test_arango_smoke.py
git commit -m "test: add ArangoDB Docker smoke tests"
```

---

### Task 3: Abstract Interfaces (ABCs)

**Files:**
- Create: `agent_memory/store/base.py`
- Test: `tests/test_base_interfaces.py`

- [ ] **Step 1: Write the failing test for ABCs**

```python
# tests/test_base_interfaces.py
"""Tests for abstract base interfaces."""

import pytest
from agent_memory.store.base import DocumentStore, VectorStore, GraphStore


class TestABCsCannotInstantiate:
    def test_document_store_is_abstract(self):
        with pytest.raises(TypeError):
            DocumentStore()

    def test_vector_store_is_abstract(self):
        with pytest.raises(TypeError):
            VectorStore()

    def test_graph_store_is_abstract(self):
        with pytest.raises(TypeError):
            GraphStore()


class ConcreteDocumentStore(DocumentStore):
    """Minimal concrete implementation for testing."""
    ROLES = {"document"}
    def insert(self, memory): return memory
    def get(self, memory_id): return None
    def update(self, memory): return memory
    def delete(self, memory_id): return False
    def list_memories(self, **kwargs): return []
    def tag_search(self, tags, **kwargs): return []
    def execute(self, query, params=None): return []
    def archive_before(self, date): return 0
    def compact(self): return 0
    def stats(self): return {}
    def close(self): pass


class ConcreteVectorStore(VectorStore):
    ROLES = {"vector"}
    def add(self, memory_id, content): pass
    def search(self, query_text, limit=20): return []
    def delete(self, memory_id): return False
    def count(self): return 0
    def close(self): pass


class ConcreteGraphStore(GraphStore):
    ROLES = {"graph"}
    def add_entity(self, name, entity_type="general", attributes=None): pass
    def add_relation(self, from_entity, to_entity, relation_type="related_to", metadata=None): pass
    def get_related(self, entity, depth=2): return []
    def get_subgraph(self, entity, depth=2): return {}
    def get_entities(self, entity_type=None): return []
    def get_edges(self, entity): return []
    def stats(self): return {}
    def save(self): pass
    def close(self): pass


class TestConcreteImplementations:
    def test_document_store_instantiates(self):
        store = ConcreteDocumentStore()
        assert "document" in store.ROLES

    def test_vector_store_instantiates(self):
        store = ConcreteVectorStore()
        assert "vector" in store.ROLES

    def test_graph_store_instantiates(self):
        store = ConcreteGraphStore()
        assert "graph" in store.ROLES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_base_interfaces.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_memory.store.base'`

- [ ] **Step 3: Write the ABCs**

```python
# agent_memory/store/base.py
"""Abstract base interfaces for storage backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from agent_memory.models import Memory


class DocumentStore(ABC):
    """Abstract document storage for Memory objects."""

    ROLES: Set[str] = set()

    @abstractmethod
    def insert(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def get(self, memory_id: str) -> Optional[Memory]: ...

    @abstractmethod
    def update(self, memory: Memory) -> Memory: ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
    def list_memories(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]: ...

    @abstractmethod
    def tag_search(
        self,
        tags: List[str],
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]: ...

    @abstractmethod
    def execute(
        self, query: str, params: Optional[List[Any]] = None
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def archive_before(self, date: str) -> int: ...

    @abstractmethod
    def compact(self) -> int: ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    @abstractmethod
    def close(self) -> None: ...


class VectorStore(ABC):
    """Abstract vector embedding storage and similarity search."""

    ROLES: Set[str] = set()

    @abstractmethod
    def add(self, memory_id: str, content: str) -> None: ...

    @abstractmethod
    def search(
        self, query_text: str, limit: int = 20
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def delete(self, memory_id: str) -> bool: ...

    @abstractmethod
    def count(self) -> int: ...

    @abstractmethod
    def close(self) -> None: ...


class GraphStore(ABC):
    """Abstract entity graph with traversal."""

    ROLES: Set[str] = set()

    @abstractmethod
    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @abstractmethod
    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None: ...

    @abstractmethod
    def get_related(self, entity: str, depth: int = 2) -> List[str]: ...

    @abstractmethod
    def get_subgraph(
        self, entity: str, depth: int = 2
    ) -> Dict[str, Any]: ...

    @abstractmethod
    def get_entities(
        self, entity_type: Optional[str] = None
    ) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def get_edges(self, entity: str) -> List[Dict[str, Any]]: ...

    @abstractmethod
    def stats(self) -> Dict[str, Any]: ...

    @abstractmethod
    def save(self) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_base_interfaces.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_memory/store/base.py tests/test_base_interfaces.py
git commit -m "feat: add DocumentStore, VectorStore, GraphStore abstract interfaces"
```

---

### Task 4: ArangoDB Backend — Document Role

**Files:**
- Create: `agent_memory/store/arango_backend.py`
- Test: `tests/test_arango_backend.py`

- [ ] **Step 1: Write the failing test for document CRUD**

```python
# tests/test_arango_backend.py
"""Tests for ArangoDB backend -- requires Docker.

Run with: .venv/bin/pytest tests/test_arango_backend.py -v -m docker
"""

import pytest

try:
    from arango import ArangoClient
    HAS_ARANGO = True
except ImportError:
    HAS_ARANGO = False

from agent_memory.models import Memory
from agent_memory.infra.docker import DockerManager

docker_required = pytest.mark.skipif(
    not HAS_ARANGO, reason="python-arango not installed"
)

ARANGO_TEST_PORT = 8530


@pytest.fixture(scope="module")
def arango_container():
    dm = DockerManager()
    try:
        info = dm.ensure_running(
            backend_name="arangodb-test",
            image="arangodb/arangodb:latest",
            port=ARANGO_TEST_PORT,
            env={"ARANGO_NO_AUTH": "1"},
            health_timeout=60,
        )
        import time
        time.sleep(3)
        yield info
    finally:
        dm.stop("arangodb-test")


@pytest.fixture
def arango_backend(arango_container):
    from agent_memory.store.arango_backend import ArangoBackend
    backend = ArangoBackend({
        "mode": "cloud",  # already running via fixture
        "url": f"http://localhost:{ARANGO_TEST_PORT}",
        "database": "memwright_test",
    })
    yield backend
    # Clean up test database
    backend.close()
    client = ArangoClient(hosts=f"http://localhost:{ARANGO_TEST_PORT}")
    sys_db = client.db("_system")
    if sys_db.has_database("memwright_test"):
        sys_db.delete_database("memwright_test")


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
        m.content = "updated"
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
        # execute() for ArangoDB takes AQL, not SQL
        results = arango_backend.execute(
            "FOR doc IN memories FILTER doc.content == @content RETURN doc",
            {"content": "exec test"},
        )
        assert len(results) >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_arango_backend.py::TestArangoDocumentStore -v -m docker`
Expected: FAIL with `ModuleNotFoundError: No module named 'agent_memory.store.arango_backend'`

- [ ] **Step 3: Write ArangoBackend — document role**

```python
# agent_memory/store/arango_backend.py
"""ArangoDB backend — single backend for document, vector, and graph roles."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set

from arango import ArangoClient

from agent_memory.models import Memory
from agent_memory.store.base import DocumentStore, VectorStore, GraphStore

logger = logging.getLogger("agent_memory")


def _resolve_env(value: Any) -> Any:
    """Resolve $ENV_VAR references in config values."""
    if isinstance(value, str) and value.startswith("$"):
        return os.environ.get(value[1:], value)
    return value


def _sanitize_rel_type(rel_type: str) -> str:
    """Normalize relation type: uppercase, safe characters only."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", rel_type)
    return sanitized.upper()


class ArangoBackend(DocumentStore, VectorStore, GraphStore):
    """Multi-role ArangoDB backend: document + vector + graph in one DB.

    Config:
        mode: "local" (auto-Docker) or "cloud" (connection string)
        url: ArangoDB endpoint (default: http://localhost:8529)
        database: database name (default: memwright)
        username: (default: root)
        password: (default: empty, resolves $ENV_VAR)
        port: Docker port for local mode (default: 8529)
    """

    ROLES: Set[str] = {"document", "vector", "graph"}

    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config
        mode = config.get("mode", "cloud")
        url = _resolve_env(config.get("url", "http://localhost:8529"))
        db_name = config.get("database", "memwright")
        username = _resolve_env(config.get("username", "root"))
        password = _resolve_env(config.get("password", ""))

        # For local mode, Docker is managed externally by DockerManager
        if mode == "local":
            port = config.get("port", 8529)
            url = f"http://localhost:{port}"

        self._client = ArangoClient(hosts=url)

        # Create database if it doesn't exist
        sys_db = self._client.db(
            "_system", username=username, password=password,
        )
        if not sys_db.has_database(db_name):
            sys_db.create_database(db_name)

        self._db = self._client.db(db_name, username=username, password=password)
        self._init_collections()
        self._init_graph()
        self._embedding_fn = None
        self._vector_index_created = False

    def _init_collections(self) -> None:
        """Create document collections and indexes."""
        if not self._db.has_collection("memories"):
            self._db.create_collection("memories")
        col = self._db.collection("memories")
        # Add persistent indexes
        existing = {idx["fields"][0] for idx in col.indexes() if idx["type"] == "persistent"}
        for field in ["category", "entity", "status", "created_at"]:
            if field not in existing:
                col.add_index({"type": "persistent", "fields": [field]})

    def _init_graph(self) -> None:
        """Create graph structure: entities (vertices) + relations (edges)."""
        if not self._db.has_collection("entities"):
            self._db.create_collection("entities")
        if not self._db.has_collection("relations"):
            self._db.create_collection("relations", edge=True)
        if not self._db.has_graph("memory_graph"):
            self._db.create_graph(
                "memory_graph",
                edge_definitions=[{
                    "edge_collection": "relations",
                    "from_vertex_collections": ["entities"],
                    "to_vertex_collections": ["entities"],
                }],
            )

    # ── DocumentStore ──

    def _memory_to_doc(self, memory: Memory) -> Dict[str, Any]:
        return {
            "_key": memory.id,
            "content": memory.content,
            "tags": memory.tags,
            "category": memory.category,
            "entity": memory.entity,
            "created_at": memory.created_at,
            "event_date": memory.event_date,
            "valid_from": memory.valid_from,
            "valid_until": memory.valid_until,
            "superseded_by": memory.superseded_by,
            "confidence": memory.confidence,
            "status": memory.status,
            "metadata": memory.metadata,
        }

    def _doc_to_memory(self, doc: Dict[str, Any]) -> Memory:
        return Memory(
            id=doc["_key"],
            content=doc["content"],
            tags=doc.get("tags", []),
            category=doc.get("category", "general"),
            entity=doc.get("entity"),
            created_at=doc["created_at"],
            event_date=doc.get("event_date"),
            valid_from=doc["valid_from"],
            valid_until=doc.get("valid_until"),
            superseded_by=doc.get("superseded_by"),
            confidence=doc.get("confidence", 1.0),
            status=doc.get("status", "active"),
            metadata=doc.get("metadata", {}),
        )

    def insert(self, memory: Memory) -> Memory:
        col = self._db.collection("memories")
        col.insert(self._memory_to_doc(memory))
        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        col = self._db.collection("memories")
        doc = col.get(memory_id)
        if doc is None:
            return None
        return self._doc_to_memory(doc)

    def update(self, memory: Memory) -> Memory:
        col = self._db.collection("memories")
        col.replace(self._memory_to_doc(memory))
        return memory

    def delete(self, memory_id: str) -> bool:
        col = self._db.collection("memories")
        doc = col.get(memory_id)
        if doc is None:
            return False
        col.delete(memory_id)
        return True

    def list_memories(
        self,
        status: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 100,
    ) -> List[Memory]:
        filters = []
        bind_vars: Dict[str, Any] = {"@col": "memories", "lim": limit}

        if status:
            filters.append("doc.status == @status")
            bind_vars["status"] = status
        if category:
            filters.append("doc.category == @category")
            bind_vars["category"] = category
        if entity:
            filters.append("doc.entity == @entity")
            bind_vars["entity"] = entity
        if after:
            filters.append("doc.created_at >= @after")
            bind_vars["after"] = after
        if before:
            filters.append("doc.created_at <= @before")
            bind_vars["before"] = before

        where = " AND ".join(filters) if filters else "true"
        aql = f"FOR doc IN @@col FILTER {where} SORT doc.created_at DESC LIMIT @lim RETURN doc"
        cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
        return [self._doc_to_memory(doc) for doc in cursor]

    def tag_search(
        self,
        tags: List[str],
        category: Optional[str] = None,
        limit: int = 20,
    ) -> List[Memory]:
        bind_vars: Dict[str, Any] = {"@col": "memories", "lim": limit, "tags": tags}
        filters = [
            "doc.status == 'active'",
            "doc.valid_until == null",
            "LENGTH(INTERSECTION(doc.tags, @tags)) > 0",
        ]
        if category:
            filters.append("doc.category == @category")
            bind_vars["category"] = category

        where = " AND ".join(filters)
        aql = f"FOR doc IN @@col FILTER {where} SORT doc.created_at DESC LIMIT @lim RETURN doc"
        cursor = self._db.aql.execute(aql, bind_vars=bind_vars)
        return [self._doc_to_memory(doc) for doc in cursor]

    def execute(
        self, query: str, params: Optional[Any] = None
    ) -> List[Dict[str, Any]]:
        """Execute raw AQL query."""
        bind_vars = params if isinstance(params, dict) else {}
        cursor = self._db.aql.execute(query, bind_vars=bind_vars)
        return [doc for doc in cursor]

    def archive_before(self, date: str) -> int:
        aql = """
        FOR doc IN memories
            FILTER doc.created_at < @date AND doc.status == 'active'
            UPDATE doc WITH { status: 'archived' } IN memories
            RETURN 1
        """
        cursor = self._db.aql.execute(aql, bind_vars={"date": date})
        return sum(1 for _ in cursor)

    def compact(self) -> int:
        aql = """
        FOR doc IN memories
            FILTER doc.status == 'archived'
            REMOVE doc IN memories
            RETURN 1
        """
        cursor = self._db.aql.execute(aql, bind_vars={})
        return sum(1 for _ in cursor)

    def stats(self) -> Dict[str, Any]:
        total_cursor = self._db.aql.execute("RETURN LENGTH(memories)")
        total = next(total_cursor)

        by_status: Dict[str, int] = {}
        status_cursor = self._db.aql.execute(
            "FOR doc IN memories COLLECT s = doc.status WITH COUNT INTO c RETURN {status: s, count: c}"
        )
        for row in status_cursor:
            by_status[row["status"]] = row["count"]

        by_category: Dict[str, int] = {}
        cat_cursor = self._db.aql.execute(
            "FOR doc IN memories COLLECT c = doc.category WITH COUNT INTO cnt RETURN {category: c, count: cnt}"
        )
        for row in cat_cursor:
            by_category[row["category"]] = row["count"]

        return {
            "total_memories": total,
            "by_status": by_status,
            "by_category": by_category,
        }

    # ── VectorStore ──
    # (implemented in Task 5)

    def add(self, memory_id: str, content: str) -> None:
        raise NotImplementedError("Vector role not yet implemented")

    def search(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        raise NotImplementedError("Vector role not yet implemented")

    # VectorStore.delete conflicts with DocumentStore.delete signature
    # We handle this by using memory_id parameter name for both

    def count(self) -> int:
        raise NotImplementedError("Vector role not yet implemented")

    # ── GraphStore ──
    # (implemented in Task 6)

    def add_entity(self, name: str, entity_type: str = "general", attributes: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError("Graph role not yet implemented")

    def add_relation(self, from_entity: str, to_entity: str, relation_type: str = "related_to", metadata: Optional[Dict[str, Any]] = None) -> None:
        raise NotImplementedError("Graph role not yet implemented")

    def get_related(self, entity: str, depth: int = 2) -> List[str]:
        raise NotImplementedError("Graph role not yet implemented")

    def get_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        raise NotImplementedError("Graph role not yet implemented")

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        raise NotImplementedError("Graph role not yet implemented")

    def get_edges(self, entity: str) -> List[Dict[str, Any]]:
        raise NotImplementedError("Graph role not yet implemented")

    def save(self) -> None:
        pass  # ArangoDB persists automatically

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: Run document tests to verify they pass**

Run: `.venv/bin/pytest tests/test_arango_backend.py::TestArangoDocumentStore -v -m docker`
Expected: All 9 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_memory/store/arango_backend.py tests/test_arango_backend.py
git commit -m "feat: add ArangoBackend document store role"
```

---

### Task 5: ArangoDB Backend — Vector Role

**Files:**
- Modify: `agent_memory/store/arango_backend.py`
- Test: `tests/test_arango_backend.py` (add vector tests)

- [ ] **Step 1: Add vector tests to test file**

Append to `tests/test_arango_backend.py`:

```python
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
        # Programming-related memories should rank higher than weather
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
        # Deleting via DocumentStore.delete removes the doc (and its vector_data)
        arango_backend.delete("vd1")
        assert arango_backend.get("vd1") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_arango_backend.py::TestArangoVectorStore -v -m docker`
Expected: FAIL with `NotImplementedError: Vector role not yet implemented`

- [ ] **Step 3: Implement vector role in ArangoBackend**

Replace the vector stub methods in `agent_memory/store/arango_backend.py`:

```python
    # ── VectorStore ──

    def _ensure_embedding_fn(self) -> None:
        """Lazy-init sentence-transformers embedding function."""
        if self._embedding_fn is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_fn = SentenceTransformer("all-MiniLM-L6-v2")

    def _ensure_vector_index(self) -> None:
        """Create vector index if enough documents have embeddings."""
        if self._vector_index_created:
            return
        col = self._db.collection("memories")
        # Check if any docs have vector_data
        cursor = self._db.aql.execute(
            "FOR doc IN memories FILTER doc.vector_data != null LIMIT 1 RETURN 1"
        )
        if sum(1 for _ in cursor) == 0:
            return
        # Check if vector index already exists
        for idx in col.indexes():
            if idx.get("type") == "vector":
                self._vector_index_created = True
                return
        # Create index — requires data to exist first
        try:
            col.add_index({
                "type": "vector",
                "fields": ["vector_data"],
                "params": {
                    "metric": "cosine",
                    "dimension": 384,
                    "nLists": 10,
                },
            })
            self._vector_index_created = True
        except Exception as e:
            logger.warning("Vector index creation failed: %s", e)

    def add(self, memory_id: str, content: str) -> None:
        """Generate embedding and store as vector_data on the memory doc."""
        self._ensure_embedding_fn()
        embedding = self._embedding_fn.encode(content).tolist()
        col = self._db.collection("memories")
        doc = col.get(memory_id)
        if doc is not None:
            col.update({"_key": memory_id, "vector_data": embedding})
        self._ensure_vector_index()

    def search(self, query_text: str, limit: int = 20) -> List[Dict[str, Any]]:
        """Vector similarity search using APPROX_NEAR_COSINE if index exists,
        fallback to exact cosine similarity."""
        self._ensure_embedding_fn()
        query_vec = self._embedding_fn.encode(query_text).tolist()

        # Try approximate search first (requires vector index)
        if self._vector_index_created:
            try:
                aql = """
                FOR doc IN memories
                    LET score = APPROX_NEAR_COSINE(doc.vector_data, @query_vec)
                    FILTER score != null
                    SORT score DESC
                    LIMIT @lim
                    RETURN {memory_id: doc._key, content: doc.content, distance: 1.0 - score}
                """
                cursor = self._db.aql.execute(
                    aql, bind_vars={"query_vec": query_vec, "lim": limit}
                )
                return list(cursor)
            except Exception:
                pass  # Fallback to exact

        # Exact cosine similarity fallback
        aql = """
        FOR doc IN memories
            FILTER doc.vector_data != null
            LET score = COSINE_SIMILARITY(doc.vector_data, @query_vec)
            FILTER score != null
            SORT score DESC
            LIMIT @lim
            RETURN {memory_id: doc._key, content: doc.content, distance: 1.0 - score}
        """
        cursor = self._db.aql.execute(
            aql, bind_vars={"query_vec": query_vec, "lim": limit}
        )
        return list(cursor)

    def count(self) -> int:
        """Count documents that have vector embeddings."""
        cursor = self._db.aql.execute(
            "RETURN LENGTH(FOR doc IN memories FILTER doc.vector_data != null RETURN 1)"
        )
        return next(cursor)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_arango_backend.py::TestArangoVectorStore -v -m docker`
Expected: All 3 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_memory/store/arango_backend.py tests/test_arango_backend.py
git commit -m "feat: add ArangoBackend vector store role"
```

---

### Task 6: ArangoDB Backend — Graph Role

**Files:**
- Modify: `agent_memory/store/arango_backend.py`
- Test: `tests/test_arango_backend.py` (add graph tests)

- [ ] **Step 1: Add graph tests to test file**

Append to `tests/test_arango_backend.py`:

```python
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
        assert bob["type"] == "person"  # upgraded from general

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
        s = arango_backend.stats()  # This calls DocumentStore.stats
        # Graph stats are separate
        graph_stats = arango_backend.graph_stats()
        assert graph_stats["nodes"] >= 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_arango_backend.py::TestArangoGraphStore -v -m docker`
Expected: FAIL with `NotImplementedError: Graph role not yet implemented`

- [ ] **Step 3: Implement graph role in ArangoBackend**

Replace the graph stub methods in `agent_memory/store/arango_backend.py`:

```python
    # ── GraphStore ──

    def add_entity(
        self,
        name: str,
        entity_type: str = "general",
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        key = name.lower()
        attrs = dict(attributes) if attributes else {}
        graph = self._db.graph("memory_graph")
        entities = graph.vertex_collection("entities")

        if entities.has(key):
            existing = entities.get(key)
            # Merge attributes
            update_doc: Dict[str, Any] = {"_key": key, "display_name": name}
            for k, v in attrs.items():
                update_doc[k] = v
            # Upgrade entity_type from general
            if existing.get("entity_type", "general") == "general" and entity_type != "general":
                update_doc["entity_type"] = entity_type
            entities.update(update_doc)
        else:
            doc = {"_key": key, "display_name": name, "entity_type": entity_type}
            doc.update(attrs)
            entities.insert(doc)

    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = "related_to",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        from_key = from_entity.lower()
        to_key = to_entity.lower()

        graph = self._db.graph("memory_graph")
        entities = graph.vertex_collection("entities")
        relations = graph.edge_collection("relations")

        # Auto-create missing entities
        if not entities.has(from_key):
            entities.insert({"_key": from_key, "display_name": from_entity, "entity_type": "general"})
        if not entities.has(to_key):
            entities.insert({"_key": to_key, "display_name": to_entity, "entity_type": "general"})

        sanitized = _sanitize_rel_type(relation_type)
        edge_doc = {
            "_from": f"entities/{from_key}",
            "_to": f"entities/{to_key}",
            "relation_type": sanitized,
        }
        if metadata:
            edge_doc.update(metadata)
        relations.insert(edge_doc)

    def get_related(self, entity: str, depth: int = 2) -> List[str]:
        start = entity.lower()
        entities = self._db.collection("entities")
        if not entities.has(start):
            return []

        aql = """
        FOR v IN 1..@depth ANY @start GRAPH 'memory_graph'
            OPTIONS { bfs: true, uniqueVertices: 'global' }
            RETURN v.display_name
        """
        cursor = self._db.aql.execute(
            aql, bind_vars={"depth": depth, "start": f"entities/{start}"}
        )
        return [name for name in cursor if name]

    def get_subgraph(self, entity: str, depth: int = 2) -> Dict[str, Any]:
        start = entity.lower()
        entities = self._db.collection("entities")
        if not entities.has(start):
            return {"entity": entity, "nodes": [], "edges": []}

        # Get vertices
        v_aql = """
        LET start_v = DOCUMENT(@start)
        LET traversed = (
            FOR v IN 1..@depth ANY @start GRAPH 'memory_graph'
                OPTIONS { bfs: true, uniqueVertices: 'global' }
                RETURN v
        )
        RETURN APPEND([start_v], traversed)
        """
        v_cursor = self._db.aql.execute(
            v_aql, bind_vars={"depth": depth, "start": f"entities/{start}"}
        )
        all_vertices = next(v_cursor, [])

        nodes = []
        node_keys = set()
        for v in all_vertices:
            if v and v.get("_key") not in node_keys:
                node_keys.add(v["_key"])
                nodes.append({
                    "name": v.get("display_name", v["_key"]),
                    "type": v.get("entity_type", "general"),
                    "key": v["_key"],
                })

        # Get edges between these nodes
        edges = []
        if node_keys:
            e_aql = """
            FOR e IN relations
                FILTER PARSE_IDENTIFIER(e._from).key IN @keys
                   AND PARSE_IDENTIFIER(e._to).key IN @keys
                RETURN {
                    source: PARSE_IDENTIFIER(e._from).key,
                    target: PARSE_IDENTIFIER(e._to).key,
                    type: e.relation_type
                }
            """
            e_cursor = self._db.aql.execute(e_aql, bind_vars={"keys": list(node_keys)})
            edges = list(e_cursor)

        return {"entity": entity, "nodes": nodes, "edges": edges}

    def get_entities(self, entity_type: Optional[str] = None) -> List[Dict[str, Any]]:
        if entity_type:
            aql = "FOR doc IN entities FILTER doc.entity_type == @et RETURN doc"
            cursor = self._db.aql.execute(aql, bind_vars={"et": entity_type})
        else:
            cursor = self._db.aql.execute("FOR doc IN entities RETURN doc")

        result = []
        for doc in cursor:
            attrs = {
                k: v for k, v in doc.items()
                if k not in ("_key", "_id", "_rev", "display_name", "entity_type")
            }
            result.append({
                "name": doc.get("display_name", doc["_key"]),
                "type": doc.get("entity_type", "general"),
                "key": doc["_key"],
                "attributes": attrs,
            })
        return result

    def get_edges(self, entity: str) -> List[Dict[str, Any]]:
        key = entity.lower()
        entities_col = self._db.collection("entities")
        if not entities_col.has(key):
            return []

        aql = """
        LET outgoing = (
            FOR v, e IN 1..1 OUTBOUND @start GRAPH 'memory_graph'
                RETURN {
                    subject: DOCUMENT(e._from).display_name,
                    predicate: e.relation_type,
                    object: DOCUMENT(e._to).display_name,
                    event_date: e.event_date || ""
                }
        )
        LET incoming = (
            FOR v, e IN 1..1 INBOUND @start GRAPH 'memory_graph'
                RETURN {
                    subject: DOCUMENT(e._from).display_name,
                    predicate: e.relation_type,
                    object: DOCUMENT(e._to).display_name,
                    event_date: e.event_date || ""
                }
        )
        RETURN APPEND(outgoing, incoming)
        """
        cursor = self._db.aql.execute(aql, bind_vars={"start": f"entities/{key}"})
        all_edges = next(cursor, [])
        # Deduplicate
        seen = set()
        result = []
        for edge in all_edges:
            triple = (edge["subject"], edge["predicate"], edge["object"])
            if triple not in seen:
                seen.add(triple)
                result.append(edge)
        return result

    def graph_stats(self) -> Dict[str, Any]:
        """Graph-specific stats (separate from DocumentStore.stats)."""
        nodes_cursor = self._db.aql.execute("RETURN LENGTH(entities)")
        edges_cursor = self._db.aql.execute("RETURN LENGTH(relations)")
        nodes = next(nodes_cursor)
        edges = next(edges_cursor)

        types: Dict[str, int] = {}
        type_cursor = self._db.aql.execute(
            "FOR doc IN entities COLLECT t = doc.entity_type WITH COUNT INTO c RETURN {type: t, count: c}"
        )
        for row in type_cursor:
            types[row["type"]] = row["count"]

        return {"nodes": nodes, "edges": edges, "types": types}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_arango_backend.py::TestArangoGraphStore -v -m docker`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_memory/store/arango_backend.py tests/test_arango_backend.py
git commit -m "feat: add ArangoBackend graph store role"
```

---

### Task 7: Backend Registry and Resolver

**Files:**
- Create: `agent_memory/store/registry.py`
- Test: `tests/test_registry.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_registry.py
"""Tests for backend registry and resolver."""

import pytest
from agent_memory.store.registry import (
    BACKEND_REGISTRY,
    BackendConflictError,
    resolve_backends,
)


class TestBackendRegistry:
    def test_default_backends_registered(self):
        assert "sqlite" in BACKEND_REGISTRY
        assert "chroma" in BACKEND_REGISTRY
        assert "networkx" in BACKEND_REGISTRY
        assert "arangodb" in BACKEND_REGISTRY

    def test_roles_declared(self):
        assert BACKEND_REGISTRY["sqlite"]["roles"] == {"document"}
        assert BACKEND_REGISTRY["chroma"]["roles"] == {"vector"}
        assert BACKEND_REGISTRY["networkx"]["roles"] == {"graph"}
        assert BACKEND_REGISTRY["arangodb"]["roles"] == {"document", "vector", "graph"}


class TestResolveBackends:
    def test_default_resolution(self):
        result = resolve_backends(["sqlite", "chroma", "networkx"])
        assert "document" in result
        assert "vector" in result
        assert "graph" in result
        assert result["document"]["name"] == "sqlite"
        assert result["vector"]["name"] == "chroma"
        assert result["graph"]["name"] == "networkx"

    def test_arangodb_fills_all_roles(self):
        result = resolve_backends(["arangodb"])
        assert result["document"]["name"] == "arangodb"
        assert result["vector"]["name"] == "arangodb"
        assert result["graph"]["name"] == "arangodb"

    def test_conflict_raises_error(self):
        with pytest.raises(BackendConflictError):
            resolve_backends(["sqlite", "arangodb"])  # both claim "document"

    def test_unknown_backend_raises(self):
        with pytest.raises(KeyError):
            resolve_backends(["nonexistent"])

    def test_partial_roles_ok(self):
        """When only document is configured, vector and graph are None."""
        result = resolve_backends(["sqlite"])
        assert result["document"]["name"] == "sqlite"
        assert result.get("vector") is None
        assert result.get("graph") is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the registry**

```python
# agent_memory/store/registry.py
"""Backend registry and role resolver."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set


class BackendConflictError(Exception):
    """Raised when two backends claim the same role."""
    pass


BACKEND_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sqlite": {
        "module": "agent_memory.store.sqlite_store",
        "class": "SQLiteStore",
        "roles": {"document"},
    },
    "chroma": {
        "module": "agent_memory.store.chroma_store",
        "class": "ChromaStore",
        "roles": {"vector"},
    },
    "networkx": {
        "module": "agent_memory.graph.networkx_graph",
        "class": "NetworkXGraph",
        "roles": {"graph"},
    },
    "arangodb": {
        "module": "agent_memory.store.arango_backend",
        "class": "ArangoBackend",
        "roles": {"document", "vector", "graph"},
    },
}


def resolve_backends(
    backend_names: List[str],
) -> Dict[str, Optional[Dict[str, Any]]]:
    """Resolve backend names to role assignments.

    Returns dict mapping role -> {name, module, class} or None if unfilled.
    Raises BackendConflictError if two backends claim the same role.
    """
    assignments: Dict[str, Optional[Dict[str, Any]]] = {
        "document": None,
        "vector": None,
        "graph": None,
    }

    for name in backend_names:
        entry = BACKEND_REGISTRY[name]  # KeyError if unknown
        for role in entry["roles"]:
            if assignments.get(role) is not None:
                existing = assignments[role]["name"]
                raise BackendConflictError(
                    f"Role '{role}' already claimed by '{existing}', "
                    f"cannot assign to '{name}'"
                )
            assignments[role] = {
                "name": name,
                "module": entry["module"],
                "class": entry["class"],
            }

    return assignments
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_registry.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add agent_memory/store/registry.py tests/test_registry.py
git commit -m "feat: add backend registry with role resolver"
```

---

### Task 8: Extend Config for Backends

**Files:**
- Modify: `agent_memory/utils/config.py`
- Test: `tests/test_config_backends.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config_backends.py
"""Tests for config with backend settings."""

import json
import os
import tempfile
from pathlib import Path

import pytest
from agent_memory.utils.config import MemoryConfig, load_config, save_config


class TestBackendConfig:
    def test_default_backends(self):
        cfg = MemoryConfig()
        assert cfg.backends == ["sqlite", "chroma", "networkx"]

    def test_custom_backends(self):
        cfg = MemoryConfig.from_dict({
            "backends": ["arangodb"],
            "arangodb": {"mode": "local", "port": 8529},
        })
        assert cfg.backends == ["arangodb"]
        assert cfg.backend_configs["arangodb"]["mode"] == "local"

    def test_env_var_resolution(self):
        os.environ["TEST_ARANGO_PW"] = "secret123"
        cfg = MemoryConfig.from_dict({
            "backends": ["arangodb"],
            "arangodb": {"mode": "cloud", "password": "$TEST_ARANGO_PW"},
        })
        assert cfg.backend_configs["arangodb"]["password"] == "$TEST_ARANGO_PW"
        # Env resolution happens at backend init, not config load
        del os.environ["TEST_ARANGO_PW"]

    def test_roundtrip_save_load(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = MemoryConfig(
                backends=["arangodb"],
                backend_configs={"arangodb": {"mode": "local"}},
            )
            save_config(Path(d), cfg)
            loaded = load_config(Path(d))
            assert loaded.backends == ["arangodb"]
            assert loaded.backend_configs["arangodb"]["mode"] == "local"

    def test_default_config_unchanged(self):
        """No backends key = default embedded setup."""
        cfg = MemoryConfig.from_dict({"default_token_budget": 5000})
        assert cfg.backends == ["sqlite", "chroma", "networkx"]
        assert cfg.default_token_budget == 5000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_config_backends.py -v`
Expected: FAIL (MemoryConfig has no `backends` field)

- [ ] **Step 3: Update MemoryConfig**

Replace the full `agent_memory/utils/config.py`:

```python
"""Configuration loading and defaults."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_BACKENDS = ["sqlite", "chroma", "networkx"]


@dataclass
class MemoryConfig:
    default_token_budget: int = 2000
    min_results: int = 3
    backends: List[str] = field(default_factory=lambda: list(DEFAULT_BACKENDS))
    backend_configs: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> MemoryConfig:
        backends = data.get("backends", list(DEFAULT_BACKENDS))
        # Extract per-backend configs (keys that match backend names)
        backend_configs: Dict[str, Dict[str, Any]] = {}
        for key, val in data.items():
            if isinstance(val, dict) and key not in ("backend_configs",):
                backend_configs[key] = val
        # Also merge explicit backend_configs if present
        if "backend_configs" in data and isinstance(data["backend_configs"], dict):
            backend_configs.update(data["backend_configs"])

        return cls(
            default_token_budget=data.get("default_token_budget", 2000),
            min_results=data.get("min_results", 3),
            backends=backends,
            backend_configs=backend_configs,
        )

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "default_token_budget": self.default_token_budget,
            "min_results": self.min_results,
        }
        if self.backends != DEFAULT_BACKENDS:
            result["backends"] = self.backends
        for name, cfg in self.backend_configs.items():
            result[name] = cfg
        return result


def load_config(path: Path) -> MemoryConfig:
    """Load config from a JSON file, falling back to defaults."""
    config_file = path / "config.json"
    if config_file.exists():
        with open(config_file) as f:
            data = json.load(f)
        return MemoryConfig.from_dict(data)
    return MemoryConfig()


def save_config(path: Path, config: MemoryConfig) -> None:
    """Save config to a JSON file."""
    config_file = path / "config.json"
    with open(config_file, "w") as f:
        json.dump(config.to_dict(), f, indent=2)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_config_backends.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Verify existing tests still pass**

Run: `.venv/bin/pytest tests/ -v --ignore=tests/test_arango_smoke.py --ignore=tests/test_arango_backend.py -m "not docker"`
Expected: All existing tests PASS

- [ ] **Step 6: Commit**

```bash
git add agent_memory/utils/config.py tests/test_config_backends.py
git commit -m "feat: extend MemoryConfig with backends and per-backend configs"
```

---

### Task 9: Refactor Existing Backends to Conform to Interfaces

**Files:**
- Modify: `agent_memory/store/sqlite_store.py`
- Modify: `agent_memory/store/chroma_store.py`
- Modify: `agent_memory/graph/networkx_graph.py`

- [ ] **Step 1: Add interface conformance test**

Append to `tests/test_base_interfaces.py`:

```python
class TestExistingBackendConformance:
    def test_sqlite_is_document_store(self):
        from agent_memory.store.sqlite_store import SQLiteStore
        assert issubclass(SQLiteStore, DocumentStore)
        assert "document" in SQLiteStore.ROLES

    def test_chroma_is_vector_store(self):
        from agent_memory.store.chroma_store import ChromaStore
        assert issubclass(ChromaStore, VectorStore)
        assert "vector" in ChromaStore.ROLES

    def test_networkx_is_graph_store(self):
        from agent_memory.graph.networkx_graph import NetworkXGraph
        assert issubclass(NetworkXGraph, GraphStore)
        assert "graph" in NetworkXGraph.ROLES
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest tests/test_base_interfaces.py::TestExistingBackendConformance -v`
Expected: FAIL (SQLiteStore doesn't inherit DocumentStore yet)

- [ ] **Step 3: Update SQLiteStore**

In `agent_memory/store/sqlite_store.py`, add import and inheritance:

Change line 1-15 to:
```python
"""SQLite storage implementation."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from agent_memory.models import Memory
from agent_memory.store.base import DocumentStore

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class SQLiteStore(DocumentStore):
    """Low-level SQLite storage for memories."""

    ROLES: Set[str] = {"document"}
```

Rest of the file stays exactly the same.

- [ ] **Step 4: Update ChromaStore**

In `agent_memory/store/chroma_store.py`, add import and inheritance:

Change line 1-14 to:
```python
"""ChromaDB vector store — zero-config local embeddings, no API key required."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import chromadb

from agent_memory.store.base import VectorStore

logger = logging.getLogger("agent_memory")


class ChromaStore(VectorStore):
    """Persistent vector store using ChromaDB with local sentence-transformer embeddings."""

    ROLES: Set[str] = {"vector"}
```

Rest of the file stays exactly the same.

- [ ] **Step 5: Update NetworkXGraph**

In `agent_memory/graph/networkx_graph.py`, add import and inheritance:

Change line 1-20 to:
```python
"""NetworkX entity graph — zero-config in-process graph with JSON persistence."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import networkx as nx

from agent_memory.store.base import GraphStore


def _sanitize_rel_type(rel_type: str) -> str:
    """Normalize relation type: uppercase, safe characters only."""
    sanitized = re.sub(r"[^A-Za-z0-9_]", "_", rel_type)
    return sanitized.upper()


class NetworkXGraph(GraphStore):
    """In-process entity graph using NetworkX MultiDiGraph with JSON persistence."""

    ROLES: Set[str] = {"graph"}
```

Rest of the file stays exactly the same.

- [ ] **Step 6: Run conformance tests**

Run: `.venv/bin/pytest tests/test_base_interfaces.py -v`
Expected: All 9 tests PASS

- [ ] **Step 7: Run ALL existing tests to verify no breakage**

Run: `.venv/bin/pytest tests/ -v --ignore=tests/test_arango_smoke.py --ignore=tests/test_arango_backend.py -m "not docker"`
Expected: All existing tests PASS

- [ ] **Step 8: Commit**

```bash
git add agent_memory/store/sqlite_store.py agent_memory/store/chroma_store.py agent_memory/graph/networkx_graph.py tests/test_base_interfaces.py
git commit -m "refactor: existing backends now implement abstract interfaces"
```

---

### Task 10: Update Type Hints in Orchestrator and Temporal Manager

**Files:**
- Modify: `agent_memory/retrieval/orchestrator.py`
- Modify: `agent_memory/temporal/manager.py`

- [ ] **Step 1: Update orchestrator type hints**

In `agent_memory/retrieval/orchestrator.py`, change the import and type hint:

Replace:
```python
from agent_memory.store.sqlite_store import SQLiteStore
```
With:
```python
from agent_memory.store.base import DocumentStore
```

Replace:
```python
    def __init__(
        self,
        store: SQLiteStore,
```
With:
```python
    def __init__(
        self,
        store: DocumentStore,
```

- [ ] **Step 2: Update temporal manager type hints**

In `agent_memory/temporal/manager.py`, change the import and type hint:

Replace:
```python
from agent_memory.store.sqlite_store import SQLiteStore
```
With:
```python
from agent_memory.store.base import DocumentStore
```

Replace:
```python
class TemporalManager:
    """Handles temporal queries, contradiction detection, and supersession."""

    def __init__(self, store: SQLiteStore):
        self.store = store
```
With:
```python
class TemporalManager:
    """Handles temporal queries, contradiction detection, and supersession."""

    def __init__(self, store: DocumentStore):
        self.store = store
```

- [ ] **Step 3: Run all tests**

Run: `.venv/bin/pytest tests/ -v --ignore=tests/test_arango_smoke.py --ignore=tests/test_arango_backend.py -m "not docker"`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add agent_memory/retrieval/orchestrator.py agent_memory/temporal/manager.py
git commit -m "refactor: use DocumentStore interface in orchestrator and temporal manager"
```

---

### Task 11: Update core.py to Use Registry

**Files:**
- Modify: `agent_memory/core.py`

- [ ] **Step 1: Run existing core tests as baseline**

Run: `.venv/bin/pytest tests/test_core.py -v`
Expected: All PASS (baseline)

- [ ] **Step 2: Update core.py to resolve backends from registry**

Replace `agent_memory/core.py` with updated version. Key changes:
- Import registry instead of concrete classes
- Resolve backends from config
- Instantiate via dynamic import
- Default behavior (no config) stays identical

```python
"""AgentMemory main class -- the public API."""

from __future__ import annotations

import importlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_memory.models import Memory, RetrievalResult
from agent_memory.retrieval.orchestrator import RetrievalOrchestrator
from agent_memory.store.base import DocumentStore, VectorStore, GraphStore
from agent_memory.store.registry import resolve_backends
from agent_memory.temporal.manager import TemporalManager
from agent_memory.utils.config import MemoryConfig, load_config, save_config

logger = logging.getLogger("agent_memory")


def _instantiate_backend(
    assignment: Dict[str, Any],
    path: Path,
    backend_configs: Dict[str, Dict[str, Any]],
) -> Any:
    """Dynamically import and instantiate a backend class."""
    mod = importlib.import_module(assignment["module"])
    cls = getattr(mod, assignment["class"])
    name = assignment["name"]
    config = backend_configs.get(name, {})

    # Each backend type has different constructor signatures
    if name == "sqlite":
        db_path = path / "memory.db"
        return cls(db_path)
    elif name == "chroma":
        return cls(path)
    elif name == "networkx":
        return cls(path)
    elif name in ("arangodb",):
        # Multi-role backends get their config dict
        return cls(config)
    else:
        # Future backends: try config dict, fallback to path
        try:
            return cls(config)
        except TypeError:
            return cls(path)


class AgentMemory:
    """Embedded memory for AI agents.

    Usage:
        mem = AgentMemory("./my-agent")
        mem.add("User prefers Python", tags=["preference"], category="preference")
        results = mem.recall("what language?")
    """

    def __init__(
        self,
        path: str | Path,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

        # Load config
        if config:
            self.config = MemoryConfig.from_dict(config)
        else:
            self.config = load_config(self.path)
        save_config(self.path, self.config)

        # Handle Docker for local-mode backends
        self._docker = None
        for backend_name in self.config.backends:
            bcfg = self.config.backend_configs.get(backend_name, {})
            if bcfg.get("mode") == "local":
                self._ensure_docker(backend_name, bcfg)

        # Resolve backends from registry
        assignments = resolve_backends(self.config.backends)

        # Track instantiated backends to reuse same instance for multi-role
        instances: Dict[str, Any] = {}

        # Initialize document store (required)
        doc_assignment = assignments.get("document")
        if doc_assignment:
            name = doc_assignment["name"]
            if name not in instances:
                instances[name] = _instantiate_backend(
                    doc_assignment, self.path, self.config.backend_configs,
                )
            self._store: DocumentStore = instances[name]
        else:
            # Fallback to SQLite (should not happen with defaults)
            from agent_memory.store.sqlite_store import SQLiteStore
            self._store = SQLiteStore(self.path / "memory.db")

        # Initialize vector store (optional, graceful degradation)
        self._vector_store: Optional[VectorStore] = None
        vec_assignment = assignments.get("vector")
        if vec_assignment:
            try:
                name = vec_assignment["name"]
                if name not in instances:
                    instances[name] = _instantiate_backend(
                        vec_assignment, self.path, self.config.backend_configs,
                    )
                self._vector_store = instances[name]
            except Exception as e:
                logger.warning("Vector store init failed: %s", e)

        # Initialize graph store (optional, graceful degradation)
        self._graph: Optional[GraphStore] = None
        graph_assignment = assignments.get("graph")
        if graph_assignment:
            try:
                name = graph_assignment["name"]
                if name not in instances:
                    instances[name] = _instantiate_backend(
                        graph_assignment, self.path, self.config.backend_configs,
                    )
                self._graph = instances[name]
            except Exception as e:
                logger.warning("Graph store init failed: %s", e)

        # Initialize managers
        self._temporal = TemporalManager(self._store)
        self._retrieval = RetrievalOrchestrator(
            self._store,
            min_results=self.config.min_results,
            vector_store=self._vector_store,
            graph=self._graph,
        )

    def _ensure_docker(self, backend_name: str, bcfg: Dict[str, Any]) -> None:
        """Start Docker container for local-mode backend."""
        from agent_memory.infra.docker import DockerManager

        if self._docker is None:
            self._docker = DockerManager()

        # Backend-specific Docker config
        docker_images = {
            "arangodb": ("arangodb/arangodb:latest", 8529, {"ARANGO_NO_AUTH": "1"}),
            "postgres": ("postgres:16", 5432, {"POSTGRES_PASSWORD": "memwright"}),
        }

        if backend_name in docker_images:
            image, default_port, env = docker_images[backend_name]
            port = bcfg.get("port", default_port)
            self._docker.ensure_running(backend_name, image, port, env)

    # Everything below this line is UNCHANGED from the original core.py

    def close(self) -> None:
        """Close all database connections."""
        if self._vector_store:
            try:
                self._vector_store.close()
            except Exception:
                pass
        if self._graph:
            try:
                self._graph.save()
            except Exception:
                pass
            if hasattr(self._graph, "close"):
                try:
                    self._graph.close()
                except Exception:
                    pass
        self._store.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # -- Write --

    def add(
        self,
        content: str,
        tags: Optional[List[str]] = None,
        category: str = "general",
        entity: Optional[str] = None,
        event_date: Optional[str] = None,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Memory:
        """Store a new memory, handling contradictions automatically."""
        memory = Memory(
            content=content,
            tags=tags or [],
            category=category,
            entity=entity,
            event_date=event_date,
            confidence=confidence,
            metadata=metadata or {},
        )

        # Check for contradictions before insert
        contradictions = self._temporal.check_contradictions(memory)

        # Insert new memory first (so FK reference is valid)
        self._store.insert(memory)

        # Then supersede old contradicting memories
        for old in contradictions:
            self._temporal.supersede(old, memory.id)

        # Store in vector DB
        if self._vector_store:
            try:
                self._vector_store.add(memory.id, content)
            except Exception:
                pass  # Non-fatal

        # Update entity graph
        if self._graph:
            try:
                from agent_memory.graph.extractor import extract_entities_and_relations
                nodes, edges = extract_entities_and_relations(
                    content, tags or [], entity, category,
                )
                for node in nodes:
                    self._graph.add_entity(
                        node["name"],
                        entity_type=node.get("type", "general"),
                        attributes=node.get("attributes"),
                    )
                for edge in edges:
                    self._graph.add_relation(
                        edge["from"],
                        edge["to"],
                        relation_type=edge.get("type", "related_to"),
                        metadata=edge.get("metadata"),
                    )
            except Exception:
                pass  # Non-fatal

        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        return self._store.get(memory_id)

    # -- Read --

    def recall(
        self, query: str, budget: Optional[int] = None
    ) -> List[RetrievalResult]:
        """Retrieve relevant memories for a query using 3-layer cascade."""
        token_budget = budget or self.config.default_token_budget
        return self._retrieval.recall(query, token_budget)

    def recall_as_context(
        self, query: str, budget: Optional[int] = None
    ) -> str:
        """Recall and format as a context string for prompt injection."""
        token_budget = budget or self.config.default_token_budget
        return self._retrieval.recall_as_context(query, token_budget)

    def search(
        self,
        query: Optional[str] = None,
        category: Optional[str] = None,
        entity: Optional[str] = None,
        status: str = "active",
        after: Optional[str] = None,
        before: Optional[str] = None,
        limit: int = 10,
    ) -> List[Memory]:
        """Search memories with filters."""
        # If there's a text query and vector store, use semantic search
        if query and self._vector_store:
            try:
                vec_results = self._vector_store.search(query, limit=limit * 2)
                if vec_results:
                    # Get full memory objects, apply filters
                    memories = []
                    for vr in vec_results:
                        mem = self._store.get(vr["memory_id"])
                        if not mem or mem.status != status:
                            continue
                        if category and mem.category != category:
                            continue
                        if entity and mem.entity != entity:
                            continue
                        if after and mem.created_at < after:
                            continue
                        if before and mem.created_at > before:
                            continue
                        memories.append(mem)
                        if len(memories) >= limit:
                            break
                    return memories
            except Exception:
                pass  # Fall through to store search

        return self._store.list_memories(
            status=status,
            category=category,
            entity=entity,
            after=after,
            before=before,
            limit=limit,
        )

    # -- Timeline --

    def timeline(self, entity: str) -> List[Memory]:
        """Get all memories about an entity in chronological order."""
        return self._temporal.timeline(entity)

    def current_facts(
        self,
        category: Optional[str] = None,
        entity: Optional[str] = None,
    ) -> List[Memory]:
        """Get only active, non-superseded memories."""
        return self._temporal.current_facts(category=category, entity=entity)

    # -- Extraction --

    def extract(
        self,
        messages: List[Dict[str, Any]],
        model: str = "claude-haiku",
        use_llm: bool = False,
    ) -> List[Memory]:
        """Extract and store memories from conversation messages."""
        from agent_memory.extraction.extractor import extract_memories

        extracted = extract_memories(messages, use_llm=use_llm, model=model)
        stored = []
        for mem in extracted:
            stored_mem = self.add(
                content=mem.content,
                tags=mem.tags,
                category=mem.category,
                entity=mem.entity,
            )
            stored.append(stored_mem)
        return stored

    # -- Batch Operations --

    def batch_embed(self, batch_size: int = 100) -> int:
        """Batch-index all active memories into vector store.

        Returns count of memories processed.
        """
        if not self._vector_store:
            return 0
        memories = self._store.list_memories(status="active", limit=1_000_000)
        count = 0
        for mem in memories:
            try:
                self._vector_store.add(mem.id, mem.content)
                count += 1
            except Exception:
                pass
        return count

    # -- Maintenance --

    def forget(self, memory_id: str) -> bool:
        """Archive a specific memory."""
        memory = self._store.get(memory_id)
        if memory:
            memory.status = "archived"
            self._store.update(memory)
            return True
        return False

    def forget_before(self, date: str) -> int:
        """Archive memories created before a date."""
        return self._store.archive_before(date)

    def compact(self) -> int:
        """Permanently remove archived memories."""
        return self._store.compact()

    def stats(self) -> Dict[str, Any]:
        """Get store statistics."""
        return self._store.stats()

    def health(self) -> Dict[str, Any]:
        """Check health of all components. Returns structured status report."""
        import time

        report: Dict[str, Any] = {
            "healthy": True,
            "checks": [],
        }

        def _check(name: str, status: str, **details):
            entry = {"name": name, "status": status, **details}
            report["checks"].append(entry)
            if status == "error":
                report["healthy"] = False

        # -- Document Store --
        try:
            t0 = time.monotonic()
            s = self._store.stats()
            latency = round((time.monotonic() - t0) * 1000, 1)
            _check("Document Store", "ok",
                   backend=type(self._store).__name__,
                   memory_count=s.get("total_memories", 0),
                   latency_ms=latency)
        except Exception as e:
            _check("Document Store", "error", error=str(e))

        # -- Vector Store --
        if self._vector_store:
            try:
                vector_count = self._vector_store.count()
                provider = getattr(self._vector_store, 'provider', 'unknown')
                _check("Vector Store", "ok",
                       backend=type(self._vector_store).__name__,
                       vector_count=vector_count,
                       embedding_provider=provider)
            except Exception as e:
                _check("Vector Store", "error", error=str(e))
        else:
            _check("Vector Store", "error", error="Not initialized")

        # -- Graph Store --
        if self._graph:
            try:
                if hasattr(self._graph, "graph_stats"):
                    graph_stats = self._graph.graph_stats()
                else:
                    graph_stats = self._graph.stats()
                _check("Graph Store", "ok",
                       backend=type(self._graph).__name__,
                       nodes=graph_stats.get("nodes", 0),
                       edges=graph_stats.get("edges", 0))
            except Exception as e:
                _check("Graph Store", "error", error=str(e))
        else:
            _check("Graph Store", "error", error="Not initialized")

        # -- Retrieval Pipeline --
        layers = ["tag_match"]
        if self._graph:
            layers.append("graph_expansion")
        if self._vector_store:
            layers.append("vector_similarity")
        _check("Retrieval Pipeline", "ok",
               active_layers=len(layers), max_layers=3, layers=layers)

        return report

    # -- Export / Import --

    def export_json(self, filepath: str) -> None:
        """Export all memories to a JSON file."""
        memories = self._store.list_memories(limit=1_000_000)
        data = [m.to_dict() for m in memories]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)

    def import_json(self, filepath: str) -> int:
        """Import memories from a JSON file. Returns count imported."""
        with open(filepath) as f:
            data = json.load(f)
        count = 0
        for item in data:
            memory = Memory(
                id=item.get("id", Memory().id),
                content=item["content"],
                tags=item.get("tags", []),
                category=item.get("category", "general"),
                entity=item.get("entity"),
                created_at=item.get("created_at", datetime.now(timezone.utc).isoformat()),
                event_date=item.get("event_date"),
                valid_from=item.get("valid_from", datetime.now(timezone.utc).isoformat()),
                valid_until=item.get("valid_until"),
                superseded_by=item.get("superseded_by"),
                confidence=item.get("confidence", 1.0),
                status=item.get("status", "active"),
                metadata=item.get("metadata", {}),
            )
            try:
                self._store.insert(memory)
                count += 1
            except Exception:
                # Skip duplicates
                pass
        return count

    # -- Raw Query --

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw query. Use with caution."""
        return self._store.execute(sql, params)
```

- [ ] **Step 3: Run ALL tests to verify nothing broke**

Run: `.venv/bin/pytest tests/ -v --ignore=tests/test_arango_smoke.py --ignore=tests/test_arango_backend.py -m "not docker"`
Expected: All existing tests PASS

- [ ] **Step 4: Commit**

```bash
git add agent_memory/core.py
git commit -m "feat: core.py uses registry to resolve and instantiate backends"
```

---

### Task 12: Add python-arango Optional Dependency

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Update pyproject.toml**

Add `arangodb` to optional dependencies:

In `[project.optional-dependencies]`, add:
```toml
arangodb = ["python-arango>=8.0.0"]
```

Update `all` to include it:
```toml
all = ["openai>=1.0.0", "python-arango>=8.0.0"]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "feat: add python-arango as optional dependency"
```

---

### Task 13: Full Integration Test — Both Backend Sets

**Files:**
- Test: `tests/test_integration_backends.py`

- [ ] **Step 1: Write integration test**

```python
# tests/test_integration_backends.py
"""Integration tests verifying both backend configurations work end-to-end."""

import tempfile
import pytest

from agent_memory import AgentMemory
from agent_memory.models import Memory


class TestDefaultBackends:
    """Verify default SQLite+ChromaDB+NetworkX still works through abstraction."""

    def test_add_and_recall(self):
        with tempfile.TemporaryDirectory() as d:
            mem = AgentMemory(d)
            mem.add("Python is my favorite language", tags=["python"], category="preference", entity="user")
            mem.add("I work at Acme Corp", tags=["work"], category="career", entity="user")
            results = mem.recall("what language does user prefer?")
            assert len(results) >= 1
            mem.close()

    def test_health_check(self):
        with tempfile.TemporaryDirectory() as d:
            mem = AgentMemory(d)
            report = mem.health()
            assert report["healthy"] is True
            assert len(report["checks"]) >= 3
            mem.close()

    def test_stats(self):
        with tempfile.TemporaryDirectory() as d:
            mem = AgentMemory(d)
            mem.add("test memory")
            s = mem.stats()
            assert s["total_memories"] == 1
            mem.close()

    def test_temporal_contradiction(self):
        with tempfile.TemporaryDirectory() as d:
            mem = AgentMemory(d)
            mem.add("User lives in NYC", entity="user", category="location")
            mem.add("User lives in SF", entity="user", category="location")
            facts = mem.current_facts(entity="user", category="location")
            assert len(facts) == 1
            assert "SF" in facts[0].content
            mem.close()


try:
    from arango import ArangoClient
    HAS_ARANGO = True
except ImportError:
    HAS_ARANGO = False

docker_required = pytest.mark.skipif(
    not HAS_ARANGO, reason="python-arango not installed"
)


@docker_required
@pytest.mark.docker
class TestArangoDBBackend:
    """Verify ArangoDB backend works end-to-end through AgentMemory."""

    @pytest.fixture(autouse=True)
    def setup_arango(self):
        from agent_memory.infra.docker import DockerManager
        dm = DockerManager()
        dm.ensure_running(
            backend_name="arangodb-integration",
            image="arangodb/arangodb:latest",
            port=8531,
            env={"ARANGO_NO_AUTH": "1"},
            health_timeout=60,
        )
        import time
        time.sleep(3)
        yield
        dm.stop("arangodb-integration")
        # Clean up database
        client = ArangoClient(hosts="http://localhost:8531")
        sys_db = client.db("_system")
        if sys_db.has_database("memwright_integration"):
            sys_db.delete_database("memwright_integration")

    def test_add_and_recall(self):
        with tempfile.TemporaryDirectory() as d:
            mem = AgentMemory(d, config={
                "backends": ["arangodb"],
                "arangodb": {
                    "mode": "cloud",
                    "url": "http://localhost:8531",
                    "database": "memwright_integration",
                },
            })
            mem.add("Python is great", tags=["python"], category="preference", entity="user")
            # Recall via document store (tag match)
            results = mem.recall("python")
            assert len(results) >= 1
            mem.close()

    def test_health_check(self):
        with tempfile.TemporaryDirectory() as d:
            mem = AgentMemory(d, config={
                "backends": ["arangodb"],
                "arangodb": {
                    "mode": "cloud",
                    "url": "http://localhost:8531",
                    "database": "memwright_integration",
                },
            })
            report = mem.health()
            assert report["healthy"] is True
            mem.close()
```

- [ ] **Step 2: Run default backend tests**

Run: `.venv/bin/pytest tests/test_integration_backends.py::TestDefaultBackends -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Run ArangoDB integration tests (requires Docker)**

Run: `.venv/bin/pytest tests/test_integration_backends.py::TestArangoDBBackend -v -m docker`
Expected: All 2 tests PASS

- [ ] **Step 4: Run the full existing test suite one final time**

Run: `.venv/bin/pytest tests/ -v --ignore=tests/test_arango_smoke.py --ignore=tests/test_arango_backend.py -m "not docker"`
Expected: ALL existing tests PASS — zero regressions

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration_backends.py
git commit -m "test: add integration tests for both backend configurations"
```

---

### Task 14: Configure pytest markers

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add docker marker to pytest config**

In `[tool.pytest.ini_options]`, add:
```toml
markers = [
    "docker: tests requiring Docker (deselect with '-m not docker')",
]
```

- [ ] **Step 2: Commit**

```bash
git add pyproject.toml
git commit -m "chore: add docker pytest marker"
```
