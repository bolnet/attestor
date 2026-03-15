# Testing Patterns

**Analysis Date:** 2026-03-15

## Test Framework

**Runner:**
- pytest 7.0+ (specified in `pyproject.toml` line 47: `pytest>=7.0`)
- Config: `pyproject.toml` lines 56-57
  ```toml
  [tool.pytest.ini_options]
  testpaths = ["tests"]
  ```
- Tests located in `tests/` directory (12 test files)

**Assertion Library:**
- Python built-in `assert` statements
- No external assertion library (not pytest-assertions, should, or similar)
- Direct equality checks: `assert retrieved.content == "test fact"`

**Run Commands:**
```bash
.venv/bin/pytest tests/ -v              # Run all tests with verbose output
.venv/bin/pytest tests/ -v --cov        # Run with coverage (pytest-cov installed)
.venv/bin/pytest tests/test_core.py -v  # Run single test file
```

## Test File Organization

**Location:**
- Colocated in `tests/` directory separate from source code
- Pattern: `tests/test_{module}.py` mirrors `agent_memory/{module}.py`
  - `agent_memory/core.py` → `tests/test_core.py`
  - `agent_memory/store/sqlite_store.py` → `tests/test_store.py`
  - `agent_memory/temporal/manager.py` → `tests/test_temporal.py`

**Naming:**
- Test files: `test_core.py`, `test_store.py`, `test_extraction.py`
- Test classes: `TestInit`, `TestCRUD`, `TestSearch`, `TestContradiction`, `TestTimeline`
- Test methods: `test_add_and_get`, `test_get_nonexistent`, `test_supersedes_old_memory`
- Pattern: `test_<behavior_or_scenario>()`

**Structure:**
```
tests/
├── conftest.py                 # Shared fixtures, test config
├── test_core.py               # AgentMemory main API
├── test_store.py              # SQLiteStore crud
├── test_vector_store.py       # pgvector integration
├── test_neo4j_graph.py        # Neo4j graph (with mocks)
├── test_retrieval.py          # Retrieval orchestrator
├── test_temporal.py           # Contradiction/supersession logic
├── test_extraction.py         # Memory extraction
├── test_embeddings.py         # Embedding computation
├── test_scorer.py             # Retrieval scoring
├── test_graph_extractor.py    # Entity/relation extraction
├── test_cli.py                # CLI commands
├── test_mab.py                # MemoryAgentBench benchmark
└── __init__.py
```

## Test Structure

**Suite Organization** (`test_core.py`):
```python
class TestInit:
    def test_creates_directory(self, mem_dir, test_config):
        path = os.path.join(mem_dir, "subdir", "nested")
        m = AgentMemory(path, config=test_config)
        assert os.path.isdir(path)
        m.close()

class TestCRUD:
    def test_add_and_get(self, mem):
        m = mem.add("User likes Python", tags=["preference"])
        assert m.id
        retrieved = mem.get(m.id)
        assert retrieved.content == "User likes Python"

class TestSearch:
    def test_search_by_category(self, mem):
        mem.add("Python", category="preference")
        results = mem.search(category="preference")
        assert all(r.category == "preference" for r in results)
```

**Patterns:**

1. **Setup via Fixtures:**
   - Fixtures defined in `conftest.py` (shared across all tests)
   - Fixtures auto-used: `mem`, `mem_dir`, `test_config` injected by pytest
   - Setup runs before test, teardown after (context manager pattern)

2. **Teardown:**
   - Fixtures use context managers: `with tempfile.TemporaryDirectory() as d:`
   - Explicit cleanup in conftest: `m.close()` called in fixture teardown
   - Database connections closed after test

3. **Assertion Pattern:**
   - Simple assertions: `assert retrieved is not None`
   - Conditional assertions: `assert all(r.category == preference for r in results)`
   - Check state changes: `assert old_retrieved.status == "superseded"`

**Example: Full test cycle** (`test_store.py` lines 28-34):
```python
def test_update(self, store):
    m = Memory(content="original", tags=["a"])
    store.insert(m)
    m.content = "updated"
    store.update(m)
    retrieved = store.get(m.id)
    assert retrieved.content == "updated"
```

## Mocking

**Framework:**
- `unittest.mock` from Python standard library
- `MagicMock` and `patch` used for isolating external dependencies

**Patterns:**

1. **Pre-import mocking** (`test_neo4j_graph.py` lines 10-12):
   ```python
   mock_neo4j = MagicMock()
   sys.modules.setdefault("neo4j", mock_neo4j)
   from agent_memory.graph.neo4j_graph import Neo4jGraph
   ```
   - Mock the neo4j module before importing code that uses it
   - Allows tests to run without actual Neo4j driver installed

2. **Patch in test method** (`test_neo4j_graph.py` lines 36-40):
   ```python
   with patch("agent_memory.graph.neo4j_graph.GraphDatabase") as mock_gdb:
       mock_driver = MagicMock()
       mock_gdb.driver.return_value = mock_driver
       mock_driver.execute_query.return_value = ([], MagicMock(), [])
       graph = Neo4jGraph(...)
   ```
   - Patch at point of use: `"agent_memory.graph.neo4j_graph.GraphDatabase"`
   - Set return values for method calls: `execute_query.return_value = (...)`

3. **Verify mock behavior** (`test_neo4j_graph.py` line 51):
   ```python
   mock_driver.verify_connectivity.assert_called_once()
   ```
   - Assert mocked methods were called with expected args

**What to Mock:**
- External services (Neo4j, PostgreSQL, OpenAI API)
- File system operations (use `tempfile` instead)
- Time/datetime (use fixtures or monkeypatch)
- Optional components that may not be installed

**What NOT to Mock:**
- Core domain logic (Memory, MemoryConfig, aggregation logic)
- Business rules (contradiction detection, temporal supersession)
- Storage layer (use actual SQLite via fixtures, not mocked)
- Retrieval scoring (use actual scorer functions)

## Fixtures and Factories

**Test Data** (`conftest.py`):
```python
TEST_CONFIG = {
    "pg_connection_string": "postgresql://memwright:memwright@localhost:5432/memwright_test",
    "neo4j_uri": "bolt://localhost:7688",
    "neo4j_user": "neo4j",
    "neo4j_password": "memwright",
    "neo4j_database": "neo4j",
}

@pytest.fixture
def test_config():
    """Config dict pointing to test databases."""
    return dict(TEST_CONFIG)

@pytest.fixture
def mem(mem_dir):
    m = AgentMemory(mem_dir, config=TEST_CONFIG)
    yield m
    m.close()
```

**Test data in tests** (`test_extraction.py` lines 30-35):
```python
messages = [
    {"role": "user", "content": "I work at SoFi as a staff engineer."},
    {"role": "assistant", "content": "That's great!"},
    {"role": "user", "content": "I prefer using Python for backend work."},
]
memories = extract_memories(messages)
```
- Inline test data for clarity
- Use realistic examples: conversation messages with roles

**Location:**
- Fixtures in `tests/conftest.py` (shared across all tests)
- Test-specific factories inline in test files
- No separate factory/builder module

## Coverage

**Requirements:**
- No explicit coverage requirement configured in `pyproject.toml`
- `pytest-cov>=4.0` available as dev dependency (line 47)
- Expected minimum: 80% (based on project guidelines in CLAUDE.md)

**View Coverage:**
```bash
.venv/bin/pytest tests/ --cov=agent_memory --cov-report=html
```

## Test Types

**Unit Tests:**
- Scope: Individual functions and classes
- Approach: Fast, isolated, no external dependencies
- Examples:
  - `test_store.py`: SQLiteStore CRUD operations
  - `test_embeddings.py`: Embedding computation (with mocked API)
  - `test_scorer.py`: Score calculation functions
- Pattern: Create minimal object, call one method, assert result

**Integration Tests:**
- Scope: Multiple components working together
- Approach: Use real SQLite database (via fixtures), mock external APIs
- Examples:
  - `test_core.py`: AgentMemory full workflow (add → search → recall)
  - `test_temporal.py`: Contradiction detection with temporal logic
  - `test_retrieval.py`: Retrieval orchestrator with tag/graph/vector layers
- Pattern: Set up state across components, exercise workflow, verify end-to-end behavior

**E2E Tests:**
- Framework: Not used
- Rationale: Project provides CLI and library API, not web application
- Alternative: `test_cli.py` tests command-line interface end-to-end
- Benchmark tests: `test_mab.py` runs MemoryAgentBench for real-world evaluation

**Example: Integration test** (`test_temporal.py` lines 15-27):
```python
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
```

## Common Patterns

**Async Testing:**
- Not applicable (no async code in project)
- All operations are synchronous

**Error Testing:**
- Testing error paths: `test_get_nonexistent()` (returns None)
- Testing graceful degradation: No test for API failures (caught silently with `except: pass`)
- Example (`test_store.py` lines 42-43):
  ```python
  def test_delete_nonexistent(self, store):
      assert not store.delete("nonexistent")
  ```

**Testing with Temporary Resources** (`test_store.py` lines 11-16):
```python
@pytest.fixture
def store():
    with tempfile.TemporaryDirectory() as d:
        s = SQLiteStore(f"{d}/memory.db")
        yield s
        s.close()
```
- Use `tempfile.TemporaryDirectory()` context manager
- Automatically cleaned up after test
- Separate test database per test (isolation)

**State validation** (`test_core.py` lines 67-71):
```python
def test_forget(self, mem):
    m = mem.add("temp fact")
    assert mem.forget(m.id)
    retrieved = mem.get(m.id)
    assert retrieved.status == "archived"
```
- Verify mutation changed state: add → modify → retrieve → check status

**Skipping optional component tests:**
- No `pytest.skip` or `pytest.mark.skipif` observed
- Instead: Neo4j tests use pre-import mocking to avoid dependency
- VectorStore tests can be skipped if PostgreSQL not running (Docker required)

## Test Execution Notes

**Requirements:** Docker must be running
- PostgreSQL container: `memwright-postgres` on port 5432
- Neo4j container: `memwright-neo4j` on port 7688 (test instance)
- Start with: `docker compose up -d`

**Isolation:**
- Each test uses separate temp directory for SQLite
- Each test uses separate test config pointing to test databases
- No test data pollution across runs

**Determinism:**
- All time values use ISO strings (not system clock)
- No random data generation
- Mock embeddings return deterministic values

---

*Testing analysis: 2026-03-15*
