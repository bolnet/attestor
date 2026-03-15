# Coding Conventions

**Analysis Date:** 2026-03-15

## Naming Patterns

**Files:**
- Lowercase with underscores: `sqlite_store.py`, `neo4j_graph.py`, `llm_extractor.py`
- Modules mirror functionality: `store/sqlite_store.py`, `graph/neo4j_graph.py`
- Test files prefixed with `test_`: `test_core.py`, `test_store.py`, `test_temporal.py`
- Each module covers one responsibility: storage, graph operations, extraction, retrieval, temporal logic

**Functions:**
- snake_case throughout: `extract_entities_and_relations()`, `get_embedding()`, `tag_search()`
- Public methods follow public-private convention (no leading underscore for public)
- Private/internal functions use leading underscore: `_rule_extract()`, `_llm_extract()`, `_get_client()`, `_sanitize_rel_type()`
- Descriptive action verbs: `insert()`, `update()`, `delete()`, `retrieve()`, `search()`, `recall()`

**Variables:**
- snake_case for all variables: `test_config`, `mem_dir`, `embedding_cache`, `query_embedding`
- Private module-level variables use leading underscore: `_client`, `_initialized`, `_embedding_cache`, `_CACHE_MAX`
- Constants in UPPERCASE: `_CACHE_MAX = 256`, `DEFAULT_CONFIG`, `TEST_CONFIG`, `_SCHEMA_PATH`
- Descriptive names over abbreviations: `vector_store` not `vs`, `memory_id` not `mid`

**Types:**
- Use type hints on all function signatures: `def get(self, memory_id: str) -> Optional[Memory]:`
- Use `from __future__ import annotations` for forward references and string annotations
- Use `Optional[T]` for nullable types, `List[T]` for collections
- Dict type hints use generic form: `Dict[str, Any]`, `Dict[str, str]`

## Code Style

**Formatting:**
- No explicit formatter configured (black/ruff not in pyproject.toml)
- Follow Python style by convention: 4-space indentation, max line length ~90-100 chars
- Blank lines between class methods and logical sections (see markers like `# ── Write ──`, `# ── Read ──`)

**Linting:**
- No linting configuration files detected (.flake8, mypy.ini, ruff.toml not present)
- Rely on type hints for static analysis
- Code uses consistent patterns (parameterized queries, exception handling, logging)

## Import Organization

**Order:**
1. `from __future__ import annotations` (first line in every module)
2. Standard library imports: `json`, `sqlite3`, `logging`, `os`, `sys`, `re`, `tempfile`, `uuid`, `hashlib`
3. Third-party imports: `pytest`, `psycopg`, `neo4j`, `openai`, `numpy`
4. Local imports: `from agent_memory.models import Memory`, `from agent_memory.store.sqlite_store import SQLiteStore`

**Path Aliases:**
- No import aliases configured
- Use absolute imports: `from agent_memory.models import Memory` not relative imports
- All imports within `agent_memory` package use full module paths

**Example structure** (`agent_memory/core.py`):
```python
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_memory.models import Memory, RetrievalResult
from agent_memory.retrieval.orchestrator import RetrievalOrchestrator
```

## Error Handling

**Patterns:**
- Explicit exception handling with try/except blocks at operation boundaries
- Non-fatal failures silently pass: `except Exception: pass` used strategically for optional components (vector store, Neo4j graph)
- Core SQLite path never breaks — errors in pgvector/Neo4j are caught and logged
- Logging uses `logger.error()` with context: `logger.error("Could not connect to PostgreSQL (%s). Run: docker compose up -d", e)`

**Example** (`agent_memory/core.py` lines 51-63):
```python
try:
    from agent_memory.store.vector_store import VectorStore
    self._vector_store = VectorStore(self.config.pg_connection_string)
except ImportError:
    logger.error(
        "pgvector requires psycopg. "
        "Install with: pip install memwright[vectors]"
    )
except Exception as e:
    logger.error(
        "Could not connect to PostgreSQL (%s). "
        "Run: docker compose up -d", e,
    )
```

## Logging

**Framework:** Python built-in `logging` module

**Patterns:**
- Module-level logger: `logger = logging.getLogger("agent_memory")` (see `core.py` line 17)
- Use `logger.error()` for actionable errors with context
- No debug logging in hot paths (retrieval, embedding lookup)
- Log connection failures with remediation hints: "Run: docker compose up -d"

## Comments

**When to Comment:**
- Module docstrings always present: `"""AgentMemory main class — the public API."""`
- Public class docstrings with usage examples: `core.py` lines 21-27
- Function docstrings document purpose and return values: all functions in `models.py`, `config.py`
- Comments mark logical sections with dividers: `# ── Write ──`, `# ── Read ──`, `# ── Temporal ──`
- Comments explain WHY, not WHAT: "Non-fatal: vector store failure doesn't block memory storage"

**JSDoc/TSDoc:**
- Not applicable (Python project, no TypeScript/JavaScript)
- Python docstrings follow standard format with Args/Returns sections

## Function Design

**Size:**
- Functions stay focused and small
- `get_embedding()`: 15 lines
- `extract_memories()`: 8 lines
- `delete()`: 4 lines
- Largest methods in `AgentMemory` class ~50-100 lines but still readable due to logical separation

**Parameters:**
- Use dataclasses for complex configuration: `MemoryConfig` in `utils/config.py`
- Optional parameters default to `None`: `budget: Optional[int] = None`
- Configuration passed as dict config at init time: `AgentMemory(path, config={...})`
- Avoid positional args beyond 3 — use keyword args for clarity

**Return Values:**
- Single-responsibility returns: functions return one thing
- Use dataclass wrappers for composite returns: `RetrievalResult` contains memory, score, match_source
- Explicitly return `None` for missing cases: `return None` not silent failures
- Return counts/status for mutation operations: `compact() -> int`, `forget() -> bool`

## Module Design

**Exports:**
- Public API in `__init__.py`: `from agent_memory.core import AgentMemory`
- Classes export via `__all__`: seen in `__init__.py` lines 3-6
- Internal modules not exported: `extractor`, `scorer`, `tag_matcher` are used internally

**Barrel Files:**
- Package init files minimal: `__init__.py` exports public classes only
- No deep re-exports: `from agent_memory.models import Memory` used directly
- Lazy imports in core (`from agent_memory.store.vector_store import VectorStore` done at init time, not module load)

## Conventions in Action

**Example: Dataclass-based models** (`models.py`):
```python
@dataclass
class Memory:
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    content: str = ""
    tags: List[str] = field(default_factory=list)

    def tags_json(self) -> str:
        return json.dumps(self.tags)

    @classmethod
    def from_row(cls, row: Dict[str, Any]) -> Memory:
        # Factory method for deserialization
```

**Example: Configuration management** (`utils/config.py`):
```python
@dataclass
class MemoryConfig:
    default_token_budget: int = 2000

    def __post_init__(self):
        """Environment variable overrides — env vars win over defaults."""
        env_pg = os.environ.get("PG_CONNECTION_STRING")
        if env_pg:
            self.pg_connection_string = env_pg
```

**Example: Method organization in classes** (`core.py` lines 131-273):
- Public methods grouped by concern: `# ── Write ──`, `# ── Read ──`, `# ── Timeline ──`
- Related operations together: `add()`, `get()`, `recall()`, `search()`
- Maintenance operations separate: `forget()`, `compact()`, `stats()`

---

*Convention analysis: 2026-03-15*
