"""AgentMemory main class — the public API."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent_memory.models import Memory, RetrievalResult
from agent_memory.retrieval.orchestrator import RetrievalOrchestrator
from agent_memory.store.sqlite_store import SQLiteStore
from agent_memory.temporal.manager import TemporalManager
from agent_memory.utils.config import MemoryConfig, load_config, save_config

logger = logging.getLogger("agent_memory")


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

        # Initialize store
        db_path = self.path / "memory.db"
        self._store = SQLiteStore(db_path)

        # Initialize pgvector store
        self._vector_store = None
        if self.config.pg_connection_string:
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
        else:
            logger.error(
                "No pg_connection_string configured. "
                "Set PG_CONNECTION_STRING in .env"
            )

        # Initialize Neo4j graph
        self._graph = None
        if self.config.neo4j_password:
            try:
                from agent_memory.graph.neo4j_graph import Neo4jGraph
                self._graph = Neo4jGraph(
                    uri=self.config.neo4j_uri,
                    auth=(self.config.neo4j_user, self.config.neo4j_password),
                    database=self.config.neo4j_database,
                )
            except ImportError:
                logger.error(
                    "Neo4j requires the neo4j package. "
                    "Install with: pip install memwright[neo4j]"
                )
            except Exception as e:
                logger.error(
                    "Could not connect to Neo4j (%s). "
                    "Run: docker compose up -d", e,
                )
        else:
            logger.error(
                "No neo4j_password configured. "
                "Set NEO4J_PASSWORD in .env"
            )

        # Initialize managers
        self._temporal = TemporalManager(self._store)
        self._retrieval = RetrievalOrchestrator(
            self._store,
            min_results=self.config.min_results,
            vector_store=self._vector_store,
            graph=self._graph,
        )


    def close(self) -> None:
        """Close all database connections."""
        if self._vector_store:
            try:
                self._vector_store.close()
            except Exception:
                pass
        if self._graph:
            try:
                self._graph.save()  # no-op for Neo4j
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

    # ── Write ──

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

        # Store embedding in vector DB (if enabled)
        if self._vector_store:
            try:
                from agent_memory.embeddings import get_embedding

                embedding = get_embedding(content)
                if embedding:
                    self._vector_store.add(memory.id, content, embedding)
            except Exception:
                pass  # Non-fatal: vector store failure doesn't block memory storage

        # Update entity graph (if enabled)
        if self._graph:
            try:
                from agent_memory.graph.extractor import extract_entities_and_relations

                nodes, edges = extract_entities_and_relations(
                    content, tags or [], entity, category
                )
                for node in nodes:
                    self._graph.add_entity(node["name"], node["type"], node.get("attributes", {}))
                for edge in edges:
                    self._graph.add_relation(
                        edge["from"], edge["to"], edge["type"], edge.get("metadata", {})
                    )
                self._graph.save()
            except Exception:
                pass  # Non-fatal

        return memory

    def get(self, memory_id: str) -> Optional[Memory]:
        """Get a specific memory by ID."""
        return self._store.get(memory_id)

    # ── Read ──

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
        if query and self._vector_store:
            # Use vector search with filters applied after
            try:
                from agent_memory.embeddings import get_embedding

                query_embedding = get_embedding(query)
                if query_embedding:
                    vec_results = self._vector_store.search(query_embedding, limit=limit * 2)
                    memories = []
                    for vr in vec_results:
                        memory = self._store.get(vr["memory_id"])
                        if memory:
                            memories.append(memory)
                    # Apply filters
                    if category:
                        memories = [m for m in memories if m.category == category]
                    if entity:
                        memories = [m for m in memories if m.entity and entity.lower() in m.entity.lower()]
                    if status:
                        memories = [m for m in memories if m.status == status]
                    if after:
                        memories = [m for m in memories if m.created_at >= after]
                    if before:
                        memories = [m for m in memories if m.created_at <= before]
                    return memories[:limit]
            except Exception:
                pass  # Fall through to list_memories
            return self._store.list_memories(
                status=status, category=category, entity=entity,
                after=after, before=before, limit=limit,
            )
        elif query:
            # No vector store — fall back to listing with filters
            return self._store.list_memories(
                status=status, category=category, entity=entity,
                after=after, before=before, limit=limit,
            )
        else:
            return self._store.list_memories(
                status=status,
                category=category,
                entity=entity,
                after=after,
                before=before,
                limit=limit,
            )

    # ── Timeline ──

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

    # ── Extraction ──

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

    # ── Batch Operations ──

    def batch_embed(self, batch_size: int = 100) -> int:
        """Batch-compute and store embeddings for all active memories.

        More efficient than per-memory embedding during add() since it uses
        batch API calls. Returns count of newly embedded memories.
        """
        if not self._vector_store:
            return 0

        from agent_memory.embeddings import get_embeddings_batch

        all_memories = self._store.list_memories(status="active", limit=100_000)
        count = 0

        for i in range(0, len(all_memories), batch_size):
            batch = all_memories[i : i + batch_size]
            texts = [m.content for m in batch]
            embeddings = get_embeddings_batch(texts)
            for mem_obj, emb in zip(batch, embeddings):
                if emb is not None:
                    self._vector_store.add(mem_obj.id, mem_obj.content, emb)
                    count += 1

        return count

    # ── Maintenance ──

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
        """Check health of all components. Returns structured status report.

        Each required service gets its own check mark. No fallbacks —
        if a service is down, it's marked as down.
        """
        import shutil
        import subprocess
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

        # ── 1. Docker Daemon ──
        docker_ok = False
        if shutil.which("docker"):
            try:
                result = subprocess.run(
                    ["docker", "info"], capture_output=True, timeout=5,
                )
                if result.returncode == 0:
                    docker_ok = True
                    _check("Docker", "ok")
                else:
                    _check("Docker", "error",
                           error="Daemon not running. Start Docker Desktop.")
            except Exception as e:
                _check("Docker", "error", error=str(e))
        else:
            _check("Docker", "error", error="Docker not installed")

        # ── 2. PostgreSQL Container ──
        if docker_ok:
            try:
                cr = subprocess.run(
                    ["docker", "inspect", "--format",
                     "{{.State.Status}}", "memwright-postgres"],
                    capture_output=True, text=True, timeout=5,
                )
                state = cr.stdout.strip() if cr.returncode == 0 else "not found"
                if state == "running":
                    _check("PostgreSQL Container", "ok")
                else:
                    _check("PostgreSQL Container", "error",
                           error=f"Container state: {state}. Fix: docker compose up -d")
            except Exception as e:
                _check("PostgreSQL Container", "error", error=str(e))
        else:
            _check("PostgreSQL Container", "error",
                   error="Docker not available")

        # ── 3. pgvector Connection ──
        try:
            import psycopg
            t0 = time.monotonic()
            conn = psycopg.connect(self.config.pg_connection_string, autocommit=True)
            cur = conn.cursor()
            cur.execute("SELECT 1")
            latency = round((time.monotonic() - t0) * 1000, 1)
            # Check pgvector extension
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            row = cur.fetchone()
            conn.close()
            if row:
                _check("pgvector", "ok", version=f"v{row[0]}", latency_ms=latency)
            else:
                _check("pgvector", "ok",
                       note="Extension not yet installed (created on first use)",
                       latency_ms=latency)
        except ImportError:
            _check("pgvector", "error",
                   error="psycopg not installed. Fix: pip install memwright[vectors]")
        except Exception as e:
            _check("pgvector", "error",
                   error="PostgreSQL unreachable. Fix: docker compose up -d")

        # ── 4. Neo4j Container ──
        if docker_ok:
            try:
                cr = subprocess.run(
                    ["docker", "inspect", "--format",
                     "{{.State.Status}}", "memwright-neo4j"],
                    capture_output=True, text=True, timeout=5,
                )
                state = cr.stdout.strip() if cr.returncode == 0 else "not found"
                if state == "running":
                    _check("Neo4j Container", "ok")
                else:
                    _check("Neo4j Container", "error",
                           error=f"Container state: {state}. Fix: docker compose up -d")
            except Exception as e:
                _check("Neo4j Container", "error", error=str(e))
        else:
            _check("Neo4j Container", "error",
                   error="Docker not available")

        # ── 5. Neo4j Connection ──
        try:
            from neo4j import GraphDatabase
            t0 = time.monotonic()
            driver = GraphDatabase.driver(
                self.config.neo4j_uri,
                auth=(self.config.neo4j_user, self.config.neo4j_password),
            )
            driver.verify_connectivity()
            latency = round((time.monotonic() - t0) * 1000, 1)
            info = driver.get_server_info()
            driver.close()
            details: Dict[str, Any] = {"latency_ms": latency}
            if info:
                details["version"] = str(info.agent)
            _check("Neo4j", "ok", **details)
        except ImportError:
            _check("Neo4j", "error",
                   error="neo4j package not installed. Fix: pip install memwright[neo4j]")
        except Exception as e:
            _check("Neo4j", "error",
                   error="Neo4j unreachable. Fix: docker compose up -d")

        # ── 6. SQLite ──
        try:
            t0 = time.monotonic()
            row = self._store._conn.execute("SELECT COUNT(*) FROM memories").fetchone()
            latency = round((time.monotonic() - t0) * 1000, 1)
            _check("SQLite", "ok",
                   memory_count=row[0],
                   db_path=str(self._store.db_path),
                   db_size_bytes=(
                       self._store.db_path.stat().st_size
                       if self._store.db_path.exists() else 0
                   ),
                   latency_ms=latency)
        except Exception as e:
            _check("SQLite", "error", error=str(e))

        # ── 7. Embeddings API ──
        try:
            from agent_memory.embeddings import available, _get_client, _model

            if available():
                client = _get_client()
                provider = "unknown"
                if hasattr(client, "base_url"):
                    base_url = str(client.base_url)
                    if "openrouter" in base_url:
                        provider = "OpenRouter"
                    else:
                        provider = "OpenAI"
                _check("Embeddings API", "ok", provider=provider, model=_model)
            else:
                _check("Embeddings API", "error",
                       error="No API key. Set OPENROUTER_API_KEY or OPENAI_API_KEY")
        except Exception as e:
            _check("Embeddings API", "error", error=str(e))

        # ── 8. Retrieval Pipeline ──
        layers = ["tag_match"]
        if self._graph:
            layers.append("graph_expansion")
        if self._vector_store:
            layers.append("vector_similarity")
        _check("Retrieval Pipeline", "ok",
               active_layers=len(layers), max_layers=3, layers=layers)

        return report

    # ── Export / Import ──

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

    # ── Raw SQL ──

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL. Use with caution."""
        return self._store.execute(sql, params)
