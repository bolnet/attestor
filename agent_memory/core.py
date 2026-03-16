"""AgentMemory main class -- the public API."""

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

        # Initialize ChromaDB vector store
        self._vector_store = None
        try:
            from agent_memory.store.chroma_store import ChromaStore
            self._vector_store = ChromaStore(self.path)
        except Exception as e:
            logger.warning("ChromaDB init failed: %s", e)

        # Initialize NetworkX graph
        self._graph = None
        try:
            from agent_memory.graph.networkx_graph import NetworkXGraph
            self._graph = NetworkXGraph(self.path)
        except Exception as e:
            logger.warning("NetworkX init failed: %s", e)

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
                pass  # Fall through to SQLite search

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
        """Batch-index all active memories into ChromaDB.

        ChromaDB handles embedding generation internally.
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
        """Check health of all components. Returns structured status report.

        Checks: SQLite, ChromaDB, NetworkX Graph, Retrieval Pipeline.
        No Docker checks. No external API checks.
        """
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

        # -- SQLite --
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

        # -- ChromaDB --
        if self._vector_store:
            try:
                vector_count = self._vector_store.count()
                chroma_dir = self.path / "chroma"
                _check("ChromaDB", "ok",
                       vector_count=vector_count,
                       chroma_dir=str(chroma_dir),
                       dir_exists=chroma_dir.exists())
            except Exception as e:
                _check("ChromaDB", "error", error=str(e))
        else:
            _check("ChromaDB", "error", error="Not initialized")

        # -- NetworkX Graph --
        if self._graph:
            try:
                graph_stats = self._graph.stats()
                graph_path = self.path / "graph.json"
                _check("NetworkX Graph", "ok",
                       nodes=graph_stats["nodes"],
                       edges=graph_stats["edges"],
                       graph_file=str(graph_path),
                       file_exists=graph_path.exists())
            except Exception as e:
                _check("NetworkX Graph", "error", error=str(e))
        else:
            _check("NetworkX Graph", "error", error="Not initialized")

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

    # -- Raw SQL --

    def execute(self, sql: str, params: Optional[List[Any]] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL. Use with caution."""
        return self._store.execute(sql, params)
