# Codebase Concerns

**Analysis Date:** 2026-03-15

## Tech Debt

**Silent Failure Pattern in Optional Services:**
- Issue: Vector store and Neo4j initialization failures are caught and logged as errors but silently allow the system to continue with degraded functionality. This is by design, but creates potential for confusing behavior: queries work but return incomplete results with no indication that layers are disabled.
- Files: `agent_memory/core.py` (lines 50-94), `agent_memory/retrieval/orchestrator.py` (lines 46-67, 95-117)
- Impact: Users can unknowingly operate without semantic search or graph expansion, thinking they have full retrieval when they don't. The non-fatal exception handlers on lines 172-173, 190-191, 251-252 mask operational issues.
- Fix approach: Add explicit warnings in health() output when services fail to initialize, and log specifics (stack traces) at DEBUG level. Consider a retrieval_layers property on AgentMemory that shows which layers are active.

**Mutation of Memory Objects:**
- Issue: Memory dataclass fields are modified in-place during supersession and forgetting operations rather than creating new objects
- Files: `agent_memory/temporal/manager.py` (lines 72-74), `agent_memory/core.py` (line 343)
- Impact: Violates immutability principle. If a Memory object is referenced elsewhere or used in concurrent operations, mutations will be visible unexpectedly. The `old_memory.status = "superseded"` pattern is risky.
- Fix approach: Create a new Memory instance with updated fields using dataclass.replace() or a copy_with() method, then update the store with the new object.

**SQL String Interpolation with User Input:**
- Issue: Tag search in `sqlite_store.py` line 151 uses parameterized queries correctly, but the `execute()` raw SQL method (lines 194-199) accepts arbitrary SQL without validation
- Files: `agent_memory/store/sqlite_store.py` (lines 194-199)
- Impact: The public API on `core.py` line 574 exposes `execute()` as a documented feature for "raw SQL", creating a potential SQL injection vector if users pass untrusted input. This is a footgun.
- Fix approach: Remove the public execute() method or add strict documentation about security implications. If needed, make it private (_execute) or require explicit opt-in via a security flag.

**Exception Handlers Too Broad:**
- Issue: Multiple `except Exception:` blocks (not `except SpecificException:`) catch all exceptions including system errors, resource leaks, and unexpected bugs
- Files: `agent_memory/core.py` (lines 59, 85, 111, 116, 121, 172, 190, 251, 394, 413, 440, 458, 500, 520), `agent_memory/retrieval/orchestrator.py` (lines 66, 116), `agent_memory/embeddings.py` (lines 101, 126), `agent_memory/extraction/llm_extractor.py` (lines 199, 221)
- Impact: Silent failures mask real bugs. A KeyError or AttributeError gets swallowed the same as a ConnectionError, making debugging difficult.
- Fix approach: Catch specific exceptions (ConnectionError, ImportError, ValueError, etc.) and handle each appropriately. Preserve unknown exceptions by re-raising or at least logging full stack traces.

**Cache LRU Without Eviction Policy Documentation:**
- Issue: Embedding cache in `embeddings.py` (lines 23-25, 97-99) uses OrderedDict LRU with hard-coded MAX=256, but this limit is not configurable and may not suit all use cases
- Files: `agent_memory/embeddings.py` (lines 23-25, 97-99)
- Impact: Large bulk operations with >256 unique texts will constantly evict cached embeddings, defeating the purpose. No way to configure cache size.
- Fix approach: Make cache size configurable via MemoryConfig or environment variable. Consider cache statistics (hit/miss rate) in health checks.

## Known Bugs

**Temporal Boost Mutates Result Objects:**
- Symptoms: Calling temporal_boost() or entity_boost() modifies RetrievalResult.score in-place
- Files: `agent_memory/retrieval/scorer.py` (lines 22-45, 48-69)
- Trigger: Any call to recall() with enable_temporal_boost=True (default)
- Impact: If results are reused or shared, modifications are visible to all references. The scoring pattern `r.score += boost` mutates existing objects.
- Fix approach: Create new RetrievalResult instances with updated scores instead of modifying in-place: `RetrievalResult(memory=r.memory, score=r.score + boost, match_source=r.match_source)`

**Tag Search Uses LIKE Without Anchoring:**
- Symptoms: Tag search in `sqlite_store.py` line 152 uses `tags LIKE '%{tag}%'`, which matches substrings
- Files: `agent_memory/store/sqlite_store.py` (lines 150-152)
- Trigger: Searching for tag "python" will match memory with tag "cpython" or "pythonic"
- Impact: False positives in tag-based retrieval. User searches for "python" but gets memories tagged with "cpython".
- Fix approach: Tags are stored as JSON array strings. Use JSON-aware querying (json_extract in SQLite) or parse tags in the application layer to check for exact matches.

**Health Check Subprocess Calls Without Timeout Protection on Some Calls:**
- Symptoms: Some health check subprocess.run() calls have timeout=5, but docker compose up command in CLI might hang
- Files: `agent_memory/core.py` (lines 385-387), `agent_memory/cli.py` (lines 328-335)
- Trigger: Docker daemon is slow or unresponsive
- Impact: health() can hang for 5+ seconds per check if Docker is slow. CLI setup can hang indefinitely.
- Fix approach: Add consistent timeout=5 to all subprocess calls. Wrap in try/except for TimeoutExpired.

## Security Considerations

**API Key Exposure in Error Messages:**
- Risk: If OpenRouter/OpenAI API calls fail, exception messages might leak the API key in stack traces
- Files: `agent_memory/embeddings.py` (lines 74-102), `agent_memory/mcp/server.py` (lines 180-183)
- Current mitigation: Broad `except Exception: return None` silences exceptions but creates observability blind spots
- Recommendations: Catch exceptions explicitly and sanitize error messages. Never include headers or full HTTP details in logs. Use a safe error formatter that strips Authorization headers.

**Neo4j Cypher Injection via Relationship Type:**
- Risk: Relationship types are sanitized in `_sanitize_rel_type()` but f-string interpolation at line 112 of `neo4j_graph.py` is still a potential injection vector if sanitization is bypassed
- Files: `agent_memory/graph/neo4j_graph.py` (lines 18-25, 109-113)
- Current mitigation: Regex validation in _sanitize_rel_type(), but only applied to explicit relation_type parameter, not to auto-detected types
- Recommendations: Use Cypher parameterized queries where possible (Neo4j driver supports APOC queries with params). Document that relationship types must be alphanumeric.

**No Input Validation on Memory Content:**
- Risk: Memory content field accepts arbitrary strings, including very large payloads or special characters
- Files: `agent_memory/models.py` (line 15), `agent_memory/core.py` (line 133-142)
- Current mitigation: None
- Recommendations: Add length limits (e.g., max 10KB per memory). Validate content in Memory.__init__() or the add() method. Reject null bytes or control characters.

## Performance Bottlenecks

**Tag Search Uses LIKE with Wildcard Prefix:**
- Problem: LIKE '%tag%' cannot use an index effectively (wildcard on left side prevents index usage)
- Files: `agent_memory/store/sqlite_store.py` (lines 150-152)
- Cause: JSON tags are stored as serialized strings, not parsed. Each search requires a table scan.
- Improvement path: Normalize tags to a separate tag junction table (memories → tags), or use JSON extraction functions (json_each) with indexed lookups.

**Vector Search Creates New numpy Array on Every Query:**
- Problem: `vector_store.py` line 65 creates `np.array(query_embedding, dtype=np.float32)` on every search
- Files: `agent_memory/store/vector_store.py` (lines 56-76)
- Cause: No caching of preprocessed embeddings
- Improvement path: Cache the vectorized query embedding for repeated searches on the same query. Consider batching search calls.

**Neo4j Query Depth Limitation:**
- Problem: Variable-length path queries (line 134 in neo4j_graph.py) with f-string depth `[*1..{d}]` scale poorly as depth increases
- Files: `agent_memory/graph/neo4j_graph.py` (lines 125-139, 141-167)
- Cause: Traversing 2+ hops requires exploring exponentially more paths
- Improvement path: Add a max traversal limit. Consider caching subgraph results. Use Neo4j's apoc.path.subgraphAll() function with configurable depth and breadth limits.

**Batch Embedding Loads All Memories into Memory:**
- Problem: `batch_embed()` loads all 100,000 active memories into memory before processing
- Files: `agent_memory/core.py` (lines 312-335)
- Cause: list_memories(limit=100_000) materializes entire result set
- Improvement path: Use pagination with limit/offset to process in smaller chunks (already done with batch_size, but the initial fetch is monolithic).

## Fragile Areas

**Retrieval Pipeline Depends on Graph and Vector Being Optional:**
- Files: `agent_memory/retrieval/orchestrator.py`
- Why fragile: The recall() method has no way to signal that it's using a degraded pipeline. If both vector_store and graph are None, it only returns tag matches, but the caller doesn't know this is suboptimal. Changing the initialization order in core.py could silently break assumptions.
- Safe modification: Add a RuntimeError or explicit status check if both advanced layers fail. Or add properties to RetrievalOrchestrator showing active layers.
- Test coverage: `tests/test_retrieval.py` tests tag matching but may not cover degradation modes. No tests for "vector store fails after initialization" scenario.

**Memory Status State Machine Not Enforced:**
- Files: `agent_memory/models.py` (line 40), `agent_memory/core.py` (line 343), `agent_memory/temporal/manager.py` (line 72)
- Why fragile: Status can transition from "active" → "archived", "active" → "superseded", but there's no validation preventing invalid transitions (e.g., archived → superseded) or double-archiving. Direct mutation of the status field bypasses any state machine.
- Safe modification: Create Status enum. Define valid transitions in TemporalManager methods. Disallow direct status assignment on Memory.
- Test coverage: `tests/test_temporal.py` tests basic supersession but not edge cases like re-superseding an already superseded memory.

**Import Order and Circular Dependencies:**
- Files: `agent_memory/core.py` (lines 52, 74, 167, 178, 229), `agent_memory/extraction/extractor.py` (lines 54-58)
- Why fragile: Lazy imports inside methods hide dependency issues at module load time. If VectorStore or Neo4jGraph have import errors, they only surface during add() or recall(), not at initialization.
- Safe modification: Import all required packages at module level to catch errors early. Use optional dependencies with clear error messages at install time.
- Test coverage: No test explicitly covers "what happens if neo4j package is not installed but PostgreSQL is".

**Default Mutable Arguments:**
- Files: Multiple: `agent_memory/models.py` (line 18, 43), `agent_memory/core.py` (line 141)
- Why fragile: `tags: List[str] = field(default_factory=list)` is correct, but mutable defaults in function signatures are a Python anti-pattern. No instances found in this codebase, but worth monitoring.
- Test coverage: Good use of default_factory in dataclasses prevents this issue.

## Scaling Limits

**SQLite Concurrent Write Limit:**
- Current capacity: SQLite with WAL mode allows concurrent reads but serializes writes. One write at a time.
- Limit: Under high-concurrency workloads (>100 writes/second), SQLite will timeout or lock
- Scaling path: Migrate to PostgreSQL for the core store (currently only used for vectors). Use a connection pool and transaction batching.

**Neo4j Memory Growth:**
- Current capacity: Depends on heap size (default ~1GB in Docker). Each add() extracts entities and relations, creating new nodes/edges.
- Limit: A 10M memory graph with high entity density will run out of heap
- Scaling path: Implement entity deduplication (MERGE behavior is already there). Add memory limits to extraction or batch cleanup jobs.

**pgvector Index Size:**
- Current capacity: IVFFlat index with lists=100 is suitable for ~10K embeddings
- Limit: Vector table will grow without a compaction strategy. Archived/deleted memories still have vectors.
- Scaling path: Implement cascade delete for archived memories' vectors. Rebuild index periodically with appropriate lists parameter (sqrt(n_vectors)).

**Embedding API Rate Limits:**
- Current capacity: OpenRouter/OpenAI allows burst requests, but batch_embed() and bulk add operations will hit rate limits
- Limit: batch_embed() on 100K memories will exhaust API quota quickly at $0.02 per 1M tokens
- Scaling path: Implement exponential backoff retry logic (partially exists in mab.py lines 580-625 but not in core embeddings). Add rate limit handling to get_embeddings_batch().

## Dependencies at Risk

**openai Package Version Pinning:**
- Risk: No explicit version constraint in pyproject.toml. OpenAI client API changes frequently.
- Impact: Major version upgrades could break embeddings.py or llm_extractor.py
- Migration plan: Pin to `openai>=1.0,<2` in pyproject.toml. Create a compatibility layer for OpenAI API changes.

**pgvector Extension Compatibility:**
- Risk: pgvector versions must match between Python pgvector package and PostgreSQL extension
- Impact: Version mismatch causes schema initialization failures
- Migration plan: Document exact version requirements (e.g., pgvector==0.2.x requires PostgreSQL 14+). Add version check to health().

**Neo4j Driver Connectivity:**
- Risk: Neo4j driver in containerized environment assumes default auth (neo4j/memwright). If auth changes, initialization fails silently.
- Impact: `add()` and `recall()` fail to update graph without user notification
- Migration plan: Validate Neo4j credentials at initialization (already done with verify_connectivity() on line 41 of neo4j_graph.py, but errors are caught broadly in core.py).

## Test Coverage Gaps

**Contradiction Detection Edge Cases:**
- What's not tested: Multiple contradictions for same entity. Supersession chains (A superseded by B, B superseded by C). Temporal contradictions with event_date ordering.
- Files: `agent_memory/temporal/manager.py` (lines 49-68), `tests/test_temporal.py`
- Risk: Cascading supersession could create orphaned memory chains where B→C→D but A is not properly marked as transitively superseded
- Priority: High - temporal logic is core to the system

**Retrieval Layer Degradation:**
- What's not tested: Recall when vector_store is None but graph exists. Recall when graph is None but vector_store exists. Both None simultaneously.
- Files: `agent_memory/retrieval/orchestrator.py`, `tests/test_retrieval.py`
- Risk: Unknown behavior under partial initialization. Results could be duplicates or weighted incorrectly.
- Priority: High - deployment scenarios where services partially fail are realistic

**Batch Operations on Large Datasets:**
- What's not tested: batch_embed() with >10K memories. import_json() with malformed data. export_json() on large stores.
- Files: `agent_memory/core.py` (lines 312-335, 543-570, 536-541), `tests/test_core.py`
- Risk: Memory exhaustion or corrupted imports that silently skip duplicates
- Priority: Medium - edge case but impact is data loss

**Concurrent Access:**
- What's not tested: Multiple AgentMemory instances on same store. Concurrent add() and recall() calls.
- Files: `agent_memory/core.py`, `agent_memory/store/sqlite_store.py`, `tests/test_store.py`
- Risk: SQLite write locks will cause timeouts or deadlocks. No test validates concurrent safety.
- Priority: Medium - not a documented use case but plausible in multi-agent scenarios

**MCP Server Error Handling:**
- What's not tested: Tool call with malformed JSON. Tool call with missing required fields. Network errors during tool execution.
- Files: `agent_memory/mcp/server.py` (lines 175-184), `tests/` (no mcp tests)
- Risk: MCP server crashes on invalid input instead of returning proper error responses
- Priority: Medium - MCP is published to registry, so external users will hit this

---

*Concerns audit: 2026-03-15*
