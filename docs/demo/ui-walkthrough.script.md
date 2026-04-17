# Memwright UI Walkthrough — Demo Script

**Duration**: ~3 minutes
**Prerequisites**: `memwright ui --path ~/.memwright` running on port 8080
**Screens**: 13 stops across 8 tabs + interactions

---

## Screen 01 — Evidence Board (Memories)
**URL**: `/ui/memories`
**Duration**: 15s

> The Evidence Board is the landing page. Every memory in the store appears as a card in a forensic-archive grid. Cards show content excerpt, category badge, entity anchor, confidence stamp, and creation date.

**What to show**:
- Grid of memory cards with tilt-based confidence stamps
- Category and entity labels on each card
- Record count in top-right chrome ("26 records · 26 active")
- Footer ticker: "MEMWRIGHT V2 · DETERMINISTIC RETRIEVAL · NO LLM IN CRITICAL PATH"

**Narration**: "This is the Evidence Board — every fact your agents have stored. Cards are ranked by confidence. Active, decayed, and superseded states are visually distinct."

---

## Screen 02 — Search & Filter
**URL**: `/ui/memories?q=architecture`
**Duration**: 10s

**What to show**:
- Type "architecture" in the search field
- Cards filter to architecture-related memories
- **Search highlighting**: matched terms wrapped in `<mark class="hl">` with rust-tinted underline
- Pagination controls at bottom

**Narration**: "Full-text search with instant highlighting. Pagination handles large stores — 60 cards per page with HTMX-powered swaps."

---

## Screen 03 — Memory Detail (Dossier)
**URL**: `/ui/memories/{id}` (click any card)
**Duration**: 20s

**What to show**:
- Click a memory card to open the dossier view
- Full content, confidence stamp, metadata fields
- Tab through: Content → Provenance → Supersession → Embedding → Graph → Access
- Show "Nearest neighbors" sidebar (cosine similarity scores)
- Show graph edges sidebar if entity has connections

**Narration**: "Every memory has a full dossier. Six tabs: content, provenance chain, supersession history, nearest vector neighbors, graph neighborhood, and access tracking with confidence decay."

---

## Screen 04 — Knowledge Graph
**URL**: `/ui/graph`
**Duration**: 15s

**What to show**:
- Full Cytoscape.js force-directed graph
- Entity nodes with typed edges (uses, depends-on, related_to)
- Hover a node to see tooltip with entity type
- Click a node to zoom and highlight connections
- Search for "order-service" in graph search box

**Narration**: "The knowledge graph visualizes entity relationships. Nodes are entities, edges are typed relations extracted from memory content. Click any node to explore its neighborhood."

---

## Screen 05 — Recall Pipeline
**URL**: `/ui/recall`
**Duration**: 25s

**What to show**:
1. Type query: "how does the order service work?"
2. Set budget: 2000
3. Click "Execute Recall"
4. Show animated pipeline diagram with **5 layers + latency badges**:
   - Tag Match: N results, X.X ms
   - Graph Expansion: N results, X.X ms
   - Vector Search: N results, X.X ms
   - Fusion + Rank: N results, X.X ms
   - Diversity + Fit: N results, X.X ms
5. Show summary: "N candidates entered → N results delivered in X.X ms"
6. Show funnel metric row
7. Scroll to per-layer detail cards with individual results and scores

**Narration**: "The recall pipeline replays the 5-layer retrieval cascade in real time. Each layer shows result count AND latency. Tag match finds keyword hits, graph expansion walks entity relationships, vector search finds semantic neighbors, fusion ranks with RRF, and diversity removes near-duplicates under the token budget."

---

## Screen 06 — Budget Explorer
**URL**: `/ui/recall` (same page, click Budget Explorer)
**Duration**: 10s

**What to show**:
1. With a query already entered, click "Budget Explorer"
2. Show the bar chart comparing 5 budget levels (500, 1000, 2000, 5000, 10000)
3. Each row shows latency bar (color-coded) and result count bar

**Narration**: "The budget explorer runs the same query at five budget levels so you can tune the token budget for your agent. Smaller budgets are faster but return fewer results."

---

## Screen 07 — Timeline
**URL**: `/ui/timeline`
**Duration**: 10s

**What to show**:
- Chronological timeline of all memories
- Validity windows (valid_from → valid_until)
- Supersession markers where contradictions were detected
- Temporal replay context

**Narration**: "The timeline shows when facts entered the system and their validity windows. Contradictions are auto-detected — older facts get superseded, never deleted."

---

## Screen 08 — Agents
**URL**: `/ui/agents`
**Duration**: 10s

**What to show**:
- Agent activity breakdown by namespace
- RBAC role distribution (ORCHESTRATOR, PLANNER, EXECUTOR, etc.)
- Per-agent memory counts and write activity

**Narration**: "The agents view shows multi-agent activity. Each agent operates in a namespace with RBAC roles. You can see who wrote what and when."

---

## Screen 09 — System Health
**URL**: `/ui/health`
**Duration**: 15s

**What to show**:
1. "ALL SYSTEMS OPERATIONAL" banner
2. **Operation Latency panel**: P50/P95/P99 percentiles + sparkline
3. Four component cards:
   - SQLiteStore: latency, record count, DB size
   - ChromaStore: latency, vector count, embedding provider
   - NetworkXGraph: latency, node count, edge count
   - Retrieval Pipeline: active layers (3/3)
4. Auto-refresh indicator (30s)

**Narration**: "System health shows all four components with per-store latency. The Operation Latency panel computes P50, P95, P99 percentiles from the ops ring buffer with a sparkline of recent operations."

---

## Screen 10 — Configuration
**URL**: `/ui/config`
**Duration**: 10s

**What to show**:
- Storage backends (SQLite, ChromaDB, NetworkX)
- Role assignments (document → sqlite, vector → chroma, graph → networkx)
- Embedding provider (local, all-MiniLM-L6-v2)
- Retrieval pipeline config (fusion_mode: rrf, mmr_lambda: 0.7, confidence_gate, decay rates)
- Store statistics

**Narration**: "The config viewer shows the full system configuration: which backends serve each role, embedding provider, retrieval tuning parameters, and store statistics. Everything is read-only."

---

## Screen 11 — Operations Log
**URL**: `/ui/ops`
**Duration**: 15s

**What to show**:
1. Summary row: Total Ops, Recalls, Adds, P50, P95, P99
2. Sparkline of recent operation latencies
3. Flight recorder table:
   - Timestamp (HH:MM:SS.mmm)
   - Op type badge (RECALL / ADD / HEALTH — color-coded)
   - Latency with visual bar
   - Input preview
   - Result count
   - Store participation (document, vector, graph)
4. Auto-refresh every 5s

**Narration**: "The operations log is a flight recorder. Every add, recall, and health check is captured with sub-millisecond timestamps, latency bars, and store participation. It auto-refreshes every 5 seconds."

---

## Screen 12 — Dark/Light Theme Toggle
**URL**: any page
**Duration**: 5s

**What to show**:
- Click the half-circle theme toggle button (top-right)
- Watch the entire UI switch from dark (forensic archive) to light theme
- Click again to return to dark

**Narration**: "Theme toggle switches between the dark forensic archive aesthetic and a light mode. Preference persists in localStorage."

---

## Screen 13 — Export
**URL**: `/ui/memories`
**Duration**: 5s

**What to show**:
- Show the Export JSON and Export CSV buttons in the sidebar
- Click Export JSON — raw JSON file downloads
- Show the JSON structure (id, content, tags, category, entity, namespace, timestamps, confidence)

**Narration**: "Export all memories as JSON or CSV for offline analysis, migration, or backup."

---

## Closing

**Duration**: 5s

**Narration**: "That's Memwright — 8 pages, 23 routes, zero LLM in the critical path. Self-hosted, deterministic retrieval, sub-10ms recalls. Install with one command: `install agent memory`."

---

## Recording Notes

- **Resolution**: 1280x800 viewport
- **Browser**: Chrome/Chromium, no extensions
- **Font rendering**: Ensure Fraunces, Departure Mono, Instrument Sans load from Google Fonts
- **Data**: Seed at least 20+ memories across multiple entities and namespaces
- **Timing**: Pause 2-3s on each screen for readability
- **Mouse**: Slow, deliberate movements — forensic aesthetic demands precision
