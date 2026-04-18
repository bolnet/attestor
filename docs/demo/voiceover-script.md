# Attestor UI — Voiceover Script

**Total duration**: ~4 minutes
**Video**: `docs/demo/ui-walkthrough.webm`

---

## [0:00] Evidence Board — Landing Page

This is the Evidence Board — the landing page. Every memory your agents have stored appears here as a card in a forensic-archive grid. Each card shows a content excerpt, category badge, entity anchor, confidence stamp, and creation date. Cards animate in with a staggered waterfall effect. You can see 29 records across 1 namespace. The footer ticker pulses live — confirming this is a real, running system.

---

## [0:08] Search with Live Filtering

Let's search. I'll type "architecture" — notice the HTMX-powered live filtering kicks in after a 350-millisecond debounce. Cards filter down instantly. Matched terms are highlighted with a rust-tinted underline — that's real text-node walking, not innerHTML replacement.

---

## [0:18] Namespace and Status Filters

I can also filter by namespace — here "default". And switch the status dropdown to "superseded" to see contradicted facts. These are memories the system auto-detected as outdated and replaced. Back to "active" to see current facts.

---

## [0:30] Card Hover and Detail Navigation

Hovering a card lifts it with a subtle translateY animation — the forensic aesthetic demands precision. Let me click into a memory to see the full dossier.

---

## [0:35] Memory Dossier — Six Tabs

Every memory has a full dossier with six tabs. We start on Content — the full text. Let me cycle through each tab.

**Provenance** shows the memory ID, content hash, namespace, status, and agent metadata. This is the chain of custody for every fact.

**Supersession** shows the linked chain — which memory this one replaced, and what replaced it. You can click through the chain to trace contradictions.

**Embedding** shows the nearest neighbors by cosine similarity — these are the memories most semantically similar to this one. Scores are shown to three decimal places.

**Graph** shows the entity's PageRank score and its one-hop neighborhood — every edge and relationship this entity participates in.

**Access** shows how many times this memory has been retrieved, when it was last accessed, and the current confidence with decay explanation. Confidence decays over time but gets boosted on each access.

---

## [1:15] Knowledge Graph

The knowledge graph visualizes all entity relationships. Nodes are entities, edges are typed relations — uses, depends-on, related-to. The force-directed layout settles organically.

I can click any node to highlight its neighborhood — everything else dims. The inspector panel on the right shows label, entity type, PageRank, degree, and all connected neighbors. Clicking a neighbor in the inspector zooms to that node.

Let me search for "Attestor" — the graph dims everything except matching nodes. I'll clear the search and try different layouts. Concentric arranges nodes by centrality. Circle distributes them evenly. Back to the force-directed layout.

I can also change node sizing — PageRank makes important nodes larger, degree sizes by connection count. The Fit View button resets the viewport.

---

## [1:55] Recall Pipeline

The recall pipeline replays the 5-layer retrieval cascade in real time. I'll type a query: "how does the order service work?" — set the token budget to 2000 — and hit Execute Recall.

Watch the pipeline diagram animate. Five layers fire in sequence:

**Tag Match** finds keyword hits via SQLite full-text search.
**Graph Expansion** walks entity relationships via BFS at depth 2.
**Vector Search** finds semantic neighbors via cosine similarity.
**Fusion and Rank** combines results using Reciprocal Rank Fusion at k=60, then applies PageRank and confidence decay scoring.
**Diversity and Fit** removes near-duplicates with MMR at lambda 0.7, then greedily packs results into the token budget.

Each layer shows its result count and latency in milliseconds. The summary line shows: how many candidates entered, how many results delivered, and total latency. Let me scroll down to see the per-layer detail cards with individual results and their scores.

---

## [2:35] Budget Explorer

Now let me click Budget Explorer. This runs the same query at five budget levels — 500, 1000, 2000, 5000, and 10,000 tokens. The chart shows latency bars color-coded by speed: green under 10 milliseconds, yellow under 50, red above. Smaller budgets are faster but return fewer results. This is how you tune the token budget for your agent's context window.

---

## [2:50] Temporal Timeline

The timeline shows when facts entered the system along a vertical spine. Cards alternate left and right. Each card shows entity, category, content excerpt, creation timestamp, namespace, and validity window.

Let me click a card — the detail panel slides in from the right showing the full dossier: ID, entity, category, confidence, creation date, validity range, access count, tags, and complete content. I'll close it.

Now I'll filter to superseded memories only. These show the pulsing arrow icon and "superseded by" links — older facts that were contradicted. The system auto-detects contradictions and supersedes older facts. Nothing is ever deleted. Back to "all" to see the full timeline.

---

## [3:15] Agent Registry

The agent registry shows multi-agent activity by namespace. Each card displays the namespace name, memory count, category distribution as a proportional color bar, top entities with frequency counts, and an activity sparkline showing write patterns over time.

Clicking a card opens the detail overlay. Inside: entity frequency table, category distribution, and a paginated list of recent memories with content excerpts, categories, entities, timestamps, and confidence scores. I'll scroll through the detail panel and close the overlay.

---

## [3:30] System Health

System health shows all four components at a glance. The green banner confirms "All Systems Operational." The latency panel shows P50, P95, and P99 percentiles computed from the operations ring buffer, with an SVG sparkline of recent latencies.

Four component cards: SQLite document store with latency and record count. ChromaDB vector store with embedding provider and vector count. NetworkX graph store with node and edge counts. Retrieval pipeline with active layer status. This page auto-refreshes every 30 seconds.

---

## [3:45] Configuration

The config viewer shows the full system configuration — read-only. Storage backends: which backend serves each role. Embedding provider: local all-MiniLM-L6-v2 at 384 dimensions. Retrieval pipeline tuning: RRF fusion mode, k=60, MMR lambda 0.7, BFS depth 2, confidence decay and boost rates. Store statistics: total memories, active count, namespaces, categories, graph nodes and edges.

---

## [3:55] Operations Log

The operations log is a flight recorder. Every add, recall, and health check is captured with sub-millisecond timestamps. The summary row shows total ops, recall count, add count, and P50/P95/P99 latencies. The sparkline visualizes the latency trend.

Each row in the table shows: timestamp, operation type badge color-coded by type, latency with a visual bar — green under 10ms, yellow under 50ms, red above — input preview, result count, and store participation. This auto-refreshes every 5 seconds.

---

## [4:10] Theme Toggle

Finally, the theme toggle. One click switches from the dark forensic archive aesthetic to a clean light mode. Backgrounds invert, cards get white with subtle shadows, text goes dark. Preference persists in localStorage. One more click — back to the forensic archive.

---

## [4:18] Closing

That's Attestor. Eight pages, 23 routes, zero LLM in the critical path. Self-hosted, deterministic retrieval, sub-10 millisecond recalls. Install with one command: "install attestor."
