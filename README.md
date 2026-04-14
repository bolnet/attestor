<!-- ══════════════════════════════════════════════════════════════
     MASTHEAD
     ══════════════════════════════════════════════════════════════ -->

<p align="center"><sub>&sect; 00 &middot; MASTHEAD &middot; FILED UNDER INFRASTRUCTURE &middot; BY SURENDRA SINGH &middot; &mdash; FOR PUBLICATION &mdash;</sub></p>

<p align="center"><sub><b>MEMWRIGHT</b> &mdash; A MEMORY JOURNAL FOR AGENTIC SYSTEMS &middot; VOL. 02 &middot; REV. 0.1 &middot; EST. 2026 &middot; NEW YORK &middot; MIT</sub></p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/bolnet/agent-memory/main/docs/logo.svg">
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/bolnet/agent-memory/main/docs/logo-dark.svg">
    <img alt="Memwright" src="https://raw.githubusercontent.com/bolnet/agent-memory/main/docs/logo.svg" width="400">
  </picture>
</p>

<h3 align="center"><em>The memory layer for agent teams.</em></h3>

<p align="center"><sub>Self&#8209;hosted &middot; Deterministic retrieval &middot; No LLM in the critical path</sub></p>

<p align="center">
  <a href="https://pypi.org/project/memwright/"><img src="https://img.shields.io/pypi/v/memwright?color=C15F3C&style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.org/project/memwright/"><img src="https://img.shields.io/pypi/dm/memwright?style=flat-square&color=C15F3C" alt="PyPI downloads"></a>
  <a href="https://github.com/bolnet/agent-memory/stargazers"><img src="https://img.shields.io/github/stars/bolnet/agent-memory?style=flat-square&color=C15F3C" alt="GitHub stars"></a>
  <a href="https://pypi.org/project/memwright/"><img src="https://img.shields.io/pypi/pyversions/memwright?style=flat-square" alt="Python"></a>
  <a href="https://github.com/bolnet/agent-memory/blob/main/LICENSE"><img src="https://img.shields.io/github/license/bolnet/agent-memory?style=flat-square" alt="License"></a>
  <a href="https://registry.modelcontextprotocol.io/servers/io.github.bolnet/memwright"><img src="https://img.shields.io/badge/MCP-Registry-C15F3C?style=flat-square" alt="MCP Registry"></a>
</p>

<p align="center"><sub><a href="#problem">Problem &darr;</a> &middot; <a href="#multi-agent">Multi&#8209;Agent &darr;</a> &middot; <a href="#pipeline">Pipeline &darr;</a> &middot; <a href="#deploy">Deploy &darr;</a> &middot; <a href="#principles">Principles &darr;</a> &middot; <a href="#reference">Reference &darr;</a></sub></p>

---

### The problem

**Agent prototypes don&rsquo;t survive production. Memory is why.** Single agents rediscover the same facts every run. Multi&#8209;agent pipelines are worse &mdash; the planner&rsquo;s decisions never reach the executor, the researcher&rsquo;s findings never reach the reviewer. Teams paper over it by stuffing giant prompts between agents, burning tokens on stale context. That&rsquo;s a workaround, not an architecture.

### The solution &mdash; at a glance

```mermaid
flowchart TB
    %% === AGENT TIER ===
    subgraph AGENTS [<b>&sect; AGENT TIER</b> &mdash; financial advisory pipeline]
        direction LR
        A1[<b>Portfolio<br/>Planner</b>]
        A2[<b>Market<br/>Researcher</b>]
        A3[<b>Risk<br/>Analyst</b>]
        A4[<b>Compliance<br/>Reviewer</b>]
        A1 ~~~ A2 ~~~ A3 ~~~ A4
    end

    %% === API SURFACE ===
    AGENTS ==>|<b>mem.add&#40;&#41;</b><br/>content &bull; tags &bull; entity| WRITE{{<b>WRITE PATH</b>}}
    AGENTS ==>|<b>mem.recall&#40;query, budget&#41;</b>| READ{{<b>READ PATH</b>}}

    %% === STORAGE TIER ===
    subgraph STORAGE [<b>&sect; STORAGE TIER</b> &mdash; three complementary stores, one write transaction]
        direction LR
        S1[(<b>Document store</b><br/><sub>SQLite &bull; Postgres &bull; Cosmos<br/><i>source of truth</i></sub>)]
        S2[(<b>Vector store</b><br/><sub>ChromaDB &bull; pgvector &bull; DiskANN<br/><i>semantic index</i></sub>)]
        S3[(<b>Graph store</b><br/><sub>NetworkX &bull; Apache AGE<br/><i>relational index</i></sub>)]
    end

    WRITE --> S1
    WRITE --> S2
    WRITE --> S3

    %% === RETRIEVAL TIER ===
    subgraph RETRIEVAL [<b>&sect; RETRIEVAL TIER</b> &mdash; 5 deterministic layers &bull; zero LLM calls]
        direction TB

        subgraph SOURCES [Stage A &bull; parallel sources]
            direction LR
            L1[<b>01 &bull; Tag Match</b><br/><sub>SQLite FTS<br/>exact + partial</sub>]
            L2[<b>02 &bull; Graph Expansion</b><br/><sub>multi-hop BFS<br/>depth 2</sub>]
            L3[<b>03 &bull; Vector Search</b><br/><sub>cosine similarity<br/>top-k nearest</sub>]
        end

        L4[<b>04 &bull; Fusion &amp; Rank</b><br/><sub>RRF k=60 &nbsp;&bull;&nbsp; PageRank &nbsp;&bull;&nbsp; confidence decay</sub>]
        L5[<b>05 &bull; Diversity &amp; Fit</b><br/><sub>MMR &lambda;=0.7 &nbsp;&bull;&nbsp; greedy token-budget pack</sub>]

        L1 --> L4
        L2 --> L4
        L3 --> L4
        L4 ==> L5
    end

    %% read fan-out and storage wiring
    READ --> L1
    READ --> L2
    READ --> L3

    S1 -. indexes .-> L1
    S3 -. traverses .-> L2
    S2 -. embeds .-> L3

    %% === OUTPUT ===
    L5 ==>|<b>ranked memories</b><br/>fit to caller&rsquo;s token budget| CTX[[<b>Agent context window</b>]]

    %% === STYLING ===
    style AGENTS    fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style A1        fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style A2        fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style A3        fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style A4        fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style STORAGE   fill:#FBF8F1,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style RETRIEVAL fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style SOURCES   fill:#F5F1E8,stroke:#6B5F4F,stroke-dasharray:4 3,color:#1A1614
    style WRITE     fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style READ      fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style L1        fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style L2        fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style L3        fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style L4        fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style L5        fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style CTX       fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
```

<sub>One shared memory tier. Every agent writes, every agent recalls. Deterministic retrieval, no LLM in the critical path, no SaaS middleman. Details in &sect; 01&ndash;&sect; 05 below.</sub>

---

<p align="center">
  <b>Production&#8209;grade memory infrastructure for multi&#8209;agent systems.</b><br>
  <sub>The memory tier your agents need when they leave your laptop and start running in production.</sub>
</p>

<p align="center"><sub>Namespace isolation &middot; RBAC &middot; Provenance tracking &middot; Temporal correctness &middot; Ranked retrieval &middot; Token budgets &mdash; built for orchestrator&#8209;worker and planner&#8209;executor pipelines. Python library, REST API, or containerized service. No SaaS middleman, no per&#8209;seat fees, no vendor lock&#8209;in.</sub></p>

```python
from agent_memory import AgentMemory

mem = AgentMemory("./store")
mem.add("Order service uses event sourcing", entity="order-service", tags=["arch"])
mem.recall("how is the order service structured?", budget=2000)
```

```bash
poetry add memwright
```

<p align="center"><sub>MIT &middot; Python 3.10&ndash;3.14 &middot; Production deploy in one command</sub></p>

<table align="center">
<tr><th colspan="2" align="center">&sect; Spec Sheet</th></tr>
<tr><td>Storage Roles</td><td><b>Doc &middot; Vector &middot; Graph</b></td></tr>
<tr><td>Interfaces</td><td><b>Python &middot; REST &middot; MCP</b></td></tr>
<tr><td>Retrieval Layers</td><td><b>5</b></td></tr>
<tr><td>RBAC Roles</td><td><b>6</b></td></tr>
<tr><td>Cloud Targets</td><td><b>AWS &middot; Azure &middot; GCP</b></td></tr>
<tr><td>License</td><td><b>MIT</b></td></tr>
</table>

---

<a id="problem"></a>

## &sect; 01 &mdash; The Problem

<sub><b>Why agent prototypes don't survive production</b></sub>

Agent prototypes don't survive production. Memory is usually why.

Single agents rediscover the same facts every run. Multi&#8209;agent pipelines are worse &mdash; the planner's decisions never reach the executor, the researcher's findings never reach the reviewer. Teams end up stuffing giant prompts between agents to paper over the gap. That's not an architecture &mdash; that's a workaround.

<sub><i>What we hear from teams building agent pipelines:</i></sub>

> *We had a planner, a coder, a reviewer, a deployer &mdash; four agents in a pipeline. None of them knew what the others learned. We were passing giant prompts between them and burning tokens on stale information.*

<table>
<tr><th>Without Memwright</th><th>With Memwright</th></tr>
<tr><td>01 &mdash; Each agent starts blind &mdash; no knowledge of what others learned</td><td>01 &mdash; Shared memory &mdash; planner writes, coder reads, reviewer sees both</td></tr>
<tr><td>02 &mdash; Giant prompts passed between agents burn context tokens</td><td>02 &mdash; Token&#8209;budget recall &mdash; each agent pulls only what fits</td></tr>
<tr><td>03 &mdash; No access control &mdash; any agent can overwrite any state</td><td>03 &mdash; Six RBAC roles, namespace isolation, write quotas per agent</td></tr>
<tr><td>04 &mdash; Contradicting facts from different agents go undetected</td><td>04 &mdash; Contradictions auto&#8209;resolved &mdash; newer facts supersede older ones</td></tr>
<tr><td>05 &mdash; Session ends, everything learned is gone forever</td><td>05 &mdash; Persistent across sessions, pipelines, and agent restarts</td></tr>
</table>

More agents, more sessions, more memories &mdash; retrieval gets better while context cost stays flat.

---

<a id="multi-agent"></a>

## &sect; 02 &mdash; Multi-Agent Systems

<sub><b>Orchestrator &middot; Planner &middot; Executor &middot; Reviewer</b></sub>

Not a chatbot plugin. Infrastructure for agent teams.

Every recall and write is scoped to an `AgentContext` &mdash; a lightweight dataclass carrying identity, role, namespace, parent trail, token budget, write quota, and visibility. Contexts are immutable; spawning a sub&#8209;agent returns a new context with inherited provenance.

| # | Primitive | What it does |
|---|---|---|
| 01 | **Namespace isolation** | Every agent, project, or tenant gets its own namespace. Planner writes, coder reads, reviewer sees both. Isolated by default, shared when you configure it. |
| 02 | **Six RBAC roles** | Orchestrator, Planner, Executor, Researcher, Reviewer, Monitor. Read&#8209;only observers to full admins. |
| 03 | **Provenance tracking** | Know which agent wrote which memory, when, and under which parent session. The reviewer can trace a decision back to the planner three sessions ago. |
| 04 | **Cross&#8209;agent contradiction resolution** | Agent A learns *"user works at Google."* Agent B learns *"user works at Meta."* Memwright auto&#8209;supersedes. Full history preserved. Zero inference calls in the critical path. |
| 05 | **Token budgets per agent** | `recall(query, budget=2000)` &mdash; a summarizer uses 500 tokens; a deep reasoner uses 5,000. Each agent receives exactly what fits in its context window. |
| 06 | **Write quotas &amp; review flags** | Rate&#8209;limit writes per namespace, flag writes for human review, add compliance tags for audit. |

---

<a id="pipeline"></a>

## &sect; 03 &mdash; The Retrieval Pipeline

<sub><b>Five layers &middot; zero inference calls</b></sub>

Five layers. No LLM. Everything deterministic.

When an agent calls `recall(query, budget)`, five cooperating layers find, fuse, score, and fit the most relevant memories into the requested token ceiling. The store can hold ten million memories; the context window never sees more than the budget.

| # | Layer | Backend | Mechanism |
|---|---|---|---|
| 01 | **Tag Match** | SQLite | Tag index, FTS, exact + partial hits |
| 02 | **Graph Expansion** | NetworkX / AGE | Multi&#8209;hop BFS (depth 2) |
| 03 | **Vector Search** | ChromaDB / pgvector | Cosine similarity |
| 04 | **Fusion + Rank** | In&#8209;process | RRF (k=60) + PageRank + confidence decay |
| 05 | **Diversity + Fit** | In&#8209;process | MMR (&lambda;=0.7) + greedy token&#8209;budget pack |

### Storage roles

<p align="center"><img src="docs/storage-roles.svg" alt="Storage roles — document store, vector store, graph store" width="100%"></p>

Every memory is persisted across **three complementary stores**. Every supported backend combination is just a different technology choice for one or more of these roles.

| Role | What it stores | Why it exists |
|---|---|---|
| **Document store** | The source of truth &mdash; content, tags, entity, category, timestamps, provenance, confidence | Where `add()` commits; where `recall()` hydrates final memory text |
| **Vector store** | Dense embedding per memory, keyed by memory ID | Finds memories by *meaning* when no tag or word overlaps the query |
| **Graph store** | Entity nodes + typed edges (`uses`, `authored-by`, `supersedes`) | Connects memories indirectly &mdash; query &ldquo;Python&rdquo; can surface &ldquo;Django&rdquo; via the graph |

### Ingestion flow &mdash; what happens on `add()`

<sub>Example framing &mdash; a <b>market intelligence system</b> feeding a financial advisor pipeline. Every signal the desk cares about lands here.</sub>

```mermaid
flowchart TB
    subgraph SOURCES [<b>&sect; MARKET INTELLIGENCE SOURCES</b>]
        direction LR
        N1[<b>News wires</b><br/><sub>Reuters &bull; Bloomberg<br/><i>breaking headlines</i></sub>]
        N2[<b>Market data</b><br/><sub>ticks &bull; OHLC<br/><i>prices &bull; volumes</i></sub>]
        N3[<b>Earnings reports</b><br/><sub>10-K &bull; 10-Q &bull; 8-K<br/><i>guidance &bull; surprises</i></sub>]
        N4[<b>Leadership changes</b><br/><sub>CEO / CFO / Board<br/><i>appointments &bull; exits</i></sub>]
        N5[<b>Geopolitical events</b><br/><sub>tariffs &bull; sanctions<br/><i>policy &bull; conflict</i></sub>]
        N1 ~~~ N2 ~~~ N3 ~~~ N4 ~~~ N5
    end

    SOURCES ==>|<b>mem.add&#40;content, tags, entity, category, provenance, ts&#41;</b>| API{{<b>INGEST API</b>}}

    subgraph WRITE [<b>&sect; PARALLEL WRITES</b> &mdash; one logical transaction]
        direction LR
        D1[(<b>Document store</b><br/><sub>insert row<br/>content &bull; tags &bull; entity<br/>ts &bull; source &bull; confidence</sub>)]
        D2[(<b>Vector store</b><br/><sub>embed text<br/>&rarr; 384-d vector<br/>keyed by memory ID</sub>)]
        D3[(<b>Graph store</b><br/><sub>extract entities + edges<br/>&#40;issuer, sector, person,<br/>country, event&#41;</sub>)]
    end

    API --> D1
    API --> D2
    API --> D3

    D1 ==> CD
    D2 ==> CD
    D3 ==> CD

    CD{<b>Contradiction check</b><br/><sub>per entity &bull; per field<br/>e.g. &ldquo;JPM CFO is X&rdquo; vs new &ldquo;JPM CFO is Y&rdquo;</sub>}

    CD ==>|newer fact wins| SUP[<b>Supersede older fact</b><br/><sub>keep in timeline for audit</sub>]
    SUP ==> DONE[[<b>Committed &bull; recallable</b>]]

    style SOURCES fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style WRITE   fill:#FBF8F1,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style API     fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style CD      fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style SUP     fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style DONE    fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style N1      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style N2      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style N3      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style N4      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style N5      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
```

<sub>The three writes commit as one logical transaction. On SQL backends it&rsquo;s a real DB transaction; on distributed backends it&rsquo;s sequenced with best-effort rollback. Contradictions don&rsquo;t overwrite &mdash; older facts are <b>superseded</b> and retained in the timeline so auditors can reconstruct what the desk knew, and when.</sub>

### Recall flow &mdash; what happens on `recall()`

<sub>Same market intelligence system &mdash; now the <b>Portfolio Planner</b> asks a real question ahead of the morning call.</sub>

```mermaid
flowchart TB
    subgraph CHAT [<b>&sect; ADVISOR CHAT INTERFACE</b> &mdash; human in the loop]
        direction LR
        U[/<b>Financial Advisor</b><br/><sub>typing into chat UI<br/>ahead of the 8am call</sub>/]
        BUBBLE[<b>&ldquo;What do we know about JPM&rsquo;s CFO transition<br/>and the fallout for US regional banks?&rdquo;</b>]
        U ==> BUBBLE
    end

    CHAT ==>|routed to| AGENT([<b>Portfolio Planner agent</b><br/><sub>decomposes intent &bull; issues recall</sub>])

    subgraph QUERIES [<b>&sect; DECOMPOSED RECALL QUERIES</b> &mdash; what the agent actually asks memwright]
        direction LR
        Q1[<i>&ldquo;JPM CFO transition&rdquo;</i>]
        Q2[<i>&ldquo;Semiconductor supply-chain<br/>risk after latest tariff move&rdquo;</i>]
        Q3[<i>&ldquo;Earnings surprises in<br/>US regional banks, last 90 days&rdquo;</i>]
        Q1 ~~~ Q2 ~~~ Q3
    end

    AGENT ==> QUERIES
    QUERIES ==>|<b>mem.recall&#40;query, budget=2000&#41;</b>| API{{<b>RECALL API</b>}}

    subgraph SOURCES [<b>&sect; STAGE A</b> &mdash; parallel sources &bull; fan-out across 3 indexes]
        direction LR
        L1[<b>01 &bull; Tag Match</b><br/><sub>&rarr; document store<br/>FTS on <code>JPM</code>, <code>CFO</code>,<br/><code>tariff</code>, <code>earnings</code></sub>]
        L2[<b>02 &bull; Graph Expansion</b><br/><sub>&rarr; graph store<br/>BFS from <code>JPM</code> &rarr;<br/><code>CFO</code> &rarr; <code>Jeremy Barnum</code></sub>]
        L3[<b>03 &bull; Vector Search</b><br/><sub>&rarr; vector store<br/>cosine on query<br/>top-K nearest embeddings</sub>]
    end

    API --> L1
    API --> L2
    API --> L3

    L1 --> IDS[(Candidate memory IDs<br/><sub>~100s &bull; deduped</sub>)]
    L2 --> IDS
    L3 --> IDS

    IDS ==>|hydrate from doc store| L4[<b>04 &bull; Fusion &amp; Rank</b><br/><sub>RRF k=60 &nbsp;&bull;&nbsp; PageRank boost on central entities<br/>&bull; confidence decay on stale prints</sub>]
    L4 ==> L5[<b>05 &bull; Diversity &amp; Fit</b><br/><sub>MMR &lambda;=0.7 &mdash; drop near-duplicate news wires<br/>&bull; greedy pack under 2,000 tokens</sub>]
    L5 ==>|<b>ranked memories &le; budget</b><br/>zero LLM calls in the path| OUT[[<b>Portfolio Planner context</b>]]
    OUT ==>|grounded answer<br/>streamed to chat| REPLY[/<b>Chat reply to advisor</b><br/><sub>sourced &bull; dated &bull; auditable</sub>/]

    style CHAT    fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style U       fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style BUBBLE  fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style AGENT   fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style REPLY   fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style QUERIES fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style SOURCES fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style API     fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style L1      fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style L2      fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style L3      fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style IDS     fill:#F5F1E8,stroke:#1A1614,color:#1A1614
    style L4      fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style L5      fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style OUT     fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style Q1      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style Q2      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style Q3      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
```

<sub>Only memory IDs travel between layers until the hydrate step. A store with ten million market-intel rows still returns a tight result set inside the caller&rsquo;s token ceiling. Graph expansion is the step that lets <i>&ldquo;tariff&rdquo;</i> surface memories about <i>&ldquo;TSMC&rdquo;</i> and <i>&ldquo;Nvidia&rdquo;</i> without either word appearing in the query.</sub>

---

<a id="deploy"></a>

## &sect; 04 &mdash; Deployment Matrix

<sub><b>Your cloud &middot; your infrastructure &middot; Terraform included</b></sub>

Same API. Every backend. Your infrastructure, not theirs.

Memwright ships as a Python library, a REST API, or a containerized service. Deploy to AWS App Runner, GCP Cloud Run, or Azure Container Apps with a single command. Terraform templates included. No SaaS middleman, no per&#8209;seat fees, no vendor lock&#8209;in.

```bash
$ pip install memwright
$ memwright api --host 0.0.0.0 --port 8080
```

<sub>Starlette ASGI on <code>http://localhost:8080</code>. SQLite + ChromaDB + NetworkX provision automatically under <code>~/.memwright</code>. Point every agent in your stack at the same URL &mdash; they share memory instantly. No Docker. No API keys. Air&#8209;gap it behind your firewall and walk away.</sub>

| # | Target | Notes |
|---|---|---|
| 01 | **AWS** &mdash; App Runner | Starlette ASGI. Auto&#8209;scaling, HTTPS, custom domains. 2 CPU &middot; 4 GB &middot; us&#8209;west&#8209;2 |
| 02 | **Azure** &mdash; Container Apps | Cosmos DB DiskANN. Scale&#8209;to&#8209;zero. Same API, same results. 2 CPU &middot; 4 GB &middot; eastus |
| 03 | **GCP** &mdash; Cloud Run | AlloyDB. Scale&#8209;to&#8209;zero. Google's managed infrastructure. 2 CPU &middot; 4 GB &middot; us&#8209;central1 |
| 04 | **PostgreSQL** backend | pgvector + Apache AGE. Neon serverless or any Postgres 16. Doc &middot; Vector &middot; Graph |
| 05 | **ArangoDB** backend | Multi&#8209;model: graph + document + vector in one engine. Oasis or self&#8209;hosted. |
| 06 | **Local / On&#8209;Prem** | SQLite + ChromaDB + NetworkX. Air&#8209;gapped deployments. No network egress. |

### Same container, pluggable stores

The Memwright Docker image is identical across every deployment. Only the three storage roles swap out:

```mermaid
flowchart LR
    subgraph Laptop["Local · laptop"]
        L[Memwright process]
        L --> LD[(SQLite<br/>doc)]
        L --> LV[(ChromaDB<br/>vector)]
        L --> LG[(NetworkX<br/>graph)]
    end
    subgraph Self["Self-host · Docker · Postgres"]
        S[Memwright container]
        S --> SD[(Postgres 16<br/>doc)]
        S --> SV[(pgvector<br/>vector)]
        S --> SG[(Apache AGE<br/>graph)]
    end
    subgraph AWS["AWS · App Runner"]
        A[Memwright container]
        A --> AAR[(ArangoDB Oasis<br/>doc · vector · graph)]
    end
    subgraph GCP["GCP · Cloud Run"]
        G[Memwright container]
        G --> GA[(AlloyDB + pgvector + AGE<br/>doc · vector · graph)]
    end
    subgraph Azure["Azure · Container Apps"]
        AZ[Memwright container]
        AZ --> AZC[(Cosmos DB DiskANN<br/>doc · vector · graph)]
    end
```

<sub><i>Every deployment is the same Python library wrapped in the same Starlette ASGI container. <code>DocumentStore</code>, <code>VectorStore</code>, and <code>GraphStore</code> are three interfaces; each row above is one implementation of each.</i></sub>

### Promotion path

```
   laptop              single-VM / dev              managed container
   ──────              ───────────────              ─────────────────
   pip install         docker compose up            App Runner  ·  Cloud Run  ·  Container Apps
        │                     │                             │
        ▼                     ▼                             ▼
   SQLite file          SQLite on volume              Postgres / ArangoDB / Cosmos
   ChromaDB dir         ChromaDB on volume            managed vector index
   NetworkX JSON        NetworkX JSON on volume       managed graph
        │                     │                             │
        └─────────────────────┴─────────────────────────────┘
                    same API · same container image
               only storage config + credentials change
```

<sub><i>Prototype on a laptop. Promote to Docker Compose on a VM without rewriting a single line. Promote to managed container runtime by swapping the storage URLs. The code never learns which backend it&rsquo;s talking to.</i></sub>

---

<a id="principles"></a>

## &sect; 05 &mdash; Principles

<sub><b>What we won't compromise on</b></sub>

| # | Principle | What it means |
|---|---|---|
| 01 | **Self&#8209;hosted by default** | Your data stays in your infrastructure. No SaaS middleman, no per&#8209;seat fees, no lock&#8209;in. Run on a laptop, a VM, or any cloud. |
| 02 | **Deterministic retrieval** | Tag match, graph traversal, vector search, RRF fusion, MMR diversity &mdash; all deterministic. No LLM judges. No hidden inference calls in the critical path. |
| 03 | **One API, every backend** | Same `mem.recall()` call whether the store is SQLite on a laptop or ArangoDB behind a Cloud Run service. Swap backends without rewriting agents. |
| 04 | **Agent teams are first&#8209;class** | Namespaces, roles, quotas, and provenance are not bolt&#8209;ons. The primitives were designed for orchestrator&ndash;worker pipelines from day one. |
| 05 | **Boring where it counts** | Postgres, SQLite, ChromaDB, NetworkX. Proven, debuggable, no magic. Terraform templates, not a hosted console. |

---

<p align="center"><b>Install <code>memwright</code> and point your agents at one URL. They share memory instantly.</b></p>

<p align="center"><sub><a href="#reference">&darr; Reference documentation follows</a></sub></p>

---

<a id="reference"></a>

# Reference

<sub><i>Everything below is the technical manual. If you are evaluating, the pitch ended at &sect; 05.</i></sub>

## Table of Contents

- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [How It Works](#how-it-works)
- [Python API](#python-api)
- [REST API](#rest-api)
- [MCP Integration](#mcp-integration)
- [Cloud Backends](#cloud-backends)
- [Cloud Deployment](#cloud-deployment)
- [Embedding Providers](#embedding-providers)
- [CLI Reference](#cli-reference)
- [Configuration](#configuration)
- [Testing](#testing)
- [Compatibility](#compatibility)
- [Uninstall](#uninstall)

---

## Quick Start

```bash
poetry add memwright
```

```python
from agent_memory import AgentMemory

mem = AgentMemory("./store")
mem.add("Architecture decision: event sourcing for order service",
        category="technical", entity="order-service", tags=["arch", "decision"])
results = mem.recall("how is the order service structured?", budget=2000)
```

For REST API self-host, MCP integration, and cloud deploy — see [REST API](#rest-api), [MCP Integration](#mcp-integration), [Cloud Deployment](#cloud-deployment). `memwright doctor ~/.memwright` verifies all four components (Document Store, Vector Store, Graph Store, Retrieval Pipeline).

---

## Architecture

<p align="center">
  <img src="https://raw.githubusercontent.com/bolnet/agent-memory/main/docs/architecture.svg" alt="Memwright Architecture" width="100%">
</p>

### Component Overview

```
agent_memory/
├── core.py                    # AgentMemory — main orchestrator
├── models.py                  # Memory + RetrievalResult dataclasses
├── context.py                 # AgentContext — multi-agent provenance & RBAC
├── client.py                  # MemoryClient — HTTP client for distributed mode
├── cli.py                     # CLI entry point (19 commands)
├── api.py                     # Starlette ASGI REST API (8 routes)
├── store/
│   ├── base.py                # Abstract interfaces: DocumentStore, VectorStore, GraphStore
│   ├── sqlite_store.py        # SQLite storage (WAL, 17 columns, 8 indexes)
│   ├── chroma_store.py        # ChromaDB vector search (local sentence-transformers)
│   ├── schema.sql             # SQLite schema definition
│   ├── postgres_backend.py    # PostgreSQL (pgvector + Apache AGE)
│   ├── arango_backend.py      # ArangoDB (native doc + vector + graph)
│   ├── aws_backend.py         # AWS (DynamoDB + OpenSearch + Neptune)
│   └── azure_backend.py       # Azure (Cosmos DB DiskANN + NetworkX)
├── graph/
│   ├── networkx_graph.py      # NetworkX MultiDiGraph with PageRank + BFS
│   └── extractor.py           # Entity/relation extraction (50+ known tools)
├── retrieval/
│   ├── orchestrator.py        # 3-layer cascade with RRF fusion
│   ├── tag_matcher.py         # Stop-word filtered tag extraction
│   └── scorer.py              # Temporal, entity, PageRank, MMR, confidence decay
├── temporal/
│   └── manager.py             # Contradiction detection + supersession
├── extraction/
│   └── extractor.py           # Rule-based + LLM memory extraction
├── mcp/
│   └── server.py              # MCP server (8 tools, 2 resources, 2 prompts)
├── hooks/
│   ├── session_start.py       # Context injection (20K token budget)
│   ├── post_tool_use.py       # Auto-capture from Write/Edit/Bash
│   └── stop.py                # Session summary generation
├── utils/
│   └── config.py              # MemoryConfig dataclass + load/save
└── infra/                     # Terraform + Docker for cloud deployments
    ├── apprunner/             # AWS App Runner
    ├── cloudrun/              # GCP Cloud Run
    └── containerapp/          # Azure Container Apps
```

### Three Storage Roles

Every backend implements one or more of these roles:

| Role | Purpose | Local Default | Cloud Options |
|------|---------|--------------|---------------|
| **Document** | Core storage, CRUD, filtering | SQLite | PostgreSQL, ArangoDB, DynamoDB, Cosmos DB |
| **Vector** | Semantic similarity search | ChromaDB | pgvector, ArangoDB, OpenSearch, Cosmos DiskANN |
| **Graph** | Entity relationships, BFS traversal | NetworkX | Apache AGE, ArangoDB, Neptune |

Cloud backends fill all 3 roles in a single service. If any optional component fails, the system degrades gracefully to document-only.

---

## How It Works

### Memory is infrastructure, not a prompt attachment

Memwright runs as a separate tier — a library, a container, or a cloud service — that agents query on demand. Stored memories never enter the context window until an agent explicitly calls `recall()` with a token budget. Retrieval cost stays constant as the store grows from 100 to 5,000,000 memories; only the ranking candidate pool expands.

### Token cost is bounded by budget, not store size

```
Naive context-injection approach:
  Month 1:   2K tokens loaded every message
  Month 6:  15K tokens loaded every message  ← context crowded

Memwright:
  Month 1:   ≤2K tokens returned per recall  (ranked from 100 memories)
  Month 6:   ≤2K tokens returned per recall  (ranked from 5,000 memories)
                                             ← bounded cost, deeper recall
```

### How a recall works

When an agent calls `memory_recall("deployment setup", budget=2000)`:

```
Store: 5,000 memories

  Tag search finds:     15 memories tagged "deployment"
  Graph search finds:    8 memories linked to "AWS", "Docker" entities
  Vector search finds:  20 semantically similar memories

  After dedup + RRF fusion:  30 unique candidates, scored and ranked

  Budget fitting (2,000 tokens):
    Memory A (score 0.95):  500 tokens → in   (total: 500)
    Memory B (score 0.90):  600 tokens → in   (total: 1,100)
    Memory C (score 0.88):  400 tokens → in   (total: 1,500)
    Memory D (score 0.85):  300 tokens → in   (total: 1,800)
    Memory E (score 0.80):  400 tokens → SKIP (exceeds 2,000)

  Result: 4 memories, 1,800 tokens. 4,996 memories never entered context.
```

---

## MCP Integration

Memwright ships an MCP server so any MCP-compatible client (Claude Code, Cursor, Windsurf, custom agents) can store and retrieve memories. Start it with `memwright mcp`.

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| `memory_add` | Store a fact | `content`, `tags[]`, `category`, `entity`, `namespace`, `event_date`, `confidence` |
| `memory_recall` | Smart multi-layer retrieval | `query`, `budget` (default: 2000), `namespace` |
| `memory_search` | Filter with date ranges | `query`, `category`, `entity`, `namespace`, `status`, `after`, `before`, `limit` |
| `memory_get` | Fetch by ID | `memory_id` |
| `memory_forget` | Archive (soft delete) | `memory_id` |
| `memory_timeline` | Chronological entity history | `entity`, `namespace` |
| `memory_stats` | Store size, counts | — |
| `memory_health` | Health check (call first!) | — |

### Categories

`core_belief` · `preference` · `career` · `project` · `technical` · `personal` · `location` · `relationship` · `event` · `session` · `general`

### MCP Resources

- **`memwright://entity/{name}`** — Entity details + related entities from graph
- **`memwright://memory/{id}`** — Full memory object

### MCP Prompts

- **`recall`** — Search memories for relevant context
- **`timeline`** — Chronological history of an entity

---

## Retrieval Pipeline

The retrieval system uses a 5-layer cascade with multi-signal fusion:

```
Query: "deployment setup"
  │
  ├─ Layer 0: Graph Expansion
  │  Extract entities from query → BFS traversal (depth=2)
  │  "deployment" → finds "AWS", "Docker", "Terraform" connections
  │
  ├─ Layer 1: Tag Match (SQLite)
  │  extract_tags(query) → tag_search() → score 1.0
  │
  ├─ Layer 2: Entity-Field Search
  │  Memories about graph-connected entities → score 0.5
  │
  ├─ Layer 3: Vector Search (ChromaDB)
  │  Semantic similarity → score = 1 - cosine_distance
  │
  ├─ Layer 4: Graph Relation Triples
  │  Inject relationship context → score 0.6
  │
  ▼ FUSION
  ├─ Reciprocal Rank Fusion (RRF, k=60)
  │  score = Σ 1/(k + rank_in_source)
  │  OR Graph Blend: 0.7 * norm_vector + 0.3 * norm_pagerank
  │
  ▼ SCORING
  ├─ Temporal Boost: +0.2 * max(0, 1 - age_days/90)
  ├─ Entity Boost:   +0.30 exact match, +0.15 substring
  ├─ PageRank Boost:  +0.3 * entity_pagerank_score
  │
  ▼ DIVERSITY
  ├─ MMR Rerank: λ*relevance - (1-λ)*max_jaccard_similarity (λ=0.7)
  │
  ▼ CONFIDENCE
  ├─ Time Decay:    -0.001 per hour since last access
  ├─ Access Boost:  +0.03 per access_count
  ├─ Clamp:         [0.1, 1.0]
  │
  ▼ BUDGET
  └─ Greedy selection by score until token budget filled
```

Querying "Python" also finds memories about "FastAPI" if they're connected in the entity graph. Multi-hop reasoning through relationship traversal.

---

## Python API

### Basic Usage

```python
from agent_memory import AgentMemory

mem = AgentMemory("./my-agent")  # auto-provisions all backends

# Store
mem.add("User prefers Python over Java",
        tags=["preference", "coding"],
        category="preference",
        entity="Python")

# Recall with token budget
results = mem.recall("what language?", budget=2000)

# Formatted context for prompt injection
context = mem.recall_as_context("user background", budget=4000)

# Search with filters
memories = mem.search(category="project", entity="Python", limit=10)

# Timeline
history = mem.timeline("Python")

# Contradiction handling — automatic
mem.add("User works at Google", tags=["career"], category="career", entity="Google")
mem.add("User works at Meta", tags=["career"], category="career", entity="Meta")
# ^ Google memory auto-superseded

# Namespace isolation
mem.add("Team standup at 9am", namespace="team:alpha")
results = mem.recall("standup time", namespace="team:alpha")

# Maintenance
mem.forget(memory_id)             # Archive
mem.forget_before("2025-01-01")   # Archive old memories
mem.compact()                     # Permanently delete archived
mem.export_json("backup.json")    # Export
mem.import_json("backup.json")    # Import (dedup by content hash)

# Health & stats
mem.health()  # → {sqlite: ok, chroma: ok, networkx: ok, retrieval: ok}
mem.stats()   # → {total: 500, active: 480, ...}

# Context manager
with AgentMemory("./store") as mem:
    mem.add("auto-closed on exit")
```

### Memory Object

```python
@dataclass
class Memory:
    id: str                    # UUID
    content: str               # The actual fact/observation
    tags: List[str]            # Searchable tags
    category: str              # Classification (preference, career, project, ...)
    entity: str                # Primary entity (company, tool, person)
    namespace: str             # Isolation key (default: "default")
    created_at: str            # ISO timestamp
    event_date: str            # When the fact occurred
    valid_from: str            # Temporal validity start
    valid_until: str           # Set when superseded
    superseded_by: str         # ID of replacement memory
    confidence: float          # 0.0-1.0
    status: str                # active | superseded | archived
    access_count: int          # Times recalled
    last_accessed: str         # Last recall timestamp
    content_hash: str          # SHA-256 for dedup
    metadata: Dict[str, Any]   # Arbitrary JSON
```

---

## Multi-Agent Systems

Memwright is built for production multi-agent pipelines — orchestrator-worker, planner-executor, researcher-reviewer, and hierarchical swarms. Every recall and write is scoped to an `AgentContext` that carries identity, role, namespace, parent trail, token budget, write quota, and visibility policy. Contexts are immutable; spawning a sub-agent returns a new context with inherited provenance.

```python
from agent_memory.context import AgentContext, AgentRole, Visibility

# Create a root context
ctx = AgentContext.from_env(
    agent_id="orchestrator",
    namespace="project:acme",
    role=AgentRole.ORCHESTRATOR,
    token_budget=20000,
)

# Spawn child contexts for sub-agents (immutable — returns new instance)
planner = ctx.as_agent("planner", role=AgentRole.PLANNER, token_budget=5000)
researcher = ctx.as_agent("researcher", role=AgentRole.RESEARCHER, read_only=True)

# Provenance tracking — metadata auto-enriched
planner.add_memory("Architecture decision: use event sourcing",
                   category="technical", visibility=Visibility.TEAM)
# metadata includes: _agent_id, _session_id, _namespace, _visibility, _role

# Recall is scoped to namespace + cached within session
results = researcher.recall("architecture decisions")

# Token budget tracked
print(researcher.token_budget - researcher.token_budget_used)

# Governance
researcher.flag_for_review("Need human approval for deployment plan")
researcher.add_compliance_tag("SOC2")

# Session introspection
summary = ctx.session_summary()
# → {agent_trail, memories_written, memories_recalled, token_usage, review_flags}
```

### AgentContext Features

| Feature | Description |
|---------|-------------|
| **Namespace isolation** | Each agent/project gets isolated memory partition |
| **RBAC roles** | ORCHESTRATOR, PLANNER, EXECUTOR, RESEARCHER, REVIEWER, MONITOR |
| **Read-only mode** | Agents can recall but not write |
| **Write quotas** | `max_writes_per_agent` (default: 100) |
| **Token budgets** | Per-agent budget tracking |
| **Recall cache** | Dedup redundant queries within a session |
| **Scratchpad** | Inter-agent data passing |
| **Provenance** | Agent trail, parent tracking, visibility levels |
| **Compliance** | Review flags, compliance tags for audit |
| **Distributed mode** | Set `memory_url` to use HTTP client instead of local |

---

## Cloud Backends

Each cloud backend fills all three roles (document, vector, graph) in a single service:

### PostgreSQL (Neon, Cloud SQL, self-hosted)

Uses pgvector for vectors, Apache AGE for graph. AGE is optional — without it, graph gracefully degrades.

```python
mem = AgentMemory("./store", config={
    "backends": ["postgres"],
    "postgres": {"url": "postgresql://user:pass@host:5432/memwright"}
})
```

### ArangoDB (ArangoGraph Cloud, Docker)

Native document, vector, and graph support in one database.

```python
mem = AgentMemory("./store", config={
    "backends": ["arangodb"],
    "arangodb": {"url": "https://instance.arangodb.cloud:8529", "database": "memwright"}
})
```

### Azure (Cosmos DB)

Cosmos DB with DiskANN vector indexing. Graph via NetworkX persisted to Cosmos containers.

```python
mem = AgentMemory("./store", config={
    "backends": ["azure"],
    "azure": {"cosmos_endpoint": "https://account.documents.azure.com:443/"}
})
```

### GCP (AlloyDB)

Extends PostgreSQL backend with AlloyDB Connector (IAM auth) and Vertex AI embeddings (768D).

```python
mem = AgentMemory("./store", config={
    "backends": ["gcp"],
    "gcp": {"project_id": "my-project", "cluster": "memwright", "instance": "primary"}
})
```

### Installing cloud extras

```bash
poetry add "memwright[postgres]"    # PostgreSQL
poetry add "memwright[arangodb]"    # ArangoDB
poetry add "memwright[aws]"         # AWS (DynamoDB + OpenSearch + Neptune)
poetry add "memwright[azure]"       # Azure Cosmos DB
poetry add "memwright[gcp]"         # GCP AlloyDB + Vertex AI
poetry add "memwright[all]"         # Everything
```

---

## Cloud Deployment

Deploy Memwright as an HTTP API on any cloud with a single command:

```bash
./scripts/deploy.sh aws        # App Runner (2 CPU / 4GB, auto-scale)
./scripts/deploy.sh gcp        # Cloud Run (auto-scale 0–3, 2 CPU / 4GB)
./scripts/deploy.sh azure      # Container Apps (scale-to-zero, 2 CPU / 4GB)

./scripts/deploy.sh aws --teardown   # Destroy everything
```

**Prerequisites**: Docker, Terraform, cloud CLI (`aws`/`gcloud`/`az`), backend credentials in `.env`.

| Cloud | Infrastructure | Terraform |
|-------|---------------|-----------|
| AWS | ECR + App Runner (2 CPU, 4GB) | `agent_memory/infra/apprunner/main.tf` |
| GCP | Artifact Registry + Cloud Run (2 CPU, 4GB) | `agent_memory/infra/cloudrun/main.tf` |
| Azure | ACR + Log Analytics + Container Apps (2 CPU, 4GB) | `agent_memory/infra/containerapp/main.tf` |

### REST API Endpoints

All deployments expose the same Starlette ASGI API:

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Component health check |
| `GET` | `/stats` | Store statistics |
| `POST` | `/add` | Add a memory |
| `POST` | `/recall` | Smart retrieval with budget |
| `POST` | `/search` | Filtered search |
| `POST` | `/timeline` | Entity chronological history |
| `POST` | `/forget` | Archive a memory |
| `GET` | `/memory/{id}` | Get memory by ID |

Response envelope: `{"ok": true, "data": {...}}` or `{"ok": false, "error": "message"}`

---

## Embedding Providers

Memwright auto-detects the best available embedding provider:

| Priority | Provider | Model | Dimensions | Trigger |
|----------|----------|-------|------------|---------|
| 1 | Cloud-native | Bedrock Titan / Azure OpenAI / Vertex AI | 768-1536 | Cloud backend configured |
| 2 | OpenAI / OpenRouter | text-embedding-3-small | 1536 | `OPENAI_API_KEY` or `OPENROUTER_API_KEY` set |
| 3 | Local (default) | all-MiniLM-L6-v2 | 384 | Always available, no API key |

The local fallback downloads ~90MB on first use. All providers implement the same interface — switching is transparent.

---

## CLI Reference

Both `memwright` and `agent-memory` work as entry points:

### MCP Server

```bash
memwright mcp                          # Start MCP server (uses ~/.memwright)
memwright mcp --path /custom/path      # Custom store location
```

### Memory Operations

```bash
agent-memory add ./store "User prefers Python" --tags "pref,coding" --category preference
agent-memory recall ./store "what language?" --budget 4000
agent-memory search ./store --category project --entity Python --limit 20
agent-memory list ./store --status active --category technical
agent-memory timeline ./store --entity Python
agent-memory get ./store <memory-id>
agent-memory forget ./store <memory-id>
```

### Maintenance

```bash
agent-memory doctor ~/.memwright       # Health check (SQLite, ChromaDB, NetworkX, Retrieval)
agent-memory stats ./store             # Memory counts, DB size, breakdowns
agent-memory export ./store -o backup.json
agent-memory import ./store backup.json
agent-memory compact ./store           # Permanently delete archived memories
agent-memory inspect ./store           # Raw DB inspection
```

### Lifecycle Hooks

```bash
memwright hook session-start           # Inject context at agent session start
memwright hook post-tool-use           # Auto-capture tool observations
memwright hook stop                    # Generate session summary on exit
```

Hooks integrate with any harness that supports session lifecycle callbacks.

---

## Configuration

### Store location

Default: `~/.memwright/`. Configurable with `--path` on any CLI command.

```
~/.memwright/
├── memory.db        # SQLite database (core storage)
├── config.json      # Retrieval tuning parameters
├── graph.json       # NetworkX entity graph
└── chroma/          # ChromaDB vector store + embeddings
```

### config.json

All fields optional. Defaults apply if the file doesn't exist:

```json
{
  "default_token_budget": 2000,
  "min_results": 3,
  "backends": ["sqlite", "chroma", "networkx"],
  "enable_mmr": true,
  "mmr_lambda": 0.7,
  "fusion_mode": "rrf",
  "confidence_gate": 0.0,
  "confidence_decay_rate": 0.001,
  "confidence_boost_rate": 0.03
}
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `default_token_budget` | 2000 | Max tokens returned per recall |
| `min_results` | 3 | Minimum results to return |
| `enable_mmr` | true | Maximal Marginal Relevance diversity reranking |
| `mmr_lambda` | 0.7 | Relevance vs diversity balance (0=diverse, 1=relevant) |
| `fusion_mode` | "rrf" | "rrf" (parameter-free) or "graph_blend" (weighted) |
| `confidence_decay_rate` | 0.001 | Score penalty per hour since last access |
| `confidence_boost_rate` | 0.03 | Score boost per access count |
| `confidence_gate` | 0.0 | Minimum confidence threshold to include in results |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `MEMWRIGHT_PATH` | Default store path |
| `MEMWRIGHT_URL` | Remote API URL (distributed mode) |
| `MEMWRIGHT_NAMESPACE` | Default namespace |
| `MEMWRIGHT_TOKEN_BUDGET` | Default token budget |
| `MEMWRIGHT_SESSION_ID` | Session ID for provenance tracking |

---

## Testing

### Running Tests

```bash
# All unit tests — no Docker, no API keys
poetry run pytest tests/ -v

# With coverage
poetry run pytest tests/ -v --cov=agent_memory --cov-report=term-missing

# Live integration tests (need credentials)
NEON_DATABASE_URL='postgresql://...' poetry run pytest tests/test_postgres_live.py -v
AZURE_COSMOS_ENDPOINT='https://...' poetry run pytest tests/test_azure_live.py -v
```

### Test Coverage

- **607 unit tests** covering all backends, retrieval, config, embeddings, and CLI
- **14 live integration tests** per cloud backend (Neon, Azure, ArangoDB)
- **Mock tests** for every cloud backend — no cloud account needed
- All unit tests run without Docker or API keys

---

## Compatibility

### MCP Clients

| Client | Config File |
|--------|-------------|
| Any MCP client | Standard MCP stdio transport |
| Claude Code | `.mcp.json` (project) or `~/.claude/.mcp.json` (global) |
| Cursor | `.cursor/mcp.json` |
| Windsurf | MCP config in settings |

Same `memwright mcp` command for every client.

### Python

- Python 3.10, 3.11, 3.12, 3.13, 3.14

---

## Uninstall

### 1. Remove MCP server config (if used)

Delete the `memory` entry from your MCP client's config file.

### 2. Uninstall the package

```bash
poetry remove memwright
```

### 3. Delete stored memories (optional)

```bash
# Export first if you want a backup
agent-memory export ~/.memwright -o memwright-backup.json

# Then delete
rm -rf ~/.memwright
```

---

## License

MIT

---

<sub>mcp-name: io.github.bolnet/memwright</sub>
