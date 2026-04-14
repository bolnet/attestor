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

<p align="center">
  <img src="docs/constellation.svg" alt="Six stars out of eight hundred light up — Memwright recalls exactly what this moment needs" width="100%">
</p>

<p align="center"><sub><b><i>Memwright doesn&rsquo;t search. It remembers.</i></b></sub></p>

---

### The problem

**Agent prototypes don&rsquo;t survive production. Memory is why.** Single agents rediscover the same facts every run. Multi&#8209;agent pipelines are worse &mdash; the planner&rsquo;s decisions never reach the executor, the researcher&rsquo;s findings never reach the reviewer. Teams paper over it by stuffing giant prompts between agents, burning tokens on stale context. That&rsquo;s a workaround, not an architecture.

### The solution &mdash; at a glance

<p align="center">
  <img src="docs/timeline.svg" alt="Write · Write · Write · Write ... Read — memories accumulate across sessions, load_context primes the next task" width="100%">
</p>

<sub><b>Memory accumulates. Load primes.</b> &nbsp;Four writes across Mon&ndash;Thu land as persisted memories. Friday morning a fresh Portfolio Planner wakes up to a new task, calls <code>mem.load_context()</code>, and Memwright ranks + dedupes + budget&#8209;fits all four back into the context window. The agent resumes with full continuity &mdash; earnings signal, risk cap, prior stance, compliance precedent &mdash; <b>not RAG over documents, but the agents&rsquo; own history replayed into a fresh context</b>. Zero LLM calls in the critical path.</sub>

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
    N1["<b>News wires</b><br/>Reuters &bull; Bloomberg<br/><i>breaking headlines</i>"]
    N2["<b>Market data</b><br/>ticks &bull; OHLC<br/><i>prices &bull; volumes</i>"]
    N3["<b>Earnings reports</b><br/>10-K &bull; 10-Q &bull; 8-K<br/><i>guidance &bull; surprises</i>"]
    N4["<b>Leadership changes</b><br/>CEO &bull; CFO &bull; Board<br/><i>appointments &bull; exits</i>"]
    N5["<b>Geopolitical events</b><br/>tariffs &bull; sanctions<br/><i>policy &bull; conflict</i>"]

    subgraph SOURCES ["&sect; MARKET INTELLIGENCE SOURCES"]
        direction LR
        N1 ~~~ N2 ~~~ N3 ~~~ N4 ~~~ N5
    end

    SOURCES ==>|"<b>mem.add(content, tags, entity, provenance, ts)</b>"| API{{"<b>INGEST API</b>"}}

    D1["<b>Document store</b><br/>insert row<br/>content &bull; tags &bull; entity<br/>ts &bull; source &bull; confidence"]
    D2["<b>Vector store</b><br/>embed text<br/>&rarr; 384-d vector<br/>keyed by memory ID"]
    D3["<b>Graph store</b><br/>extract entities + edges<br/>issuer &bull; sector &bull; person<br/>country &bull; event"]

    subgraph WRITE ["&sect; PARALLEL WRITES &mdash; one logical transaction"]
        direction LR
        D1 ~~~ D2 ~~~ D3
    end

    API --> D1
    API --> D2
    API --> D3

    CD["<b>Contradiction check</b><br/>per entity &bull; per field<br/>e.g. JPM CFO is X vs new JPM CFO is Y"]

    D1 ==> CD
    D2 ==> CD
    D3 ==> CD

    CD ==>|newer fact wins| SUP["<b>Supersede older fact</b><br/>keep in timeline for audit"]
    SUP ==> DONE["<b>Committed &bull; recallable</b>"]

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
    style D1      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style D2      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style D3      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
```

<sub>The three writes commit as one logical transaction. On SQL backends it&rsquo;s a real DB transaction; on distributed backends it&rsquo;s sequenced with best-effort rollback. Contradictions don&rsquo;t overwrite &mdash; older facts are <b>superseded</b> and retained in the timeline so auditors can reconstruct what the desk knew, and when.</sub>

---

### Ingestion flow &mdash; variant B &bull; agent-to-agent conversation capture

<sub>A <b>different kind of ingestion</b>: the memories are not external feeds &mdash; they are <b>the agents&rsquo; own conversation</b> as they debate a trade proposal. Every turn is captured with speaker, timestamp, entity, and a decision marker.</sub>

```mermaid
flowchart TB
    T1["<b>Portfolio Planner</b><br/><i>&ldquo;Proposing to add 2% JPM<br/>ahead of Q3 print&rdquo;</i>"]
    T2["<b>Market Researcher</b><br/><i>&ldquo;Consensus EPS is +4% QoQ,<br/>whisper number suggests beat&rdquo;</i>"]
    T3["<b>Risk Analyst</b><br/><i>&ldquo;Tariff headline risk this week<br/>&mdash; raise stop to 8%&rdquo;</i>"]
    T4["<b>Compliance Reviewer</b><br/><i>&ldquo;Ok with position limit.<br/>Logging rationale.&rdquo;</i>"]
    DEC["<b>Decision reached</b><br/>buy 2% JPM &bull; stop 8% &bull; pre-earnings"]

    subgraph CONV ["&sect; AGENT-TO-AGENT CONVERSATION &mdash; trade proposal thread"]
        direction LR
        T1 --> T2
        T2 --> T3
        T3 --> T4
        T4 --> DEC
    end

    CAP["<b>Turn-level capture</b><br/>speaker &bull; utterance &bull; ts<br/>entity: JPM &bull; topic: position sizing<br/>kind: <i>conversation</i>"]

    CONV ==>|auto-capture hook<br/>every turn| CAP
    CAP ==>|"<b>mem.add(content, speaker, thread_id, kind=&quot;chat&quot;)</b>"| API{{"<b>INGEST API</b>"}}

    DOC[("<b>Document store</b><br/>turn rows, thread_id,<br/>speaker, ts, decision flag")]
    VEC[("<b>Vector store</b><br/>embedding per turn")]
    GR[("<b>Graph store</b><br/>edges: agent &rarr; entity<br/>agent &rarr; decision")]

    API --> DOC
    API --> VEC
    API --> GR

    DONE["<b>Thread memorialised</b><br/>replayable &bull; attributable &bull; auditable"]
    DOC ==> DONE
    VEC ==> DONE
    GR ==> DONE

    style CONV fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style CAP  fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style API  fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style DONE fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style T1   fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style T2   fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style T3   fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style T4   fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style DEC  fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style DOC  fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style VEC  fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style GR   fill:#FBF8F1,stroke:#1A1614,color:#1A1614
```

<sub>Different from feed ingestion: the <b>source is the agents themselves</b>, not the outside world. Every turn is attributed to a speaker, tied to a thread, and flagged if it contained a decision. Nothing is paraphrased &mdash; the verbatim utterance is preserved so the reasoning can be reconstructed under audit.</sub>

### Recall flow &mdash; what happens on `recall()`

<sub>Same market intelligence system &mdash; now the <b>Portfolio Planner</b> asks a real question ahead of the morning call.</sub>

```mermaid
flowchart TB
    U["<b>Financial Advisor</b><br/>typing into chat UI<br/>ahead of the 8am call"]
    BUBBLE["<b>Chat message</b><br/><i>What do we know about JPM&rsquo;s CFO transition<br/>and the fallout for US regional banks?</i>"]

    subgraph CHAT ["&sect; ADVISOR CHAT INTERFACE &mdash; human in the loop"]
        direction LR
        U ==> BUBBLE
    end

    CHAT ==>|routed to| AGENT(["<b>Portfolio Planner agent</b><br/>decomposes intent &bull; issues recall"])

    Q1["<i>JPM CFO transition</i>"]
    Q2["<i>Semiconductor supply-chain<br/>risk after latest tariff move</i>"]
    Q3["<i>Earnings surprises in<br/>US regional banks, last 90 days</i>"]

    subgraph QUERIES ["&sect; DECOMPOSED RECALL QUERIES &mdash; what the agent actually asks memwright"]
        direction LR
        Q1 ~~~ Q2 ~~~ Q3
    end

    AGENT ==> QUERIES
    QUERIES ==>|"<b>mem.recall(query, budget=2000)</b>"| API{{"<b>RECALL API</b>"}}

    L1["<b>01 &bull; Tag Match</b><br/>&rarr; document store<br/>FTS on JPM, CFO,<br/>tariff, earnings"]
    L2["<b>02 &bull; Graph Expansion</b><br/>&rarr; graph store<br/>BFS: JPM &rarr; CFO<br/>&rarr; Jeremy Barnum"]
    L3["<b>03 &bull; Vector Search</b><br/>&rarr; vector store<br/>cosine on query<br/>top-K nearest embeddings"]

    subgraph SOURCES ["&sect; STAGE A &mdash; parallel sources &bull; fan-out across 3 indexes"]
        direction LR
        L1 ~~~ L2 ~~~ L3
    end

    API --> L1
    API --> L2
    API --> L3

    IDS[("Candidate memory IDs<br/>~100s &bull; deduped")]

    L1 --> IDS
    L2 --> IDS
    L3 --> IDS

    L4["<b>04 &bull; Fusion &amp; Rank</b><br/>RRF k=60 &bull; PageRank boost on central entities<br/>&bull; confidence decay on stale prints"]
    L5["<b>05 &bull; Diversity &amp; Fit</b><br/>MMR &lambda;=0.7 &mdash; drop near-duplicate news wires<br/>&bull; greedy pack under 2,000 tokens"]
    OUT["<b>Portfolio Planner context</b>"]
    REPLY["<b>Chat reply to advisor</b><br/>sourced &bull; dated &bull; auditable"]

    IDS ==>|hydrate from doc store| L4
    L4 ==> L5
    L5 ==>|ranked memories &le; budget<br/>zero LLM calls in the path| OUT
    OUT ==>|grounded answer streamed to chat| REPLY

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

### Recall flow &mdash; variant B &bull; agent context recall

<sub>A <b>different kind of recall</b>: no human in the loop. An agent resuming a task &mdash; or handing off to a peer &mdash; pulls back <b>its own prior working context</b>: earlier decisions, peer rationale, what was true at the last checkpoint.</sub>

```mermaid
flowchart TB
    RESUME["<b>Risk Analyst</b> resuming mid-task<br/>or <b>Compliance Reviewer</b> taking handoff<br/><i>no user prompt &mdash; agent self-initiated</i>"]

    INTENT["<b>Context intent</b><br/>&ldquo;what did the desk already decide<br/>about JPM this week, and why?&rdquo;"]

    RESUME ==> INTENT
    INTENT ==>|"<b>mem.recall_context(thread_id, entity=&quot;JPM&quot;, since=7d)</b>"| API{{"<b>CONTEXT RECALL API</b>"}}

    F1["<b>Thread filter</b><br/>same thread_id<br/>or same namespace"]
    F2["<b>Entity filter</b><br/>entity = JPM<br/>and related via graph"]
    F3["<b>Temporal filter</b><br/>within last 7d<br/>supersedes resolved"]
    F4["<b>Speaker filter</b><br/>peer agents in role<br/>RBAC-visible only"]

    subgraph FILTERS ["&sect; CONTEXT FILTERS &mdash; tighter than open-query recall"]
        direction LR
        F1 ~~~ F2 ~~~ F3 ~~~ F4
    end

    API --> FILTERS

    TURNS["<b>Prior conversation turns</b><br/>Planner proposal &bull; Researcher consensus<br/>Analyst stop raise &bull; Compliance sign-off"]
    DECS["<b>Prior decisions</b><br/>buy 2% JPM &bull; stop 8%<br/>decided 3 days ago"]
    DELTA["<b>What changed since</b><br/>new tariff headline today<br/>stop needs re-evaluation"]

    FILTERS ==> TURNS
    FILTERS ==> DECS
    FILTERS ==> DELTA

    PACK["<b>Context pack</b><br/>chronologically ordered<br/>&bull; speaker-attributed<br/>&bull; fit to agent token budget"]

    TURNS ==> PACK
    DECS ==> PACK
    DELTA ==> PACK

    PACK ==>|loaded into agent<br/>working memory| RESUMED["<b>Agent resumes task</b><br/>with full prior context<br/>&bull; no re-asking peers<br/>&bull; no lost decisions"]

    style RESUME   fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style INTENT   fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style API      fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style FILTERS  fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style F1       fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style F2       fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style F3       fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style F4       fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style TURNS    fill:#F5F1E8,stroke:#1A1614,color:#1A1614
    style DECS     fill:#F5F1E8,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style DELTA    fill:#F5F1E8,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style PACK     fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style RESUMED  fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
```

<sub>Differs from open-query recall in three ways: <b>(1)</b> the caller is an <b>agent</b>, not a human &mdash; triggered by resume / handoff, not by a chat message; <b>(2)</b> the filters are <b>tighter</b> &mdash; thread, namespace, RBAC, time window &mdash; not just semantic similarity; <b>(3)</b> the output preserves <b>chronology and attribution</b> rather than ranking purely by relevance. This is how a long-running pipeline stays coherent across restarts, handoffs, and multi-day workflows.</sub>

---

### Isolation &mdash; namespace &amp; RBAC boundary

<sub>Multi-tenant by construction. Every memory lives inside a <b>namespace</b>; every agent is bound to a <b>role</b>; every call is authorised before it touches storage.</sub>

```mermaid
flowchart TB
    CALLER["<b>Incoming call</b><br/>agent identity &bull; API key / JWT<br/>claims: namespace, role, thread"]

    AUTH["<b>AuthZ gate</b><br/>verify signature &bull; resolve role<br/>&bull; check write quota per agent"]

    CALLER ==> AUTH

    R1["<b>admin</b><br/>read/write/delete<br/>cross-namespace"]
    R2["<b>writer</b><br/>add + update<br/>own namespace"]
    R3["<b>reader</b><br/>recall only<br/>own namespace"]
    R4["<b>auditor</b><br/>read + timeline<br/>all namespaces"]
    R5["<b>agent</b><br/>add + recall<br/>own thread only"]
    R6["<b>service</b><br/>system writes<br/>provenance-signed"]

    subgraph ROLES ["&sect; RBAC ROLES &mdash; 6 built-in, customisable"]
        direction LR
        R1 ~~~ R2 ~~~ R3 ~~~ R4 ~~~ R5 ~~~ R6
    end

    AUTH ==> ROLES

    NS1["<b>namespace: fund-alpha</b><br/>portfolio planner threads<br/>risk analyst threads"]
    NS2["<b>namespace: fund-beta</b><br/>separate data plane<br/>no cross-read"]
    NS3["<b>namespace: research</b><br/>shared feed ingest<br/>read-only for funds"]

    subgraph TENANTS ["&sect; NAMESPACE BOUNDARIES &mdash; hard isolation"]
        direction LR
        NS1 ~~~ NS2 ~~~ NS3
    end

    ROLES ==>|"filtered by namespace + role"| TENANTS

    STORE[("<b>Doc &bull; Vector &bull; Graph</b><br/>row-level tenant column<br/>filtered on every query")]

    TENANTS ==> STORE

    style CALLER  fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style AUTH    fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style ROLES   fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style TENANTS fill:#F5F1E8,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style STORE   fill:#FBF8F1,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style R1      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style R2      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style R3      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style R4      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style R5      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style R6      fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style NS1     fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style NS2     fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style NS3     fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
```

<sub>Namespaces are enforced at the row level (tenant column on every row, filtered on every query), not just in application code. Cross-namespace reads require <code>admin</code> or <code>auditor</code>. Service role carries cryptographic provenance so feed ingests cannot impersonate an agent.</sub>

---

### Temporal &mdash; timeline &amp; supersession

<sub>Memwright doesn&rsquo;t overwrite &mdash; it <b>supersedes</b>. Every fact has a validity window, and the timeline is replayable to any point in the past. Auditors can answer not just <i>&ldquo;what does the desk know?&rdquo;</i> but <i>&ldquo;what did the desk know on 2026-04-10 at 08:00?&rdquo;</i></sub>

```mermaid
flowchart LR
    F1["<b>fact v1</b><br/><i>JPM CFO is Jeremy Barnum</i><br/>valid_from: 2022-05<br/>valid_to: <b>&infin;</b><br/>confidence: 0.95"]
    F2["<b>fact v2</b><br/><i>JPM CFO is Jane Doe</i><br/>valid_from: 2026-04-11<br/>valid_to: <b>&infin;</b><br/>confidence: 0.98<br/>supersedes: v1"]
    F3["<b>fact v3</b><br/><i>JPM CFO appointment delayed</i><br/>valid_from: 2026-04-12<br/>valid_to: <b>&infin;</b><br/>confidence: 0.90<br/>supersedes: v2"]

    subgraph TL ["&sect; TIMELINE &mdash; same entity, multiple states, none deleted"]
        direction LR
        F1 ==>|"new fact ingested<br/>contradiction detected"| F2
        F2 ==>|"corrected headline<br/>contradiction detected"| F3
    end

    Q1["<b>recall today</b><br/><i>who is JPM CFO?</i>"]
    Q2["<b>recall as-of 2026-04-11</b><br/><i>who was JPM CFO yesterday?</i>"]
    Q3["<b>auditor replay</b><br/><i>show full timeline</i>"]

    Q1 -->|"valid_to = &infin;<br/>latest confident fact"| A1["<b>v3 only</b><br/>&ldquo;appointment delayed&rdquo;"]
    Q2 -->|"filter: valid_at 2026-04-11"| A2["<b>v2 only</b><br/>&ldquo;Jane Doe&rdquo;"]
    Q3 -->|"no filter<br/>full chain"| A3["<b>v1 &rarr; v2 &rarr; v3</b><br/>with supersession edges"]

    TL ==> Q1
    TL ==> Q2
    TL ==> Q3

    style TL fill:#F5F1E8,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style F1 fill:#FBF8F1,stroke:#6B5F4F,color:#1A1614
    style F2 fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style F3 fill:#FBF8F1,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style Q1 fill:#1A1614,stroke:#C15F3C,color:#F5F1E8
    style Q2 fill:#1A1614,stroke:#C15F3C,color:#F5F1E8
    style Q3 fill:#1A1614,stroke:#C15F3C,color:#F5F1E8
    style A1 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style A2 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style A3 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
```

<sub>Supersession is a <b>graph edge</b>, not a delete. The document store keeps every version; the graph store links <code>v1 &mdash;supersedes&rarr; v2</code>. Recall defaults to &ldquo;latest confident fact,&rdquo; but any call can pass <code>as_of</code> to replay the past, which is how regulatory audit and post-mortem reconstruction both work on the same primitive.</sub>

---

### Provenance &mdash; source-to-citation chain

<sub>Every sentence the agent writes back to the advisor is <b>traceable to its source</b>. Not a paraphrase of a paraphrase &mdash; a cryptographic chain from raw feed ingest to grounded answer.</sub>

```mermaid
flowchart LR
    RAW["<b>Raw signal</b><br/><i>Reuters wire, 08:14 UTC</i><br/>&ldquo;JPM names Jane Doe CFO&rdquo;<br/>sha256: a1b2c3&hellip;"]

    ING["<b>Ingest</b><br/>parse &bull; entity extract<br/>attach provenance envelope<br/>source_id &bull; source_ts &bull; hash"]

    MEM["<b>Memory row</b><br/>id: mem_42<br/>content, entity=JPM<br/>provenance: {source_id, hash,<br/>ingest_ts, confidence: 0.98}"]

    RECALL["<b>Retrieval</b><br/>5-layer pipeline<br/>returns mem_42 among others"]

    ANS["<b>Agent answer</b><br/><i>&ldquo;Jane Doe was named CFO on<br/>2026-04-11 [source: Reuters]&rdquo;</i><br/>citations: [mem_42]"]

    AUDIT["<b>Auditor click-through</b><br/>mem_42 &rarr; raw Reuters wire<br/>with timestamp + hash check"]

    RAW ==> ING
    ING ==> MEM
    MEM ==> RECALL
    RECALL ==> ANS
    ANS -.->|citation link<br/>resolves to| AUDIT
    AUDIT -.->|verifies hash<br/>against raw signal| RAW

    style RAW    fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style ING    fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style MEM    fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style RECALL fill:#FBF8F1,stroke:#C15F3C,color:#1A1614
    style ANS    fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style AUDIT  fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
```

<sub>The citation in the agent&rsquo;s reply is not a string the LLM chose to emit &mdash; it&rsquo;s the <code>memory_id</code> carried through the retrieval pipeline. An auditor clicks the citation and lands on the raw wire with timestamp and content hash. If the upstream signal was tampered with, the hash check fails. This is what distinguishes <i>grounded</i> from <i>plausible</i>.</sub>

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

<p align="center">
  <img src="docs/deployment-matrix.svg" alt="Deployment matrix — one Memwright container, six targets, three storage roles swap per target" width="100%">
</p>

<sub><i>Every deployment is the same Python library wrapped in the same Starlette ASGI container. <code>DocumentStore</code>, <code>VectorStore</code>, and <code>GraphStore</code> are three interfaces; each column above is one implementation of each. Same API, same retrieval behavior, your infrastructure.</i></sub>

### Runtime topology &mdash; three integration modes

<sub>The same Memwright engine runs in three shapes &mdash; same storage, same retrieval, different coupling. Pick by latency budget and blast radius.</sub>

```mermaid
flowchart TB
    subgraph M1 ["&sect; MODE A &mdash; EMBEDDED LIBRARY &bull; lowest latency"]
        direction LR
        AG1["<b>Agent process</b><br/>Python &bull; import agent_memory"]
        MW1["<b>Memwright in-proc</b><br/>AgentMemory('./store')"]
        ST1[("<b>Local stores</b><br/>SQLite · Chroma · NetworkX<br/>on local disk")]
        AG1 ==> MW1 ==> ST1
    end

    subgraph M2 ["&sect; MODE B &mdash; SIDECAR CONTAINER &bull; process isolation"]
        direction LR
        AG2["<b>Agent container</b><br/>any language<br/>HTTP client"]
        MW2["<b>Memwright sidecar</b><br/>memwright api<br/>on localhost:8080"]
        ST2[("<b>Shared volume</b><br/>or managed backends")]
        AG2 ==>|"HTTP / MCP<br/>same pod"| MW2 ==> ST2
    end

    subgraph M3 ["&sect; MODE C &mdash; SHARED SERVICE &bull; multi-agent mesh"]
        direction LR
        AF1["<b>Agent A</b>"]
        AF2["<b>Agent B</b>"]
        AF3["<b>Agent C</b>"]
        MW3["<b>Memwright service</b><br/>App Runner · Cloud Run<br/>· Container Apps"]
        ST3[("<b>Managed backends</b><br/>Postgres · ArangoDB · Cosmos")]
        AF1 ==> MW3
        AF2 ==> MW3
        AF3 ==> MW3
        MW3 ==> ST3
    end

    style M1 fill:#F5F1E8,stroke:#1A1614,stroke-width:2px,color:#1A1614
    style M2 fill:#F5F1E8,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style M3 fill:#F5F1E8,stroke:#C15F3C,stroke-width:3px,color:#1A1614
    style AG1 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style MW1 fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style ST1 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style AG2 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style MW2 fill:#FBF8F1,stroke:#C15F3C,stroke-width:2px,color:#1A1614
    style ST2 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style AF1 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style AF2 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style AF3 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
    style MW3 fill:#1A1614,stroke:#C15F3C,stroke-width:2px,color:#F5F1E8
    style ST3 fill:#FBF8F1,stroke:#1A1614,color:#1A1614
```

<sub><b>Mode A</b> is sub-millisecond for a single agent prototyping on a laptop. <b>Mode B</b> adds language independence &mdash; a Go or Rust agent can call the sidecar over HTTP without Python in its image. <b>Mode C</b> is the production shape: one Memwright service in front of a multi-agent mesh, with managed storage behind. Code path is identical across all three &mdash; only configuration changes.</sub>

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
