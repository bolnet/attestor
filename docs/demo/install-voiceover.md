# Install Attestor — Voiceover Script

**Video**: `docs/demo/install-local.mp4`

---

## [0:00] Opening

Open Claude Code and type "install attestor." That's the only command you need.

## [0:05] Detection

The wizard scans your system. Python 3.12 — found. pipx — available. No existing attestor installation. No store. No MCP config. Clean slate.

## [0:12] Q1 — Scope

First question: where should the MCP config live? Global makes attestor available in every project. Project scope limits it to one repo. We'll go with Global.

## [0:18] Q2 — Store Path

Where to store memories? The default is ~/.attestor. SQLite database, ChromaDB vectors, and NetworkX graph all live in this one directory. We'll keep the default.

## [0:25] Q3 — Backend

Backend type. Local runs everything on your machine — SQLite for documents, ChromaDB for vectors, NetworkX for the graph. No Docker. No API keys. Everything embedded. The cloud options are there if you need them — Postgres, ArangoDB, AWS, Azure, GCP — but local is zero config.

## [0:38] Q4 — Embeddings

Embedding provider. Local uses all-MiniLM-L6-v2 — 384 dimensions, about 90 megabytes, completely free. Runs sentence-transformers in-process. No network calls. No API key. Zero cost, zero latency.

## [0:50] Q5 — Hooks

Claude Code hooks. Three to choose from. Session-start injects 20,000 tokens of context when you open a conversation. Post-tool-use auto-captures memories from Write, Edit, and Bash calls. Stop writes a session summary on exit. We'll enable all three.

## [1:02] Q6 — Namespace

Default namespace. Namespaces isolate memories between agents. Each agent writes to its own namespace. We'll use "default."

## [1:08] Q7 — Token Budget

Token budget for recall. This caps how many tokens each recall query returns. 10,000 is recommended for multi-agent workflows. Gives enough context without flooding the prompt.

## [1:16] Installing

Now it runs. pipx installs attestor 2.0.1. The store provisions — SQLite database created, ChromaDB collection initialized, NetworkX graph ready. MCP config written to ~/.claude/.mcp.json. All three hooks wired into settings.json.

## [1:30] Doctor Check

Attestor doctor runs automatically. Document store — OK, 0.2 milliseconds. Vector store — OK, 1.1 milliseconds. Graph store — OK, 0.1 milliseconds. Retrieval pipeline — all 3 layers active. All systems operational.

## [1:42] Summary

That's it. Attestor installed. SQLite plus ChromaDB plus NetworkX. Local embeddings. Three hooks wired. No Docker. No API keys. Zero config. Restart Claude Code and your agents have memory.
