# Voiceover script — setup-04-local-openai-embeddings.webm

**Video:** `docs/demo/setup-04-local-openai-embeddings.webm`
**Duration:** ~2m 13s (133s)
**Variant:** 4 of 8 — Local backend, **OpenAI embeddings** (vs local sentence-transformers)
**What changes from variant 01:** Q4 picks OpenAI (`text-embedding-3-large`) instead of local. Everything else is the same defaults.
**Tone:** calm, confident. One beat slower than conversational.

---

## Open — Claude Code header (0:00 – 0:08)

> Same install command. Same local backend. One upgrade — the embeddings.
>
> When recall quality matters more than offline-first, this is the variant you want.

## "install attestor" prompt typed (0:08 – 0:18)

> *Install agent memory.* The skill loads, the seven-question wizard starts. Questions one through three are the defaults from variant one: global scope, `~/.attestor` store, local backend. Three enters.

## Q1 — scope, default (0:18 – 0:24)

> Global. Default. Same as variant one.

## Q2 — store path, default (0:24 – 0:30)

> `~/.attestor`. Default. Same as variant one.

## Q3 — backend, default (0:30 – 0:38)

> Local backend — SQLite, ChromaDB, NetworkX. Everything embedded. Same as variant one.
>
> The interesting choice is the next one.

## Q4 — embeddings, OpenAI (0:38 – 1:15)

> Q4 is the one we change. Embeddings provider.
>
> The default, local sentence-transformers, runs `all-MiniLM-L6-v2` — a ninety-megabyte model, 384 dimensions, completely offline, completely free. It works. But it's a 2020-era architecture and it shows on multi-hop reasoning.
>
> On our own multi-agent benchmark — MAB — local embeddings score six point four percent. The same pipeline, same retrieval, same scoring, with OpenAI's `text-embedding-3-large` swapped in, jumps to roughly nineteen percent. Three times the recall on exactly the questions that matter for planner-executor and orchestrator-worker pipelines: questions where the right memory isn't lexically adjacent to the query.
>
> For technical documentation, code, API signatures, and cross-file reasoning, OpenAI's semantic space is simply tighter. If your agent is answering "which function in this codebase handles X," the stronger embedding earns its keep.
>
> Trade-offs are real. You need `OPENAI_API_KEY` exported — the wizard will ask for it right after this selection. Every `add` call sends the content to OpenAI; if that's a compliance problem, stop and use local. Cost is around thirteen cents per million tokens, so a normal solo workload is pennies a month. And you pick up network latency on every write, typically fifty to two hundred milliseconds.
>
> One arrow-key down. Enter.

## Q5 — hooks (1:15 – 1:35)

> Hooks. Default — all three. Session-start, post-tool-use, stop. Same as variant one.
>
> Worth noting: post-tool-use auto-captures every edit and shell command, so with OpenAI embeddings selected, every captured fact hits the OpenAI API once. Still cheap, but factor it into your budget if you're running an agent fleet.

## Q6 — namespace, default (1:35 – 1:45)

> Namespace `user`. Default. Same as variant one.

## Q7 — token budget, default (1:45 – 1:55)

> Ten thousand tokens. Default. Same as variant one.

## Install + doctor + MCP merge (1:55 – 2:13)

> `uv tool install attestor`, then `attestor doctor` verifies all four layers — and with OpenAI selected, the doctor's embedding check makes a live round-trip to confirm the key works. MCP entry merged into `~/.claude/.mcp.json`, hooks wired.
>
> Restart Claude Code, and recall now runs against a frontier embedding model — the same retrieval pipeline, sharper semantic matches.

---

## Recording notes

- The wizard will prompt for `OPENAI_API_KEY` if it's not already in the environment. Export it in the shell before `vhs` runs so the capture is clean; otherwise the tape will sit on the key prompt and you'll want to extend the final `Sleep`.
- Q4 section is the load-bearing one — let the voiceover breathe through the benchmark numbers and the trade-off list. Don't rush it.
- Keep the same one-beat-behind rhythm as variants 01 and 02. Let the highlight move to "OpenAI" before you name it.
- If you re-record with a longer Q4 explanation, push the `Sleep 7s` after Q3 up to `Sleep 10s` so the viewer can read the embeddings card before the arrow key fires.
