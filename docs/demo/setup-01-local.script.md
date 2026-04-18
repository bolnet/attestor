# Voiceover script — setup-01-local.webm

**Video:** `docs/demo/setup-01-local.webm`
**Duration:** ~1m 42s (102s)
**Variant:** 1 of 6 — Local zero-config (SQLite + ChromaDB + NetworkX)
**Tone:** calm, confident, one beat slower than conversational. No hype.

---

## Open — Claude Code header appears (0:00 – 0:06)

> Attestor. The memory layer for agent teams.
>
> This is Claude Code, opened in any project. To install Attestor, you don't run a command. You just tell it what you want.

## "install attestor" prompt typed (0:06 – 0:14)

> Three words. *Install agent memory.* That's the entire interface.
>
> Claude picks up the install skill and starts the wizard — seven questions, all with sensible defaults, all answerable with a single keystroke.

## Q1 — MCP scope (0:14 – 0:24)

> First question: scope. Global means every Claude Code project on this machine sees Attestor. Project means just this directory.
>
> Default is global. We'll take it.

## Q2 — store path (0:24 – 0:34)

> Where should the memory live? Default is `~/.attestor` — one folder, three embedded backends inside it. Pick custom if you want it on a different drive. We'll keep the default.

## Q3 — backend type (0:34 – 0:44)

> Local or cloud? Local runs entirely on your machine. SQLite for the source of truth. ChromaDB for vectors. NetworkX for the entity graph. No Docker. No API keys. No external service.
>
> Cloud-managed is for teams that want a shared service across an agent fleet. Today we're showing local.

## Q4 — embeddings (0:44 – 0:54)

> Embeddings provider. Local sentence-transformers downloads a 90-megabyte model on first use and runs it in-process — completely offline, completely free. OpenAI or a cloud provider if you want stronger embeddings. Local is the recommended default.

## Q5 — hooks (0:54 – 1:08)

> Now the part that makes this feel native to Claude Code: hooks.
>
> Session-start injects up to twenty thousand tokens of relevant memory the moment a session begins, so Claude wakes up already knowing what it needs. Post-tool-use auto-captures facts from your edits and shell commands. Stop writes a session summary when you exit.
>
> All three are recommended. Multi-select, hit submit.

## Q6 — namespace (1:08 – 1:16)

> Namespace. Memories are tenant-isolated at the row level, so your personal memory never leaks into a work project, and a planner agent never reads an executor's working notes.
>
> Default `user` is fine for a solo install.

## Q7 — token budget (1:16 – 1:24)

> Last question: how many tokens of memory should `recall` return by default. Two thousand for tight context. Ten thousand for multi-agent workloads where the orchestrator needs broad situational awareness. Ten thousand is the recommended default.

## Install + doctor + MCP merge (1:24 – 1:42)

> That's all seven. Claude installs Attestor through `uv tool install`, runs `attestor doctor` to verify all four layers — document store, vector store, graph, and the five-layer retrieval pipeline — then merges the MCP server into your global config and wires the hooks into `settings.json`.
>
> Restart Claude Code once and the memory layer is attached. Sub-millisecond recall, deterministic ranking, no LLM in the critical path.
>
> Next variant: the same wizard, but pointed at Neon Postgres.

---

## Recording notes

- Pause the typing on each question card so the viewer can read the options before the voiceover lands.
- If you re-record longer, push the `Sleep` values in the tape; the script's section breaks already match question boundaries so you can rebalance per-section.
- Keep the voiceover one beat behind the on-screen action — don't pre-narrate what's about to be selected.
- Cut the music bed under the install/doctor section so the terminal output reads cleanly.
