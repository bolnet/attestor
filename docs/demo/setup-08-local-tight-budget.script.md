# Voiceover script — setup-08-local-tight-budget.webm

**Video:** `docs/demo/setup-08-local-tight-budget.webm`
**Duration:** ~2m 15s (135s)
**Variant:** 8 of 8 — Local backend, **tight token budget** (2000 instead of 10000)
**What changes from variant 01:** Q7 picks `2000` tokens for the default `recall` budget. Q1–Q6 are the same defaults as variant 01.
**Tone:** calm, confident. One beat slower than conversational.

---

## Open — Claude Code header (0:00 – 0:08)

> Same install command. Same local stack. One knob turned down at the end — the token budget.
>
> This is the variant for people who want Attestor lean.

## "install attestor" prompt typed (0:08 – 0:18)

> *Install agent memory.* Claude loads the skill, the seven-question wizard begins. We're going to sprint through the first six and land on the last one.

## Q1 — MCP scope, Global (0:18 – 0:30)

> Default, same as variant one. Global scope, every Claude Code session on this machine sees Attestor.

## Q2 — store path, default (0:30 – 0:40)

> Default `~/.attestor`. Same as variant one.

## Q3 — backend, Local (0:40 – 0:52)

> Local stack. SQLite, ChromaDB, NetworkX. Same as variant one.

## Q4 — embeddings, local (0:52 – 1:02)

> Local sentence-transformers. Same as variant one.

## Q5 — hooks (1:02 – 1:15)

> All three hooks — session-start, post-tool-use, stop. Same as variant one.

## Q6 — namespace (1:15 – 1:25)

> Default `user` namespace. Same as variant one.

## Q7 — token budget, 2000 (1:25 – 2:05)

> This is the one that matters for this variant.
>
> `recall` returns up to N tokens of memories per call. The default — ten thousand — is sized for multi-agent orchestration, where a planner needs broad situational awareness before it decides what the executors should do. Broad context, broad budget.
>
> Two thousand is the right choice when you're driving a single agent, working inside tight Claude Code sessions, and you don't want auto-injection to crowd out your actual prompt. Every recalled token is paid input on the next turn. Over a long session, two thousand versus ten thousand is real money.
>
> The five-layer retrieval pipeline doesn't get worse at a smaller budget. Tag match, graph expansion, vector cosine, RRF fusion, MMR diversity — the ranking runs the same. Then the greedy packer fills to whatever ceiling you gave it. You still get the top-ranked memories. You just get fewer of them.
>
> And it's a ceiling, not a floor. When you actually need more on a specific call, you pass `max_tokens` explicitly — `mem.recall(query, max_tokens=20000)` — and you get it. The wizard is setting a sane default, not a hard cap.
>
> Two up-arrows from Recommended lands on `2000`. Enter.

## Install + doctor + MCP merge (2:05 – 2:15)

> `uv tool install attestor`, then `attestor doctor` verifies all four layers — document store, vector store, graph, retrieval pipeline — then the MCP entry writes into `~/.claude/.mcp.json` and the hooks land in `settings.json`.
>
> Restart Claude Code once and you're attached. Same local stack as variant one — just quieter in your context window.

---

## Recording notes

- Q7 is the only question where the cursor has to move; sleep generously before the first `Up` so the viewer can read the three budget options before the highlight jumps.
- 400ms between the two `Up` keystrokes is deliberately slow — it lets the viewer *see* the second move rather than blurring both into one.
- Don't pre-narrate "user picks 2000" before the keystroke lands. Let the highlight arrive on `2000`, *then* name it.
- The long Q7 voiceover is the whole point of this cut — it's the only variant that justifies slowing the pace. If you re-record, push Q7's `Sleep` values higher rather than speeding up the narration.
- Voice Q1–Q6 as near-identical to variant 01 so the contrast at Q7 reads sharply.
