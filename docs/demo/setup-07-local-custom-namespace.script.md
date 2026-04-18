# Voiceover script — setup-07-local-custom-namespace.webm

**Video:** `docs/demo/setup-07-local-custom-namespace.webm`
**Duration:** ~2m 15s (135s)
**Variant:** 7 of 8 — Local backend, **Custom namespace** (`aarjay-personal`)
**What changes from variant 01:** Q6 picks Custom and types a namespace name instead of accepting the default `user`. Everything else is the same defaults.
**Tone:** calm, confident. One beat slower than conversational.

---

## Open — Claude Code header (0:00 – 0:08)

> Same install command. Same defaults on scope, store, backend, embeddings, and hooks. One thing changes — the namespace.
>
> This is the variant where you decide whose memory this really is.

## "install attestor" prompt typed (0:08 – 0:18)

> *Install agent memory.* Wizard fires up. We're going to walk through the first five questions on autopilot — global scope, default path, local backend, local embeddings, all three hooks — then stop and actually think about question six.

## Q1 — scope, default (0:18 – 0:28)

> Global scope. Default, same as variant one. Enter.

## Q2 — store path, default (0:28 – 0:38)

> Default `~/.attestor`. Same as variant one. Enter.

## Q3 — backend, Local (0:38 – 0:48)

> Local backend. SQLite, ChromaDB, NetworkX. Same as variant one. Enter.

## Q4 — embeddings, local (0:48 – 0:58)

> Local sentence-transformers. Same as variant one. Enter.

## Q5 — hooks, default (0:58 – 1:10)

> All three hooks on — session-start, post-tool-use, stop. Same as variant one. Submit.

## Q6 — namespace, Custom `aarjay-personal` (1:10 – 1:55)

> This is the question that matters for this variant.
>
> Every memory Attestor stores carries a namespace column. It's a row-level tenant tag — not a separate database, not a separate store, just a column. `recall` filters on it by default, which means memories written under one namespace are invisible to queries from another. Personal notes don't bleed into a work project. A research agent doesn't read an ops agent's working notes. An executor doesn't see the planner's scratchpad unless you explicitly share.
>
> The default `user` is fine if you're running a single-purpose install on a single machine. Most people should keep it.
>
> Custom is the right choice when you need separation. A consultant juggling three clients wants `client-acme`, `client-beta`, `client-gamma` — one store, three tenants, zero leakage. Someone running distinct personas — research, ops, personal — wants a namespace per persona. Teams running dev, staging, and prod agents on the same infrastructure want one per environment.
>
> In multi-agent topologies, namespaces are how you draw the collaboration boundary. Give the orchestrator, planner, and executor their own namespaces for strict isolation, or share one so they pool context for a joint task. Both are valid. The schema supports it either way.
>
> One caveat — namespaces are cheap to create and hard to merge after the fact. Memories written under `aarjay-personal` stay tagged `aarjay-personal` unless you write a migration. Pick the name early and pick it on purpose.
>
> Here, one arrow down to Custom. Enter. Type `aarjay-personal`. Enter.

## Q7 — token budget, default (1:55 – 2:05)

> Ten thousand tokens. Default, same as variant one. Enter.

## Install + doctor + MCP merge (2:05 – 2:15)

> Seven questions answered. `uv tool install attestor`, `attestor doctor`, MCP merge into the global config, hooks wired into `settings.json`.
>
> Restart Claude Code and this session is attached to the `aarjay-personal` namespace. Everything written in this agent's lifetime — every captured fact, every session summary — lands with that tenant tag. Nothing else in the store can see it, and it can't see anything else.

---

## Recording notes

- Q6 is where the narrator should slow down most — the first five questions are one-beat filler to establish that we're running the same defaults as variant 01, and the payoff is the namespace discussion.
- The video ends mid-wizard at hooks for variants 01 and 02 because Q5 is multi-select; this variant goes all the way through because we deliberately pick Custom and type a value. Tape timing assumes the hooks multi-select auto-submits on Enter — if it doesn't in practice, add `Down Down Down Down Enter` before Q6 and rebalance the sleep budget.
- Don't pre-narrate "user types aarjay-personal" — let the text appear in the terminal, then describe.
- Keep the closing under ten seconds so the series stays roughly uniform in duration.
