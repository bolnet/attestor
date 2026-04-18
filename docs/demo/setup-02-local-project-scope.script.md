# Voiceover script — setup-02-local-project-scope.webm

**Video:** `docs/demo/setup-02-local-project-scope.webm`
**Duration:** ~2m 13s (133s)
**Variant:** 2 of 8 — Local backend, **Project** scope (vs Global)
**What changes from variant 01:** Q1 picks Project (`./.mcp.json`) instead of Global (`~/.claude/.mcp.json`). Everything else is the same defaults.
**Tone:** calm, confident. One beat slower than conversational.

---

## Open — Claude Code header (0:00 – 0:08)

> Same install command, different scope.
>
> When you want Attestor attached to one specific repo — not every Claude Code session on your machine — you pick the project scope. The wizard handles it.

## "install attestor" prompt typed (0:08 – 0:18)

> Same three words as before. *Install agent memory.* Claude loads the install skill and starts the seven-question wizard.
>
> First question, scope, is the only one we change for this variant.

## Q1 — pick Project (0:18 – 0:35)

> Global writes the MCP entry to `~/.claude/.mcp.json` — every project on this machine sees Attestor. Project writes to `./.mcp.json` — only this directory does.
>
> Use Project when this repo needs its own isolated memory store, or when you want Attestor to ship with the codebase so collaborators get it automatically when they clone.
>
> One arrow-key down. Enter.

## Q2 — store path, default (0:35 – 0:50)

> Even with Project scope, the store itself can still live in your home directory — multiple project-scoped MCP entries can point to the same `~/.attestor`, sharing one memory pool across repos. Or each project can have its own.
>
> Default is `~/.attestor`. We'll keep it.

## Q3 — backend, Local (0:50 – 1:08)

> Backend stays Local. SQLite, ChromaDB, NetworkX — all embedded. The advantage of Local plus Project scope is portability: the entire memory layer lives in two places — one binary in your home and one MCP entry in the repo. No service, no daemon, no shared infrastructure.

## Q4 — embeddings, local (1:08 – 1:25)

> Local sentence-transformers. Same as variant one. The 90-megabyte model downloads once into your shared cache and serves every project-scoped install on this machine.

## Q5 — hooks (1:25 – 1:50)

> Hooks. Project-scoped MCP, but the hooks still write to your global `settings.json` — that's a Claude Code architectural choice, not ours. Session-start, post-tool-use, stop. All three recommended.
>
> If you'd rather keep this repo's behavior fully self-contained, deselect the hooks here and add per-repo hooks via Claude Code's project settings file later.

## Trailing wizard activity (1:50 – 2:13)

> The wizard is still asking — namespace, then token budget, then it'll run `uv tool install attestor`, run `attestor doctor`, and write the MCP entry into `./.mcp.json` right next to your code.
>
> The result: this repo, and only this repo, gets a Attestor-backed Claude Code session on the next restart.

---

## Recording notes

- The video ends mid-wizard at the hooks question because Q5 is multi-select — `Enter` toggles checkboxes rather than submitting. A future re-record can complete it by typing `Down Down Down Down Enter` to land on Submit, but the current cut is enough to demo the interaction.
- Variant 01 ends in the same spot for the same reason. Voice over both with the same closing pattern so they feel like a series.
- Don't pre-narrate "user picks Project" before the keystroke happens — let the highlight move first, then describe.
