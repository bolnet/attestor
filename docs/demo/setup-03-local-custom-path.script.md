# Voiceover script — setup-03-local-custom-path.webm

**Video:** `docs/demo/setup-03-local-custom-path.webm`
**Duration:** ~2m 15s (135s)
**Variant:** 3 of 8 — Local backend, **custom store path** (`~/code/memstore`)
**What changes from variant 01:** Q2 picks a custom path instead of the default `~/.attestor`. Everything else stays on defaults — Global scope, Local backend, local embeddings, all three hooks, `user` namespace, 10K token budget.
**Tone:** calm, confident. One beat slower than conversational.

---

## Open — Claude Code header (0:00 – 0:08)

> Same install command, same local backend — this time we move the store somewhere of our choosing.
>
> The default `~/.attestor` works for most people. But the wizard lets you put the memory anywhere on disk, and that matters more than it sounds.

## "install attestor" prompt typed (0:08 – 0:18)

> Three words again. *Install agent memory.* Claude loads the install skill and starts the seven-question wizard.
>
> First question, scope, we leave on Global — same as variant one.

## Q1 — scope, default Global (0:18 – 0:28)

> Global. Every Claude Code project on this machine will see Attestor through `~/.claude/.mcp.json`. Default, Enter, move on.

## Q2 — store path, custom (0:28 – 1:05)

> This is the question that changes.
>
> Default is `~/.attestor` — one folder in your home, three embedded backends inside it. It's the right choice for most installs. But there are real reasons to move it.
>
> You might want the store on an external SSD — memory grows fast when hooks are auto-capturing, and a fast NVMe drive keeps recall latency flat even when the store reaches gigabytes.
>
> You might want a bigger volume — laptop internal disks are expensive; a dedicated code drive usually isn't.
>
> You might want a dedicated per-project memory pool — point this install at `~/code/memstore`, point the next install at `~/research/memstore`, and you get clean isolation without running multiple services.
>
> Or you just prefer everything under `~/code` because that's where your brain already lives.
>
> One arrow-key down to land on Custom path. Enter. Then type the path — `~/code/memstore`. The wizard expands the tilde, creates the directory if it doesn't exist, and passes it to every backend that needs it.

## Q3 — backend, Local (1:05 – 1:20)

> Backend stays Local — same as variant one. SQLite for the source of truth, ChromaDB for vectors, NetworkX for the graph. All three will be provisioned inside `~/code/memstore` instead of `~/.attestor`.
>
> No Docker. No API keys. No external service. The path changed, the architecture didn't.

## Q4 — embeddings, local (1:20 – 1:35)

> Embeddings. Local sentence-transformers, same as variant one — the 90-megabyte model caches once and serves every store on this machine, no matter where the store lives on disk.

## Q5 — hooks, all three (1:35 – 2:00)

> Hooks. Session-start, post-tool-use, stop — all three recommended, same as variant one. They read and write through the MCP server, not through the filesystem directly, so they don't care that the store moved. Point the server at `~/code/memstore` once in the MCP config, and every hook follows.

## Trailing wizard activity (2:00 – 2:15)

> Namespace stays on `user`. Token budget stays on ten thousand. `uv tool install attestor` runs, `attestor doctor` verifies all four layers against the new path, and the MCP entry merges into your global config pointing at `~/code/memstore`.
>
> Restart Claude Code once. Memory is live — just living somewhere you chose.

---

## Recording notes

- Give Q2 extra breathing room. It's the variant's entire reason to exist, and the typed path is the only place in the video where the user enters free text — let the keystrokes land cleanly.
- Like variants 01 and 02, the video may end mid-wizard at the hooks multi-select. Voice the closing line the same way so the series feels continuous.
- Before recording: `rm -rf ~/code/memstore` as well as the usual cleanup, so the wizard actually creates the directory on camera.
- Don't pre-narrate the custom path — let the viewer see "Custom path" highlight, then the text entry field, before describing what's being typed.
