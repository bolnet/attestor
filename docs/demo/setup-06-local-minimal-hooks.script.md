# Voiceover script — setup-06-local-minimal-hooks.webm

**Video:** `docs/demo/setup-06-local-minimal-hooks.webm`
**Duration:** ~2m 15s (135s)
**Variant:** 6 of 8 — Local backend, **minimal hooks** (session-start only)
**What changes from variant 01:** Q5 deselects `post-tool-use` and `stop`, leaving only `session-start`. Everything else is defaults.
**Tone:** calm, confident. One beat slower than conversational.

---

## Open — Claude Code header (0:00 – 0:08)

> Same install command. Same local backend. One deliberate change — how much of the session Attestor listens to.

## "install attestor" prompt typed (0:08 – 0:18)

> *Install agent memory.* The wizard opens. Seven questions. We're going to take the defaults on six of them and customize exactly one: hooks.

## Q1 — scope, default (0:18 – 0:28)

> Global scope. Default. Same as variant one.

## Q2 — store path, default (0:28 – 0:38)

> `~/.attestor`. Default.

## Q3 — backend, Local (0:38 – 0:50)

> Local — SQLite, ChromaDB, NetworkX. Default.

## Q4 — embeddings, local (0:50 – 1:02)

> Local sentence-transformers. Default.

## Q5 — hooks, session-start only (1:02 – 1:50)

> This is the question worth slowing down on.
>
> Three hooks are offered. *Session-start* injects up to twenty thousand tokens of relevant memory when a Claude Code session begins — you wake up with the project's context already loaded. *Post-tool-use* auto-captures facts from every `Write`, `Edit`, and `Bash` tool call Claude makes. *Stop* writes a session summary when you exit.
>
> The default is all three. It's the right default for most people. But there's a workflow where *minimal* is better.
>
> Post-tool-use captures everything. In a disciplined session where you're landing a specific change, that's gold — you end the session with a precise record of what was edited and why. In an exploratory session where Claude is reading ten files, writing four scratch scripts, and rewriting two of them, it's noise. The store fills up with half-formed artifacts that dilute future retrievals.
>
> Stop writes a session summary. If you already close each session with a detailed git commit, the summary is duplicative — two records of the same work, one of them less precise.
>
> Session-start alone is the highest-value hook. You get the twenty-thousand-token context injection on every session — the thing that actually changes how Claude behaves. But writes only happen when *you* decide to call `mem.add` or invoke the recall/capture skills explicitly. Capture becomes deliberate instead of automatic.
>
> Two mental models. *Capture everything, filter later* — pick all three. *Capture deliberately, keep the signal clean* — pick only session-start.
>
> On the card: session-start is already checked at the top. We arrow down past post-tool-use, past stop, past the "none" option, onto Submit. Enter.
>
> Easy to change later. Rerun `install attestor`, the wizard detects the existing install, and offers to update the hook selection in place.

## Q6 — namespace (1:50 – 2:00)

> Namespace default. Same as variant one.

## Q7 — token budget (2:00 – 2:08)

> Ten thousand tokens for recall. Default.

## Install + doctor + MCP merge (2:08 – 2:15)

> `uv tool install attestor`. `attestor doctor`. MCP merged into `~/.claude/.mcp.json`. `settings.json` gets exactly one hook entry — session-start. Clean.
>
> Restart Claude Code once. On the next session, Attestor injects context at the start, and from then on the store only grows when you choose to grow it.

---

## Recording notes

- Q5 gets the airtime. Viewers watching the series will see variants 01 and 02 take the default-all-three path; this one is the counterweight. Let the voiceover slow down through the post-tool-use / stop tradeoff — that's the whole point of the variant.
- The Q5 navigation is four `Down` presses without pressing `Space`: post-tool-use, stop, none, Submit. Don't describe individual keystrokes in the voiceover — describe the intent ("arrow down to Submit") and let the card do the showing.
- Don't pre-narrate "user picks minimal" before the selector lands on Submit.
- If re-recording to tighten pacing, the Q5 section can absorb extra seconds from Q1–Q4 since those are pure defaults. The other sections can each lose a beat.
