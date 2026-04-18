# Recording demo videos with VHS

All `docs/demo/*.webm` and `*.gif` files are produced with [VHS](https://github.com/charmbracelet/vhs) — a headless terminal that runs a real PTY, executes the commands you script, and renders the result to video via ffmpeg.

## Install

```bash
brew install vhs ffmpeg
```

## Render a tape

```bash
vhs docs/demo/setup-01-local.tape
```

Every `.tape` file in this directory is self-contained. The first `Output` line declares the destination(s).

## The series

Each tape walks through **one** setup variant of `install attestor` so a viewer can pick the path that matches their environment:

| # | Tape | Variant |
|---|------|---------|
| 01 | `setup-01-local.tape`     | Zero-config local (SQLite + ChromaDB + NetworkX) |
| 02 | `setup-02-postgres.tape`  | Neon PostgreSQL (pgvector + Apache AGE) |
| 03 | `setup-03-arangodb.tape`  | ArangoDB Oasis (doc + vector + graph in one) |
| 04 | `setup-04-aws.tape`       | AWS (DynamoDB + OpenSearch + Neptune) |
| 05 | `setup-05-azure.tape`     | Azure Cosmos DB DiskANN |
| 06 | `setup-06-gcp.tape`       | GCP AlloyDB (pgvector + AGE + ScaNN) |

Render all:

```bash
for t in docs/demo/setup-*.tape; do vhs "$t"; done
```

## Conventions

- **Dimensions:** `1440x900` (matches the existing `claude-finance` recordings so they crop identically).
- **Theme:** `Catppuccin Mocha`. Font size `16`. Padding `20`.
- **Typing speed:** `35ms` (readable but not painful).
- **Output:** WebM only by default (smaller, lossless-enough for landing pages). Add an `Output …gif` line if a GIF is needed.

## Patterns

### Run a real Claude Code session

VHS spawns a real PTY, so `claude` actually launches. Sleep long enough for the output to stream:

```
Type "claude"
Enter
Sleep 3s
Type "install attestor"
Enter
Sleep 30s
```

For the install wizard's `AskUserQuestion` prompts, use arrow keys + Enter:

```
Sleep 2s          # let the question render
Down              # move selector
Down
Enter             # accept
```

### Simulate without running

When the real flow is too slow, too noisy, or requires secrets, render a fake transcript with `cat <<'EOF'` heredocs (see `install-wizard.tape` for the canonical example).

### Hide setup

```
Hide
Type "rm -rf ~/.attestor-demo && export ATTESTOR_STORE=~/.attestor-demo"
Enter
Show
```

Anything between `Hide`/`Show` runs but is not visible in the recording.

## Re-encoding to mp4 / gif

VHS emits VP9 WebM. To get an mp4 or smaller gif:

```bash
ffmpeg -i setup-01-local.webm -c:v libx264 -pix_fmt yuv420p setup-01-local.mp4
ffmpeg -i setup-01-local.webm -vf "fps=15,scale=900:-1:flags=lanczos" setup-01-local.gif
```
