# VHS tape: Attestor local install demo (SQLite + ChromaDB + NetworkX, no API keys)
# Render: vhs docs/demo/install-local.tape
# Output: docs/demo/install-local.gif + docs/demo/install-local.mp4

Output docs/demo/install-local.gif
Output docs/demo/install-local.mp4

Set FontSize 15
Set Width 1200
Set Height 800
Set Theme "Dracula"
Set Padding 20
Set TypingSpeed 45ms
Set PlaybackSpeed 1.0

# ── Open Claude Code ──
Type "claude"
Enter
Sleep 3s

# ── User types the install command ──
Type "install attestor"
Enter
Sleep 2s

# ── Step 1: Detection ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "echo ''"
Enter
Type "echo '  ╔══════════════════════════════════════════════════════╗'"
Enter
Type "echo '  ║     ATTESTOR INSTALL WIZARD                        ║'"
Enter
Type "echo '  ║     Memory layer for agent teams                    ║'"
Enter
Type "echo '  ╚══════════════════════════════════════════════════════╝'"
Enter
Type "echo ''"
Enter
Type "echo '  Detecting current state...'"
Enter
Type "echo '  ✓ Python 3.12 found'"
Enter
Type "echo '  ✓ pipx available'"
Enter
Type "echo '  ✗ attestor not installed'"
Enter
Type "echo '  ✗ No existing store found'"
Enter
Type "echo '  ✗ No MCP config for attestor'"
Enter
Type "echo ''"
Enter
Sleep 3s

# ── Q1: Scope ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  Q1 · Where should the MCP config live?              │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  › Global  (~/.claude/.mcp.json)    ← every project"
Enter
Type "    Project (./.mcp.json)            ← this repo only"
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 3s

# User selects Global
Type "echo '  ✓ Selected: Global (~/.claude/.mcp.json)'"
Enter
Sleep 1500ms

# ── Q2: Store path ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  Q2 · Where to store memories?                       │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  › ~/.attestor/              (Recommended)"
Enter
Type "    Custom path"
Enter
Type ""
Enter
Type "  SQLite DB, ChromaDB vectors, and NetworkX graph"
Enter
Type "  will all live in this directory."
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 3s

Type "echo '  ✓ Selected: ~/.attestor/'"
Enter
Sleep 1500ms

# ── Q3: Backend ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  Q3 · Backend type                                   │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  › Local (Recommended)"
Enter
Type "    ├─ SQLite      (document store, built into Python)"
Enter
Type "    ├─ ChromaDB    (vector store, local embeddings)"
Enter
Type "    └─ NetworkX    (graph store, in-process)"
Enter
Type ""
Enter
Type "    Cloud-managed"
Enter
Type "    ├─ PostgreSQL  (pgvector + Apache AGE)"
Enter
Type "    ├─ ArangoDB    (doc + vector + graph)"
Enter
Type "    ├─ AWS         (DynamoDB + OpenSearch + Neptune)"
Enter
Type "    ├─ Azure       (Cosmos DB DiskANN)"
Enter
Type "    └─ GCP         (AlloyDB + ScaNN)"
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 4s

Type "echo '  ✓ Selected: Local'"
Enter
Type "echo '    No Docker. No API keys. Everything embedded.'"
Enter
Sleep 2s

# ── Q4: Embedding provider ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  Q4 · Embedding provider                             │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  › Local — all-MiniLM-L6-v2 (384D, ~90MB, FREE)"
Enter
Type "    OpenAI — text-embedding-3-small (requires API key)"
Enter
Type "    Bedrock — amazon.titan-embed-text-v2 (requires AWS)"
Enter
Type "    Vertex — text-embedding-005 (requires GCP)"
Enter
Type ""
Enter
Type "  Local runs sentence-transformers in-process."
Enter
Type "  No API key needed. No network calls."
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 4s

Type "echo '  ✓ Selected: Local (all-MiniLM-L6-v2, 384D)'"
Enter
Type "echo '    Zero cost. Zero latency. Zero API keys.'"
Enter
Sleep 2s

# ── Q5: Hooks ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  Q5 · Claude Code hooks                              │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  [✓] session-start   — inject 20K tokens of context"
Enter
Type "  [✓] post-tool-use   — auto-capture from Write/Edit/Bash"
Enter
Type "  [✓] stop            — session summary on exit"
Enter
Type "  [ ] none"
Enter
Type ""
Enter
Type "  Hooks auto-wire into ~/.claude/settings.json."
Enter
Type "  They capture memories without you asking."
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 4s

Type "echo '  ✓ Selected: session-start + post-tool-use + stop'"
Enter
Sleep 1500ms

# ── Q6: Namespace ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  Q6 · Default namespace                              │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "  › default"
Enter
Type "    Custom name"
Enter
Type ""
Enter
Type "  Namespaces isolate memories between agents."
Enter
Type "  Each agent writes to its own namespace."
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 3s

Type "echo '  ✓ Selected: default'"
Enter
Sleep 1500ms

# ── Q7: Token budget ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ┌──────────────────────────────────────────────────────┐"
Enter
Type "  │  Q7 · Default token budget for recall()              │"
Enter
Type "  └──────────────────────────────────────────────────────┘"
Enter
Type ""
Enter
Type "    2000   (conservative)"
Enter
Type "    5000   (balanced)"
Enter
Type "  › 10000  (recommended for multi-agent)"
Enter
Type "    20000  (aggressive)"
Enter
Type "    Custom"
Enter
Type ""
Enter
Type "  This caps how many tokens recall() returns per query."
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 3s

Type "echo '  ✓ Selected: 10000 tokens'"
Enter
Sleep 1500ms

# ── Installing ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "echo ''"
Enter
Type "echo '  Installing attestor...'"
Enter
Sleep 1s
Type "echo '  $ pipx install attestor'"
Enter
Sleep 2s
Type "echo '    installed package attestor 2.0.1'"
Enter
Type "echo '    ✓ attestor → ~/.local/bin/attestor'"
Enter
Sleep 1500ms

Type "echo ''"
Enter
Type "echo '  Provisioning store at ~/.attestor/...'"
Enter
Sleep 1s
Type "echo '    ✓ SQLite database created'"
Enter
Type "echo '    ✓ ChromaDB collection initialized'"
Enter
Type "echo '    ✓ NetworkX graph provisioned'"
Enter
Sleep 2s

Type "echo ''"
Enter
Type "echo '  Writing MCP config → ~/.claude/.mcp.json'"
Enter
Sleep 1s
Type "echo '    ✓ MCP server: attestor mcp --path ~/.attestor'"
Enter
Sleep 1500ms

Type "echo ''"
Enter
Type "echo '  Wiring hooks → ~/.claude/settings.json'"
Enter
Sleep 1s
Type "echo '    ✓ session-start hook added'"
Enter
Type "echo '    ✓ post-tool-use hook added'"
Enter
Type "echo '    ✓ stop hook added'"
Enter
Sleep 2s

# ── Doctor check ──
Type "echo ''"
Enter
Type "echo '  Running attestor doctor...'"
Enter
Sleep 1500ms
Type "echo ''"
Enter
Type "echo '  ┌─────────────────────────────────────────────┐'"
Enter
Type "echo '  │  ATTESTOR DOCTOR                            │'"
Enter
Type "echo '  ├─────────────────────────────────────────────┤'"
Enter
Type "echo '  │  ● Document Store (SQLite)        OK  0.2ms │'"
Enter
Type "echo '  │  ● Vector Store (ChromaDB)        OK  1.1ms │'"
Enter
Type "echo '  │  ● Graph Store (NetworkX)         OK  0.1ms │'"
Enter
Type "echo '  │  ● Retrieval Pipeline             OK  3 / 3 │'"
Enter
Type "echo '  ├─────────────────────────────────────────────┤'"
Enter
Type "echo '  │  ALL SYSTEMS OPERATIONAL                     │'"
Enter
Type "echo '  └─────────────────────────────────────────────┘'"
Enter
Sleep 4s

# ── Final summary ──
Type@0ms "clear"
Enter
Sleep 300ms
Type "cat <<'EOF'"
Enter
Type ""
Enter
Type "  ╔══════════════════════════════════════════════════════╗"
Enter
Type "  ║  INSTALL COMPLETE                                    ║"
Enter
Type "  ╠══════════════════════════════════════════════════════╣"
Enter
Type "  ║                                                      ║"
Enter
Type "  ║  Binary     ~/.local/bin/attestor                   ║"
Enter
Type "  ║  Store      ~/.attestor/                            ║"
Enter
Type "  ║  Backends   SQLite + ChromaDB + NetworkX             ║"
Enter
Type "  ║  Embeddings all-MiniLM-L6-v2 (384D, local)          ║"
Enter
Type "  ║  MCP        ~/.claude/.mcp.json                      ║"
Enter
Type "  ║  Hooks      session-start, post-tool-use, stop       ║"
Enter
Type "  ║  Namespace  default                                  ║"
Enter
Type "  ║  Budget     10000 tokens                             ║"
Enter
Type "  ║                                                      ║"
Enter
Type "  ║  ▶ Restart Claude Code to activate.                  ║"
Enter
Type "  ║                                                      ║"
Enter
Type "  ║  No Docker. No API keys. Zero config.               ║"
Enter
Type "  ║                                                      ║"
Enter
Type "  ╚══════════════════════════════════════════════════════╝"
Enter
Type ""
Enter
Type "EOF"
Enter
Sleep 6s
