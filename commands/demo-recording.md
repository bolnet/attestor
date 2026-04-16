---
description: Record a demo video + screenshots of the Memwright UI
argument-hint: "(no args — uses running server)"
allowed-tools: Bash, Read, Write, Edit, Glob, mcp__playwright__browser_navigate, mcp__playwright__browser_take_screenshot, mcp__playwright__browser_evaluate, mcp__playwright__browser_snapshot, mcp__playwright__browser_click, mcp__playwright__browser_fill_form, mcp__playwright__browser_press_key, mcp__playwright__browser_wait_for
---

# Record Memwright UI Demo

You are recording a demo walkthrough of the Memwright UI — screenshots of every page and a screen-recorded video navigating the full interface.

## Prerequisites

1. Memwright UI must be running (`memwright ui --path <store>` on port 8080)
2. The store should have 20+ seeded memories across multiple entities/namespaces for realistic content
3. Playwright MCP must be available for screenshots
4. Node.js + `playwright` npm package must be installed for video recording

## Step 1 — Verify Server

Check that `http://127.0.0.1:8080/ui/memories` is reachable. If not, tell the user to start the server first.

## Step 2 — Capture Screenshots

Navigate to each page in order using Playwright MCP and take a screenshot at 1280x800. Save all screenshots to `docs/demo/screens/`.

| # | Page | URL | Filename | Notes |
|---|------|-----|----------|-------|
| 01 | Evidence Board | `/ui/memories` | `01-evidence-board.png` | Landing page, card grid |
| 02 | Search | `/ui/memories?q=architecture` | `02-search-highlight.png` | Type "architecture", show highlights |
| 03 | Memory Detail | Click first card | `03-memory-detail.png` | Dossier view |
| 04 | Graph | `/ui/graph` | `04-graph.png` | Force-directed entity graph |
| 05 | Recall Pipeline | `/ui/recall` | `05-recall-pipeline.png` | Execute query, show 5-layer cascade |
| 06 | Budget Explorer | Same page, click Budget Explorer | `06-budget-explorer.png` | 5-budget comparison chart |
| 07 | Timeline | `/ui/timeline` | `07-timeline.png` | Chronological memory timeline |
| 08 | Agents | `/ui/agents` | `08-agents.png` | Agent activity breakdown |
| 09 | Health | `/ui/health` | `09-health.png` | System health + latency percentiles |
| 10 | Config | `/ui/config` | `10-config.png` | System configuration viewer |
| 11 | Ops Log | `/ui/ops` | `11-ops-log.png` | Flight recorder table |
| 12 | Light Theme | `/ui/memories` + toggle | `12-light-theme.png` | Switch to light theme |

For interactive pages (Recall, Budget Explorer), fill in the query "how does the order service work?" with budget 2000 before capturing.

After each screenshot, switch back to dark theme if you toggled to light.

## Step 3 — Record Video

Run the video recording script:

```bash
node docs/demo/record-demo.js
```

This uses Playwright's `recordVideo` API to navigate all pages with pauses for readability. The output is a `.webm` file in `docs/demo/`.

After recording, rename the output to `docs/demo/ui-walkthrough.webm`:

```bash
# Find the auto-generated video file and rename it
mv docs/demo/*.webm docs/demo/ui-walkthrough.webm 2>/dev/null || true
```

If `record-demo.js` doesn't exist or fails, create/fix it following this structure:
- Launch Chromium with `recordVideo` at 1280x800
- Visit each page in sequence with 2-3s pauses
- For Recall: fill query + budget, click Execute, wait for results
- For Budget Explorer: click the button, scroll to see chart
- For Theme: toggle light then back to dark
- Close context to finalize video

## Step 4 — Verify Outputs

List all files in `docs/demo/screens/` and confirm 12 PNG screenshots exist.
Confirm `docs/demo/ui-walkthrough.webm` exists and report its file size.

## Step 5 — Summary

Report what was captured:
- Number of screenshots
- Video file path and size
- Any pages that had issues (empty content, errors, missing data)

## Recording Tips

- **Resolution**: 1280x800 viewport
- **Wait times**: 2-3s per page for readability in video
- **Mouse**: Slow, deliberate movements
- **Data**: If the store looks empty, suggest seeding with `memwright add`
- **Theme**: Always end in dark mode (the default aesthetic)
- **Narration script**: See `docs/demo/ui-walkthrough.script.md` for the full narration guide
