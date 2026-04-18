---
description: Record a comprehensive demo video + voiceover script of the Attestor UI
argument-hint: "(no args — uses running server on port 8080)"
allowed-tools: Bash, Read, Write, Edit, Glob, mcp__playwright__browser_navigate, mcp__playwright__browser_take_screenshot, mcp__playwright__browser_evaluate, mcp__playwright__browser_snapshot, mcp__playwright__browser_click, mcp__playwright__browser_fill_form, mcp__playwright__browser_press_key, mcp__playwright__browser_wait_for, mcp__playwright__browser_close, mcp__playwright__browser_resize
---

# Record Attestor UI Demo

Record a comprehensive demo walkthrough of the Attestor UI — a screen-recorded video that explores **every page, every interactive element, every sub-menu, input field, and animation**. Then generate a timestamped voiceover script the user can read for narration.

## Prerequisites

1. Attestor UI must be running: `attestor ui --path <store>` on port 8080
2. Store should have 20+ seeded memories across multiple entities/namespaces
3. Node.js + `playwright` npm package installed (`npm install playwright`)

## Step 1 — Verify Server

```bash
curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:8080/ui/memories
```

If not 200, tell the user to start the server.

## Step 2 — Run the Recording Script

The recording script lives at `docs/demo/record-demo.js`. Run it:

```bash
node docs/demo/record-demo.js
```

Then rename the output:

```bash
# Find the auto-generated webm and rename
mv docs/demo/page@*.webm docs/demo/ui-walkthrough.webm
```

### What the Script Records (18 Screens)

The script navigates every page and interacts with every element. Here is the full sequence — if the script needs updating, follow this exact interaction order:

| # | Page | Interactions | Duration |
|---|------|-------------|----------|
| 01 | Evidence Board | Land, watch card stagger animation, scroll down and back | 5s |
| 02 | Search | Type "architecture" in search, watch HTMX live filter + highlights, scroll | 5s |
| 03 | Namespace filter | Clear search, type "default" in namespace field | 3s |
| 04 | Status filter | Switch to "superseded", then back to "active" | 4s |
| 05 | Clear filters | Navigate fresh to `/ui/memories` | 2s |
| 06 | Card hover | Hover 2 cards to show lift animation | 2s |
| 07 | Memory dossier | Click first card, scroll detail page | 4s |
| 08 | Dossier tabs | Click all 6 tabs: provenance → supersession → embedding → graph → access → content, scroll each | 12s |
| 09 | Knowledge graph | Land, click 2 nodes, show inspector panel | 7s |
| 10 | Graph controls | Search "Attestor", switch layouts (concentric → circle → cose), change sizing (pagerank → degree), fit view | 12s |
| 11 | Recall pipeline | Type query, set budget, execute, watch 5-layer animation, scroll results | 8s |
| 12 | Budget explorer | Click Budget Explorer, wait for 5-budget comparison, scroll chart | 8s |
| 13 | Timeline | Auto-load, scroll, click card → detail slide-in, close, filter superseded, reset | 12s |
| 14 | Agents | Auto-load, scroll, click namespace card → detail overlay, scroll inside, close | 10s |
| 15 | Health | Land, scroll to see all 4 component cards + percentiles + sparkline | 5s |
| 16 | Config | Land, scroll all config sections | 5s |
| 17 | Ops log | Land, scroll flight recorder table | 5s |
| 18 | Theme toggle | Switch dark → light, scroll in light mode, switch back to dark | 7s |

**Key recording principles:**
- **Always scroll** every page top-to-bottom and back — use `window.scrollTo({ behavior: 'smooth' })`
- **Slow type** into input fields (80ms delay per character) — looks natural
- **Pause 2-3s** on each screen for readability
- **Show hover states** on cards
- **Cycle through every option** in dropdowns (layouts, sizing, status)
- **Open and close** every modal/panel/overlay
- **Click every tab** in tabbed interfaces

### Script Structure

The recording script must use this pattern:

```javascript
const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: 'docs/demo/', size: { width: 1280, height: 800 } }
  });
  const page = await context.newPage();
  const base = 'http://127.0.0.1:8080';

  // Helper: smooth scroll
  async function scrollPage(ms = 1500) {
    await page.evaluate(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }));
    await page.waitForTimeout(ms);
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
    await page.waitForTimeout(800);
  }

  // Helper: slow type
  async function slowType(selector, text, delay = 80) {
    await page.click(selector);
    await page.waitForTimeout(300);
    await page.type(selector, text, { delay });
  }

  // ... 18 screens of interactions ...

  const videoPath = await page.video().path();
  await context.close();
  await browser.close();
  console.log('Demo video saved to:', videoPath);
})();
```

## Step 3 — Generate Voiceover Script

After recording, generate a timestamped voiceover script at `docs/demo/voiceover-script.md`.

Structure:
```markdown
# Attestor UI — Voiceover Script

**Total duration**: ~4 minutes
**Video**: `docs/demo/ui-walkthrough.webm`

---

## [0:00] Screen Name

Narration text describing what the viewer sees on screen.
Call out specific UI elements, animations, data, and interactions.

---

## [0:XX] Next Screen Name

...
```

**Narration guidelines:**
- Describe what's happening visually as it happens
- Name specific UI elements: "the confidence stamp", "the category bar", "the detail overlay"
- Call out numbers: "29 records", "P50 at 2.1ms", "5 layers"
- Explain WHY things work the way they do, not just WHAT they are
- End with the tagline: "Eight pages, 23 routes, zero LLM in the critical path."

## Step 4 — Take Updated Screenshots

After recording, optionally re-take the 12 static screenshots using Playwright MCP (navigate + screenshot) and save to `docs/demo/screens/`:

```
01-evidence-board.png    07-timeline.png
02-search-highlight.png  08-agents.png
03-memory-detail.png     09-health.png
04-graph.png             10-config.png
05-recall-pipeline.png   11-ops-log.png
06-budget-explorer.png   12-light-theme.png
```

## Step 5 — Verify and Report

- Confirm video exists and report file size
- Confirm voiceover script exists
- Open the video: `open docs/demo/ui-walkthrough.webm`
- List any pages that had issues (empty content, errors, missing data)

## Output Files

| File | Purpose |
|------|---------|
| `docs/demo/ui-walkthrough.webm` | Screen-recorded demo video |
| `docs/demo/voiceover-script.md` | Timestamped narration for voiceover |
| `docs/demo/record-demo.js` | Playwright recording script (reusable) |
| `docs/demo/screens/*.png` | Static screenshots (optional) |
