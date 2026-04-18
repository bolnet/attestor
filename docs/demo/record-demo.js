// Comprehensive Attestor UI demo recorder
// Run: node docs/demo/record-demo.js
// Requires: npm install playwright

const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: 'docs/demo/', size: { width: 1280, height: 800 } }
  });
  const page = await context.newPage();
  const base = 'http://127.0.0.1:8080';

  // Helper: smooth scroll to bottom then back to top
  async function scrollPage(ms = 1500) {
    await page.evaluate(() => window.scrollTo({ top: document.body.scrollHeight, behavior: 'smooth' }));
    await page.waitForTimeout(ms);
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
    await page.waitForTimeout(800);
  }

  // Helper: slow type into a field
  async function slowType(selector, text, delay = 80) {
    await page.click(selector);
    await page.waitForTimeout(300);
    await page.type(selector, text, { delay });
  }

  console.log('Recording comprehensive demo...');

  // ================================================================
  // SCREEN 01 — Evidence Board (landing page)
  // ================================================================
  await page.goto(base + '/ui/memories');
  await page.waitForTimeout(2500); // let card animations complete
  // Scroll to see all cards
  await scrollPage(2000);
  await page.waitForTimeout(1000);

  // ================================================================
  // SCREEN 02 — Search with live filtering + highlighting
  // ================================================================
  await slowType('input[name="q"]', 'architecture');
  await page.waitForTimeout(2000); // HTMX live filter fires
  await scrollPage(1500);
  await page.waitForTimeout(1000);

  // ================================================================
  // SCREEN 03 — Filter by namespace
  // ================================================================
  // Clear search first
  await page.fill('input[name="q"]', '');
  await page.waitForTimeout(500);
  await slowType('input[name="namespace"]', 'default');
  await page.waitForTimeout(1500);

  // ================================================================
  // SCREEN 04 — Filter by status (superseded)
  // ================================================================
  await page.selectOption('select[name="status"]', 'superseded');
  await page.waitForTimeout(2000);
  // Reset to active
  await page.selectOption('select[name="status"]', 'active');
  await page.waitForTimeout(1500);

  // ================================================================
  // SCREEN 05 — Clear filters
  // ================================================================
  await page.goto(base + '/ui/memories');
  await page.waitForTimeout(2000);

  // ================================================================
  // SCREEN 06 — Hover cards (show lift animation)
  // ================================================================
  const cards = page.locator('.card');
  const cardCount = await cards.count();
  if (cardCount > 0) {
    await cards.nth(0).hover();
    await page.waitForTimeout(800);
    if (cardCount > 1) {
      await cards.nth(1).hover();
      await page.waitForTimeout(800);
    }
  }

  // ================================================================
  // SCREEN 07 — Memory Detail / Dossier
  // ================================================================
  if (cardCount > 0) {
    await cards.nth(0).click();
    await page.waitForTimeout(2500);
    await scrollPage(1500);
  }

  // ================================================================
  // SCREEN 08 — Dossier tabs: cycle through all 6
  // ================================================================
  const tabNames = ['provenance', 'supersession', 'embedding', 'graph', 'access', 'content'];
  for (const tab of tabNames) {
    const tabLink = page.locator(`.tabs__item[href*="tab=${tab}"]`);
    if (await tabLink.count() > 0) {
      await tabLink.click();
      await page.waitForTimeout(1800);
      await scrollPage(1200);
    }
  }
  await page.waitForTimeout(1000);

  // ================================================================
  // SCREEN 09 — Knowledge Graph
  // ================================================================
  await page.goto(base + '/ui/graph');
  await page.waitForTimeout(3500); // let force layout settle

  // Click a node (first one we can find)
  const graphCanvas = page.locator('#cy');
  if (await graphCanvas.count() > 0) {
    // Click near center to hit a node
    const box = await graphCanvas.boundingBox();
    if (box) {
      await page.mouse.click(box.x + box.width * 0.4, box.y + box.height * 0.4);
      await page.waitForTimeout(2000);
      // Click another area
      await page.mouse.click(box.x + box.width * 0.6, box.y + box.height * 0.6);
      await page.waitForTimeout(1500);
    }
  }

  // Search for an entity in graph
  const graphSearch = page.locator('#graph-search');
  if (await graphSearch.count() > 0) {
    await slowType('#graph-search', 'Attestor');
    await page.waitForTimeout(2000);
    // Clear
    await page.fill('#graph-search', '');
    await page.waitForTimeout(1000);
  }

  // Change layout
  const layoutSelect = page.locator('#graph-layout');
  if (await layoutSelect.count() > 0) {
    await page.selectOption('#graph-layout', 'concentric');
    await page.waitForTimeout(2000);
    await page.selectOption('#graph-layout', 'circle');
    await page.waitForTimeout(2000);
    await page.selectOption('#graph-layout', 'cose');
    await page.waitForTimeout(2000);
  }

  // Change node sizing
  const sizingSelect = page.locator('#graph-sizing');
  if (await sizingSelect.count() > 0) {
    await page.selectOption('#graph-sizing', 'pagerank');
    await page.waitForTimeout(1500);
    await page.selectOption('#graph-sizing', 'degree');
    await page.waitForTimeout(1500);
  }

  // Fit view
  const fitBtn = page.locator('#graph-fit');
  if (await fitBtn.count() > 0) {
    await fitBtn.click();
    await page.waitForTimeout(1500);
  }

  // ================================================================
  // SCREEN 10 — Recall Pipeline
  // ================================================================
  await page.goto(base + '/ui/recall');
  await page.waitForTimeout(1500);

  // Fill query
  await slowType('#recall-query', 'how does the order service work?');
  await page.waitForTimeout(500);

  // Set budget
  await page.fill('#recall-budget', '2000');
  await page.waitForTimeout(500);

  // Execute recall
  await page.click('#recall-go');
  await page.waitForTimeout(4000); // let pipeline animation play

  // Scroll to see all layers
  await scrollPage(2500);
  await page.waitForTimeout(1500);

  // ================================================================
  // SCREEN 11 — Budget Explorer
  // ================================================================
  const budgetBtn = page.locator('#budget-explore-btn');
  if (await budgetBtn.count() > 0) {
    await budgetBtn.click();
    await page.waitForTimeout(5000); // runs 5 budget levels

    // Scroll to see the chart
    await page.evaluate(() => {
      const el = document.getElementById('budget-explore-area');
      if (el) el.scrollIntoView({ behavior: 'smooth' });
    });
    await page.waitForTimeout(2500);
    await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
    await page.waitForTimeout(1000);
  }

  // ================================================================
  // SCREEN 12 — Timeline
  // ================================================================
  await page.goto(base + '/ui/timeline');
  await page.waitForTimeout(3000); // auto-loads

  // Scroll the timeline
  await scrollPage(2500);
  await page.waitForTimeout(1000);

  // Click a timeline card to open detail panel
  const tlCard = page.locator('.tl-card').first();
  if (await tlCard.count() > 0) {
    await tlCard.click();
    await page.waitForTimeout(2500);

    // Close detail
    const tlClose = page.locator('#tl-detail-close');
    if (await tlClose.count() > 0) {
      await tlClose.click();
      await page.waitForTimeout(1000);
    }
  }

  // Filter by status: superseded
  await page.selectOption('#tl-status', 'superseded');
  await page.click('#tl-go');
  await page.waitForTimeout(2500);
  await scrollPage(1500);

  // Reset to all
  await page.selectOption('#tl-status', 'all');
  await page.click('#tl-go');
  await page.waitForTimeout(2000);

  // ================================================================
  // SCREEN 13 — Agents
  // ================================================================
  await page.goto(base + '/ui/agents');
  await page.waitForTimeout(3000); // auto-loads

  // Scroll to see agent cards
  await scrollPage(1500);

  // Click an agent card to open detail overlay
  const agentCard = page.locator('.agents-card').first();
  if (await agentCard.count() > 0) {
    await agentCard.click();
    await page.waitForTimeout(3000);

    // Scroll inside the detail panel
    await page.evaluate(() => {
      const panel = document.getElementById('agents-detail');
      if (panel) panel.scrollTo({ top: panel.scrollHeight, behavior: 'smooth' });
    });
    await page.waitForTimeout(2000);
    await page.evaluate(() => {
      const panel = document.getElementById('agents-detail');
      if (panel) panel.scrollTo({ top: 0, behavior: 'smooth' });
    });
    await page.waitForTimeout(1000);

    // Close overlay
    const closeBtn = page.locator('#agents-detail-close');
    if (await closeBtn.count() > 0) {
      await closeBtn.click();
      await page.waitForTimeout(1000);
    }
  }

  // ================================================================
  // SCREEN 14 — System Health
  // ================================================================
  await page.goto(base + '/ui/health');
  await page.waitForTimeout(3000);

  // Scroll to see all health cards
  await scrollPage(2000);
  await page.waitForTimeout(1500);

  // ================================================================
  // SCREEN 15 — Configuration
  // ================================================================
  await page.goto(base + '/ui/config');
  await page.waitForTimeout(2500);

  // Scroll to see all config sections
  await scrollPage(2500);
  await page.waitForTimeout(1000);

  // ================================================================
  // SCREEN 16 — Operations Log
  // ================================================================
  await page.goto(base + '/ui/ops');
  await page.waitForTimeout(3000);

  // Scroll to see ops table
  await scrollPage(2000);
  await page.waitForTimeout(2000); // let auto-refresh tick

  // ================================================================
  // SCREEN 17 — Theme toggle (dark → light → dark)
  // ================================================================
  await page.goto(base + '/ui/memories');
  await page.waitForTimeout(2000);

  // Toggle to light
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
  await page.waitForTimeout(2500);

  // Scroll in light mode
  await scrollPage(1500);

  // Toggle back to dark
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  await page.waitForTimeout(2500);

  // ================================================================
  // SCREEN 18 — Export (show sidebar buttons)
  // ================================================================
  // Scroll down to show export buttons if visible
  await page.evaluate(() => {
    const btn = document.querySelector('a[href*="export"]');
    if (btn) btn.scrollIntoView({ behavior: 'smooth' });
  });
  await page.waitForTimeout(2000);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'smooth' }));
  await page.waitForTimeout(2000);

  // ================================================================
  // Finalize
  // ================================================================
  const videoPath = await page.video().path();
  await context.close();
  await browser.close();

  console.log('Demo video saved to:', videoPath);
  console.log('Rename with: mv "' + videoPath + '" docs/demo/ui-walkthrough.webm');
})();
