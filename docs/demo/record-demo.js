// Demo video recorder — run with: npx playwright test docs/demo/record-demo.js
// Or: node docs/demo/record-demo.js (requires playwright installed)

const { chromium } = require('playwright');

(async () => {
  const browser = await chromium.launch();
  const context = await browser.newContext({
    viewport: { width: 1280, height: 800 },
    recordVideo: { dir: 'docs/demo/', size: { width: 1280, height: 800 } }
  });
  const page = await context.newPage();
  const base = 'http://127.0.0.1:8080';

  console.log('Recording demo...');

  // 01 — Evidence Board
  await page.goto(base + '/ui/memories');
  await page.waitForTimeout(3000);

  // 02 — Search with highlighting
  const searchInput = page.locator('input[type="search"]');
  if (await searchInput.count() > 0) {
    await searchInput.fill('architecture');
    await searchInput.press('Enter');
  }
  await page.waitForTimeout(2500);

  // 03 — Memory Detail
  const card = page.locator('.card').first();
  if (await card.count() > 0) {
    await card.click();
    await page.waitForTimeout(3000);
  }

  // 04 — Graph
  await page.goto(base + '/ui/graph');
  await page.waitForTimeout(3000);

  // 05 — Recall Pipeline
  await page.goto(base + '/ui/recall');
  await page.waitForTimeout(1000);
  await page.fill('#recall-query', 'how does the order service work?');
  await page.fill('#recall-budget', '2000');
  await page.click('#recall-go');
  await page.waitForTimeout(4000);

  // 06 — Budget Explorer
  await page.click('#budget-explore-btn');
  await page.waitForTimeout(5000);

  // Scroll down to see budget chart
  await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
  await page.waitForTimeout(2000);
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(1000);

  // 07 — Timeline
  await page.goto(base + '/ui/timeline');
  await page.waitForTimeout(1000);
  // Click Reconstruct to load timeline data (also auto-loads now)
  const tlGo = page.locator('#tl-go');
  if (await tlGo.count() > 0) await tlGo.click();
  await page.waitForTimeout(3000);

  // 08 — Agents
  await page.goto(base + '/ui/agents');
  // Wait for agents data to auto-load
  await page.waitForTimeout(3500);

  // 09 — Health
  await page.goto(base + '/ui/health');
  await page.waitForTimeout(3000);

  // 10 — Config
  await page.goto(base + '/ui/config');
  await page.waitForTimeout(2500);

  // 11 — Ops Log
  await page.goto(base + '/ui/ops');
  await page.waitForTimeout(3000);

  // 12 — Theme toggle
  await page.goto(base + '/ui/memories');
  await page.waitForTimeout(1500);
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'light'));
  await page.waitForTimeout(2500);
  await page.evaluate(() => document.documentElement.setAttribute('data-theme', 'dark'));
  await page.waitForTimeout(2000);

  // Finalize
  const videoPath = await page.video().path();
  await context.close();
  await browser.close();

  console.log('Demo video saved to:', videoPath);
})();
