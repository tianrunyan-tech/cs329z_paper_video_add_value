// Renders findings.html to finding_1.png and finding_2.png. Requires Playwright with Chromium.
const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1400, height: 900 }, deviceScaleFactor: 2 });
  for (const f of [1, 2]) {
    await page.goto('file://' + path.join(__dirname, 'findings.html') + '?f=' + f);
    await page.waitForTimeout(300);
    await (await page.$('#fig')).screenshot({ path: path.join(__dirname, `finding_${f}.png`) });
  }
  await browser.close();
})();
