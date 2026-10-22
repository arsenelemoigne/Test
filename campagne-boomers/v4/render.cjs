const { chromium } = require('/opt/node-tools/node_modules/playwright');
const fs = require('fs');
(async () => {
  const scale = Number(process.argv[2] || 1);
  const b = await chromium.launch();
  fs.mkdirSync(__dirname + '/png', { recursive: true });
  for (const f of fs.readdirSync(__dirname).filter(f => f.endsWith('.html'))) {
    const story = f.startsWith('story');
    const p = await b.newPage({ viewport: story ? { width: 1080, height: 1920 } : { width: 2000, height: 1500 }, deviceScaleFactor: scale * (story ? 2 : 1) });
    await p.goto('file://' + __dirname + '/' + f); await p.evaluate(() => document.fonts.ready);
    await p.screenshot({ path: __dirname + '/png/' + f.replace('.html', '.png') }); await p.close();
  }
  await b.close(); console.log('rendered');
})();
