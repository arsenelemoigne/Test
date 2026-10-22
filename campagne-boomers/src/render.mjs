import { createRequire } from 'module'; const { chromium } = createRequire(process.cwd() + '/')('playwright');
import fs from 'fs'; import path from 'path';
const root = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const scale = Number(process.argv[2] || 1);
fs.mkdirSync(`${root}/png`, { recursive: true });
const b = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium-1194/chrome-linux/chrome' }).catch(() => chromium.launch());
const p = await b.newPage({ viewport: { width: 2000, height: 1500 }, deviceScaleFactor: scale });
for (const f of fs.readdirSync(`${root}/html`).filter(f => f.endsWith('.html')).sort()) {
  await p.goto(`file://${root}/html/${f}`); await p.evaluate(() => document.fonts.ready);
  await p.screenshot({ path: `${root}/png/${f.replace('.html', '.png')}` });
}
await b.close(); console.log('rendered');
