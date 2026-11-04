import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import { generateCatalogKeys, publicKeyFromPrivate } from '../shared/crypto.ts';
import type { ImpressionReport, Offer } from '../shared/types.ts';
import { seedMarketplace } from './seed.ts';
import { Marketplace } from './store.ts';

const PORT = Number(process.env.MARKETPLACE_PORT ?? 4000);
const DEMO_PUBLISHER_ID = process.env.PUBLISHER_ID ?? 'demo-chat';
const DEMO_PUBLISHER_SECRET = process.env.PUBLISHER_SECRET ?? 'dev-secret-change-me-please';
const MAX_BODY_BYTES = 64 * 1024;

// In production the private key comes from a secret store and the public key is pinned in the SDK.
const privateKeyPem = process.env.CATALOG_PRIVATE_KEY_PEM ?? generateCatalogKeys().privateKeyPem;
const publicKeyPem = publicKeyFromPrivate(privateKeyPem);

const market = new Marketplace(privateKeyPem);
seedMarketplace(market, { id: DEMO_PUBLISHER_ID, secret: DEMO_PUBLISHER_SECRET });

const server = createServer((req, res) => {
  handle(req, res).catch((err: unknown) => {
    console.error(err);
    if (!res.headersSent) send(res, 500, { error: 'internal error' });
  });
});

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', `http://${req.headers.host ?? 'localhost'}`);
  const path = url.pathname;

  if (req.method === 'GET' && path === '/catalog') return send(res, 200, market.buildCatalog());
  if (req.method === 'GET' && path === '/catalog/public-key') return sendText(res, 200, publicKeyPem);
  if (req.method === 'GET' && path === '/stats') return send(res, 200, market.stats());

  const click = path.match(/^\/c\/([^/]+)$/);
  if (req.method === 'GET' && click) {
    const q = url.searchParams;
    const result = market.recordClick({
      offerId: decodeURIComponent(click[1]),
      publisherId: q.get('pub') ?? '',
      ts: q.get('ts') ?? '',
      nonce: q.get('n') ?? '',
      sig: q.get('sig') ?? '',
    });
    if (!result.redirectTo) return send(res, 404, { error: 'unknown offer' });
    console.log(result.charged ? `click ${click[1]} charged ${result.chargedCents}c` : `click ${click[1]} not billed: ${result.reason}`);
    res.writeHead(302, { location: result.redirectTo, 'cache-control': 'no-store' });
    res.end();
    return;
  }

  if (req.method === 'POST' && path === '/impressions') {
    const report = (await readJson(req)) as ImpressionReport;
    const ok = market.recordImpressions(report, String(req.headers['x-signature'] ?? ''));
    res.writeHead(ok ? 204 : 401);
    res.end();
    return;
  }

  // Prototype only: no advertiser authentication or review yet.
  if (req.method === 'POST' && path === '/offers') {
    try {
      return send(res, 201, market.addOffer((await readJson(req)) as Offer));
    } catch (err) {
      return send(res, 400, { error: err instanceof Error ? err.message : 'invalid offer' });
    }
  }

  send(res, 404, { error: 'not found' });
}

function send(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body, null, 2));
}

function sendText(res: ServerResponse, status: number, body: string): void {
  res.writeHead(status, { 'content-type': 'text/plain' });
  res.end(body);
}

async function readJson(req: IncomingMessage): Promise<unknown> {
  let size = 0;
  const chunks: Buffer[] = [];
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > MAX_BODY_BYTES) throw new Error('body too large');
    chunks.push(chunk as Buffer);
  }
  return JSON.parse(Buffer.concat(chunks).toString('utf8') || 'null');
}

server.listen(PORT, () => {
  console.log(`marketplace listening on http://localhost:${PORT}`);
  console.log(`  GET /catalog  GET /catalog/public-key  GET /c/:offerId  POST /impressions  POST /offers  GET /stats`);
});
