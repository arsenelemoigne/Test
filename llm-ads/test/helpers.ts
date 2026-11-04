import { seedMarketplace } from '../marketplace/seed.ts';
import { Marketplace } from '../marketplace/store.ts';
import { generateCatalogKeys } from '../shared/crypto.ts';

export const PUBLISHER = { id: 'demo-chat', secret: 'test-secret-0123456789' };
export const MARKETPLACE_URL = 'http://marketplace.test';

export function seededMarketplace() {
  const keys = generateCatalogKeys();
  const market = new Marketplace(keys.privateKeyPem);
  seedMarketplace(market, PUBLISHER);
  return { market, publicKeyPem: keys.publicKeyPem };
}

/** A fetch that serves the marketplace in-process, so SDK tests need no network. */
export function inProcessFetch(market: Marketplace): typeof fetch {
  return (async (input: string | URL | Request) => {
    const url = new URL(input instanceof Request ? input.url : input);
    if (url.pathname === '/catalog') return Response.json(market.buildCatalog());
    return new Response('not found', { status: 404 });
  }) as typeof fetch;
}

/** Turns a signed click URL back into the parameters the marketplace receives. */
export function clickParams(clickUrl: string) {
  const url = new URL(clickUrl);
  return {
    offerId: decodeURIComponent(url.pathname.split('/').pop() ?? ''),
    publisherId: url.searchParams.get('pub') ?? '',
    ts: url.searchParams.get('ts') ?? '',
    nonce: url.searchParams.get('n') ?? '',
    sig: url.searchParams.get('sig') ?? '',
  };
}
