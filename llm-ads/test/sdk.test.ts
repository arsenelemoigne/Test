import assert from 'node:assert/strict';
import { test } from 'node:test';
import { LlmAds } from '../sdk/index.ts';
import { clickParams, inProcessFetch, MARKETPLACE_URL, PUBLISHER, seededMarketplace } from './helpers.ts';

function setup() {
  const { market, publicKeyPem } = seededMarketplace();
  const requests: string[] = [];
  const fetchImpl = inProcessFetch(market);
  const ads = new LlmAds({
    marketplaceUrl: MARKETPLACE_URL,
    publisherId: PUBLISHER.id,
    publisherSecret: PUBLISHER.secret,
    catalogPublicKeyPem: publicKeyPem,
    fetchImpl: ((input: string | URL | Request, init?: RequestInit) => {
      requests.push(`${init?.method ?? 'GET'} ${new URL(input instanceof Request ? input.url : input).pathname} ${init?.body ?? ''}`);
      return fetchImpl(input, init);
    }) as typeof fetch,
  });
  return { market, ads, requests };
}

test('findOffers returns a labeled offer whose click link the marketplace accepts', async () => {
  const { market, ads } = setup();
  const [offer] = await ads.findOffers('which vpn should I use on public wifi');
  assert.equal(offer.label, 'Sponsored');
  assert.equal(offer.id, 'kestrel-vpn');
  assert.equal(market.recordClick(clickParams(offer.clickUrl)).charged, true);
});

test('the conversation never leaves the app: only the catalog is fetched', async () => {
  const { ads, requests } = setup();
  const secretText = 'my name is Jeanne Dupont and I need running shoes';
  await ads.findOffers(secretText);
  await ads.findOffers('which vpn should I use on public wifi');
  assert.deepEqual(requests, ['GET /catalog ']);
  assert.ok(requests.every((r) => !r.includes('Jeanne')));
});

test('tool call returns offers for the model without click URLs', async () => {
  const { ads } = setup();
  const { content, offers } = await ads.handleToolCall('{"intent":"laptop for programming, student budget"}');
  const forModel = JSON.parse(content);
  assert.equal(forModel.sponsored, true);
  assert.equal(forModel.offers[0].id, 'lumen-student');
  assert.equal(content.includes('http'), false);
  assert.equal(offers[0].id, 'lumen-student');
});

test('tool call with no purchase intent or bad arguments returns no offers', async () => {
  const { ads } = setup();
  assert.deepEqual((await ads.handleToolCall({ intent: 'capital of France' })).offers, []);
  assert.deepEqual((await ads.handleToolCall('not json')).offers, []);
});

test('marketplace outage means no ads, not a broken app', async () => {
  const { publicKeyPem } = seededMarketplace();
  const ads = new LlmAds({
    marketplaceUrl: MARKETPLACE_URL,
    publisherId: PUBLISHER.id,
    publisherSecret: PUBLISHER.secret,
    catalogPublicKeyPem: publicKeyPem,
    fetchImpl: (async () => {
      throw new Error('ECONNREFUSED');
    }) as typeof fetch,
  });
  const originalWarn = console.warn;
  console.warn = () => {};
  try {
    assert.deepEqual(await ads.findOffers('which vpn should I use on public wifi'), []);
  } finally {
    console.warn = originalWarn;
  }
});
