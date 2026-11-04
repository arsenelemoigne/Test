import assert from 'node:assert/strict';
import { test } from 'node:test';
import { verifyCatalog } from '../sdk/catalog.ts';
import { signedClickUrl } from '../sdk/tracking.ts';
import { canonicalJson, hmacHex } from '../shared/crypto.ts';
import { clickParams, MARKETPLACE_URL, PUBLISHER, seededMarketplace } from './helpers.ts';

function click(offerId: string, secret = PUBLISHER.secret) {
  return clickParams(signedClickUrl({ marketplaceUrl: MARKETPLACE_URL, publisherId: PUBLISHER.id, publisherSecret: secret, offerId }));
}

test('catalog signature verifies, and a tampered bid is rejected', () => {
  const { market, publicKeyPem } = seededMarketplace();
  const signed = market.buildCatalog();
  assert.ok(verifyCatalog(signed, publicKeyPem).offers.length > 0);

  const tampered = structuredClone(signed);
  tampered.catalog.offers[0].bidCents = 1;
  assert.throws(() => verifyCatalog(tampered, publicKeyPem), /bad catalog signature/);
});

test('catalog never exposes advertiser ids or landing URLs', () => {
  const { market } = seededMarketplace();
  for (const offer of market.buildCatalog().catalog.offers) {
    assert.equal('url' in offer, false);
    assert.equal('advertiserId' in offer, false);
  }
});

test('expired catalogs are rejected', () => {
  const { market, publicKeyPem } = seededMarketplace();
  const signed = market.buildCatalog(new Date('2020-01-01T00:00:00Z'));
  assert.throws(() => verifyCatalog(signed, publicKeyPem), /expired/);
});

test('a valid click charges the bid and pays the publisher 70%', () => {
  const { market } = seededMarketplace();
  const result = market.recordClick(click('kestrel-vpn'));
  assert.deepEqual(result, { charged: true, redirectTo: 'https://example.com/kestrel-vpn', chargedCents: 120 });
  assert.equal(market.advertisers.get('kestrel')?.spentCents, 120);
  assert.equal(market.publishers.get(PUBLISHER.id)?.earnedCents, 84);
  assert.equal(market.stats().platformRevenueCents, 36);
});

test('replayed, forged and stale clicks still redirect but are not billed', () => {
  const { market } = seededMarketplace();
  const params = click('kestrel-vpn');
  market.recordClick(params);

  const replay = market.recordClick(params);
  assert.equal(replay.charged, false);
  assert.equal(replay.redirectTo, 'https://example.com/kestrel-vpn');
  assert.match(replay.charged ? '' : replay.reason, /replayed/);

  const forged = market.recordClick(click('kestrel-vpn', 'wrong-secret-0123456789'));
  assert.match(forged.charged ? '' : forged.reason, /bad signature/);

  const stale = market.recordClick(click('kestrel-vpn'), Date.now() + 2 * 60 * 60 * 1000);
  assert.match(stale.charged ? '' : stale.reason, /expired/);

  assert.equal(market.advertisers.get('kestrel')?.spentCents, 120);
});

test('an offer leaves the catalog once its budget cannot pay for another click', () => {
  const { market } = seededMarketplace();
  // Wayfare has a 100c budget and bids 35c: two clicks leave 30c.
  assert.equal(market.recordClick(click('wayfare-rail')).charged, true);
  assert.equal(market.recordClick(click('wayfare-rail')).charged, true);
  assert.equal(market.buildCatalog().catalog.offers.some((o) => o.id === 'wayfare-rail'), false);
  const third = market.recordClick(click('wayfare-rail'));
  assert.match(third.charged ? '' : third.reason, /budget exhausted/);
});

test('impression reports must be signed by the publisher', () => {
  const { market } = seededMarketplace();
  const report = { publisherId: PUBLISHER.id, periodStart: 'a', periodEnd: 'b', counts: { 'kestrel-vpn': 3, unknown: 5 } };
  assert.equal(market.recordImpressions(report, hmacHex('wrong-secret', canonicalJson(report))), false);
  assert.equal(market.recordImpressions(report, hmacHex(PUBLISHER.secret, canonicalJson(report))), true);
  assert.equal(market.impressions.get('kestrel-vpn'), 3);
  assert.equal(market.impressions.has('unknown'), false);
});
