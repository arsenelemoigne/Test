import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runLocalAuction } from '../sdk/auction.ts';
import { LexicalScorer } from '../sdk/relevance.ts';
import { seededMarketplace } from './helpers.ts';

async function auction(text: string, minRelevance = 0.25) {
  const { market } = seededMarketplace();
  const { catalog } = market.buildCatalog();
  const scorer = new LexicalScorer();
  await scorer.index(catalog.offers);
  return runLocalAuction(catalog.offers, await scorer.score(text), { minRelevance, maxResults: 3 });
}

test('a very relevant offer beats a higher bid that fits less well', async () => {
  const [winner, runnerUp] = await auction('I need running shoes, I have flat feet');
  assert.equal(winner.offer.id, 'stride-stability'); // bids 45c
  assert.equal(runnerUp.offer.id, 'trailhead-trail'); // bids 60c
});

test('with similar relevance, the higher bid wins', async () => {
  const [winner] = await auction('running shoes');
  assert.equal(winner.offer.id, 'trailhead-trail');
});

test('no offer when nothing is being bought', async () => {
  assert.deepEqual(await auction('What is the capital of France?'), []);
  assert.deepEqual(await auction('how do I fix a null pointer exception in java'), []);
  assert.deepEqual(await auction("I'm running late for my meeting"), []);
});

test('expired offers never win', async () => {
  const { market } = seededMarketplace();
  const { catalog } = market.buildCatalog();
  const offers = catalog.offers.map((o) => (o.id === 'kestrel-vpn' ? { ...o, expiresAt: '2000-01-01T00:00:00Z' } : o));
  const scorer = new LexicalScorer();
  await scorer.index(offers);
  const result = runLocalAuction(offers, await scorer.score('which vpn for public wifi'), { minRelevance: 0.25, maxResults: 3 });
  assert.deepEqual(result, []);
});
