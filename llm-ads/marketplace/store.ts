import { randomUUID } from 'node:crypto';
import { canonicalJson, clickMessage, hmacHex, ID_PATTERN, safeEqualHex, signPayload } from '../shared/crypto.ts';
import type { Catalog, CatalogOffer, ClickParams, ImpressionReport, Offer, SignedCatalog } from '../shared/types.ts';

/** Share of each click price paid to the publisher (the chat app). The rest is the platform's cut. */
export const PUBLISHER_SHARE = 0.7;
/** How long a signed click link stays billable after the publisher created it. */
export const CLICK_LINK_TTL_SECONDS = 60 * 60;
/** How long publishers may keep using a catalog before fetching a new one. */
export const CATALOG_TTL_SECONDS = 15 * 60;

export interface Advertiser {
  id: string;
  name: string;
  budgetCents: number;
  spentCents: number;
}

export interface Publisher {
  id: string;
  name: string;
  /** Shared secret used to sign click links and impression reports. */
  secret: string;
  earnedCents: number;
}

export interface ClickRecord {
  id: string;
  offerId: string;
  publisherId: string;
  chargedCents: number;
  at: string;
}

export type ClickResult =
  | { charged: true; redirectTo: string; chargedCents: number }
  | { charged: false; redirectTo: string | null; reason: string };

/**
 * The bidding marketplace. Advertisers set offers, bids and budgets here.
 * It never receives a conversation: publishers pick offers locally and only
 * report clicks and aggregated impression counts.
 */
export class Marketplace {
  readonly advertisers = new Map<string, Advertiser>();
  readonly offers = new Map<string, Offer>();
  readonly publishers = new Map<string, Publisher>();
  readonly clicks: ClickRecord[] = [];
  readonly impressions = new Map<string, number>();
  /** "publisherId:nonce" -> unix second after which it can be forgotten. */
  private readonly seenNonces = new Map<string, number>();
  private readonly privateKeyPem: string;
  private version = 1;

  constructor(privateKeyPem: string) {
    this.privateKeyPem = privateKeyPem;
  }

  addAdvertiser(input: { id: string; name: string; budgetCents: number }): Advertiser {
    assertId(input.id);
    assertPositiveInt(input.budgetCents, 'budgetCents');
    const advertiser = { ...input, spentCents: 0 };
    this.advertisers.set(advertiser.id, advertiser);
    this.version++;
    return advertiser;
  }

  addOffer(offer: Offer): Offer {
    assertId(offer.id);
    if (!this.advertisers.has(offer.advertiserId)) throw new Error(`unknown advertiser ${offer.advertiserId}`);
    assertPositiveInt(offer.bidCents, 'bidCents');
    if (!/^https?:\/\//.test(offer.url)) throw new Error('url must be http(s)');
    if (!offer.brand || !offer.title || offer.keywords.length === 0) throw new Error('brand, title and keywords are required');
    this.offers.set(offer.id, offer);
    this.version++;
    return offer;
  }

  addPublisher(input: { id: string; name: string; secret: string }): Publisher {
    assertId(input.id);
    if (input.secret.length < 16) throw new Error('publisher secret must be at least 16 characters');
    const publisher = { ...input, earnedCents: 0 };
    this.publishers.set(publisher.id, publisher);
    return publisher;
  }

  /** Offers that are live and whose advertiser can still pay for at least one click. */
  activeOffers(now = new Date()): Offer[] {
    return [...this.offers.values()].filter((offer) => {
      if (offer.expiresAt && new Date(offer.expiresAt) <= now) return false;
      const advertiser = this.advertisers.get(offer.advertiserId);
      return !!advertiser && advertiser.budgetCents - advertiser.spentCents >= offer.bidCents;
    });
  }

  /** The signed catalog every publisher downloads. Bids travel with it; the auction runs on the publisher's side. */
  buildCatalog(now = new Date()): SignedCatalog {
    const offers: CatalogOffer[] = this.activeOffers(now).map(({ advertiserId: _advertiserId, url: _url, ...rest }) => rest);
    const catalog: Catalog = {
      version: this.version,
      issuedAt: now.toISOString(),
      expiresAt: new Date(now.getTime() + CATALOG_TTL_SECONDS * 1000).toISOString(),
      offers,
    };
    return { catalog, signature: signPayload(catalog, this.privateKeyPem) };
  }

  /**
   * Handles a click on a signed link. The user is always sent to the landing page when the
   * offer exists; we only refuse to bill. Billing is first-price CPC: the advertiser's bid.
   */
  recordClick(params: ClickParams & { sig: string }, nowMs = Date.now()): ClickResult {
    const offer = this.offers.get(params.offerId);
    if (!offer) return { charged: false, redirectTo: null, reason: 'unknown offer' };
    const redirectTo = offer.url;

    const publisher = this.publishers.get(params.publisherId);
    if (!publisher) return { charged: false, redirectTo, reason: 'unknown publisher' };

    const expected = hmacHex(publisher.secret, clickMessage(params));
    if (!safeEqualHex(expected, params.sig)) return { charged: false, redirectTo, reason: 'bad signature' };

    const nowS = Math.floor(nowMs / 1000);
    const ts = Number(params.ts);
    if (!Number.isInteger(ts) || ts > nowS + 60 || nowS - ts > CLICK_LINK_TTL_SECONDS) {
      return { charged: false, redirectTo, reason: 'expired link' };
    }

    this.forgetOldNonces(nowS);
    const nonceKey = `${publisher.id}:${params.nonce}`;
    if (this.seenNonces.has(nonceKey)) return { charged: false, redirectTo, reason: 'replayed click' };
    this.seenNonces.set(nonceKey, ts + CLICK_LINK_TTL_SECONDS);

    const advertiser = this.advertisers.get(offer.advertiserId);
    if (!advertiser || advertiser.budgetCents - advertiser.spentCents < offer.bidCents) {
      return { charged: false, redirectTo, reason: 'budget exhausted' };
    }

    const price = offer.bidCents;
    advertiser.spentCents += price;
    publisher.earnedCents += Math.floor(price * PUBLISHER_SHARE);
    this.clicks.push({
      id: randomUUID(),
      offerId: offer.id,
      publisherId: publisher.id,
      chargedCents: price,
      at: new Date(nowMs).toISOString(),
    });
    // The active set may have changed (budget ran out), so bump the catalog version.
    if (advertiser.budgetCents - advertiser.spentCents < offer.bidCents) this.version++;
    return { charged: true, redirectTo, chargedCents: price };
  }

  /** Accepts aggregated view counts signed by the publisher. Reporting only, never billed. */
  recordImpressions(report: ImpressionReport, signatureHex: string): boolean {
    const publisher = this.publishers.get(report.publisherId);
    if (!publisher) return false;
    if (!safeEqualHex(hmacHex(publisher.secret, canonicalJson(report)), signatureHex)) return false;
    for (const [offerId, count] of Object.entries(report.counts ?? {})) {
      if (!this.offers.has(offerId) || !Number.isInteger(count) || count <= 0 || count > 1_000_000) continue;
      this.impressions.set(offerId, (this.impressions.get(offerId) ?? 0) + count);
    }
    return true;
  }

  stats() {
    const charged = this.clicks.reduce((sum, c) => sum + c.chargedCents, 0);
    const paidOut = [...this.publishers.values()].reduce((sum, p) => sum + p.earnedCents, 0);
    return {
      catalogVersion: this.version,
      platformRevenueCents: charged - paidOut,
      advertisers: [...this.advertisers.values()],
      publishers: [...this.publishers.values()].map(({ secret: _secret, ...rest }) => rest),
      offers: [...this.offers.values()].map((o) => ({
        id: o.id,
        brand: o.brand,
        bidCents: o.bidCents,
        impressions: this.impressions.get(o.id) ?? 0,
        clicks: this.clicks.filter((c) => c.offerId === o.id).length,
      })),
    };
  }

  private forgetOldNonces(nowS: number): void {
    for (const [key, expiry] of this.seenNonces) if (expiry < nowS) this.seenNonces.delete(key);
  }
}

function assertId(id: string): void {
  if (!ID_PATTERN.test(id)) throw new Error(`invalid id "${id}": use lowercase letters, digits and dashes`);
}

function assertPositiveInt(value: number, name: string): void {
  if (!Number.isInteger(value) || value <= 0) throw new Error(`${name} must be a positive integer`);
}
