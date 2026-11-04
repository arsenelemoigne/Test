/** An offer as an advertiser submits it to the marketplace. */
export interface Offer {
  id: string;
  advertiserId: string;
  brand: string;
  /** Headline approved by the advertiser. */
  title: string;
  /** Claims the advertiser approved. Publishers and models may only repeat these. */
  claims: string[];
  /** Targeting terms used by the local relevance scorer. */
  keywords: string[];
  /** Dotted categories such as "sports.running". */
  categories: string[];
  /** Landing page. Users only reach it through the marketplace click redirect. */
  url: string;
  promoCode?: string;
  /** Most the advertiser pays for one click, in euro cents. */
  bidCents: number;
  /** ISO date after which the offer must not be shown. */
  expiresAt?: string;
}

/** What publishers receive: no advertiser id, no landing URL (clicks go through the redirect). */
export type CatalogOffer = Omit<Offer, 'advertiserId' | 'url'>;

export interface Catalog {
  version: number;
  issuedAt: string;
  /** Publishers must stop using a catalog after this time. */
  expiresAt: string;
  offers: CatalogOffer[];
}

/** The catalog plus an Ed25519 signature, so publishers cannot tamper with bids. */
export interface SignedCatalog {
  catalog: Catalog;
  signature: string;
}

/** Aggregated counts only: no user, no conversation, no timestamps per view. */
export interface ImpressionReport {
  publisherId: string;
  periodStart: string;
  periodEnd: string;
  counts: Record<string, number>;
}

/** An offer ready to render in a chat app, always labeled. */
export interface SponsoredOffer {
  label: 'Sponsored';
  id: string;
  brand: string;
  title: string;
  claims: string[];
  promoCode?: string;
  clickUrl: string;
  relevance: number;
}

export interface ClickParams {
  offerId: string;
  publisherId: string;
  /** Unix seconds when the link was created. */
  ts: string;
  nonce: string;
}
