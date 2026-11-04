import type { CatalogOffer } from '../shared/types.ts';

export interface AuctionCandidate {
  offer: CatalogOffer;
  relevance: number;
  /** relevance x bid: a high bid cannot win without relevance, and relevance alone does not pay. */
  score: number;
}

export interface AuctionOptions {
  /** Offers below this relevance are never shown, whatever they bid. */
  minRelevance: number;
  maxResults: number;
  now?: Date;
}

/** The auction runs on the publisher's side, with the bids that came in the signed catalog. */
export function runLocalAuction(
  offers: CatalogOffer[],
  relevance: Map<string, number>,
  options: AuctionOptions,
): AuctionCandidate[] {
  const now = options.now ?? new Date();
  return offers
    .filter((offer) => !offer.expiresAt || new Date(offer.expiresAt) > now)
    .map((offer) => ({ offer, relevance: relevance.get(offer.id) ?? 0 }))
    .filter((c) => c.relevance >= options.minRelevance)
    .map((c) => ({ ...c, score: c.relevance * c.offer.bidCents }))
    .sort((a, b) => b.score - a.score)
    .slice(0, options.maxResults);
}
