import type { Catalog, SponsoredOffer } from '../shared/types.ts';
import { runLocalAuction } from './auction.ts';
import { CatalogClient } from './catalog.ts';
import { LexicalScorer, type RelevanceScorer } from './relevance.ts';
import { ImpressionCounter, signedClickUrl } from './tracking.ts';

export { runLocalAuction } from './auction.ts';
export { CatalogClient, verifyCatalog } from './catalog.ts';
export { EmbeddingScorer, LexicalScorer, type RelevanceScorer } from './relevance.ts';
export { ImpressionCounter, signedClickUrl } from './tracking.ts';

export const OFFERS_TOOL_NAME = 'get_sponsored_offers';

/** Tool definition in the OpenAI-compatible format (Ollama, vLLM, OpenRouter, Mistral, OpenAI...). */
export const offersToolDefinition = {
  type: 'function',
  function: {
    name: OFFERS_TOOL_NAME,
    description:
      'Look up sponsored offers for something the user is clearly considering buying. ' +
      'Only call this when the user shows purchase intent. Results are paid placements: ' +
      'they are shown to the user as separate cards labeled "Sponsored". If you mention one, say it is sponsored, ' +
      'use only the listed claims, and never present it as your own independent recommendation.',
    parameters: {
      type: 'object',
      properties: {
        intent: {
          type: 'string',
          description: 'What the user wants to buy, in a few words. No names, emails or other personal details.',
        },
      },
      required: ['intent'],
      additionalProperties: false,
    },
  },
} as const;

export interface LlmAdsOptions {
  marketplaceUrl: string;
  publisherId: string;
  publisherSecret: string;
  catalogPublicKeyPem: string;
  scorer?: RelevanceScorer;
  /** Default 0.25. Raise it to show fewer, more relevant offers. */
  minRelevance?: number;
  refreshMs?: number;
  fetchImpl?: typeof fetch;
}

/**
 * What a publisher (any app running an LLM) adds to its stack.
 * The conversation stays in this process: the catalog comes in, only clicks and view counts go out.
 */
export class LlmAds {
  private readonly options: LlmAdsOptions;
  private readonly catalog: CatalogClient;
  private readonly scorer: RelevanceScorer;
  private readonly impressions: ImpressionCounter;
  private indexed: Catalog | null = null;

  constructor(options: LlmAdsOptions) {
    this.options = options;
    this.catalog = new CatalogClient({
      marketplaceUrl: options.marketplaceUrl,
      publicKeyPem: options.catalogPublicKeyPem,
      refreshMs: options.refreshMs,
      fetchImpl: options.fetchImpl,
    });
    this.scorer = options.scorer ?? new LexicalScorer();
    this.impressions = new ImpressionCounter(options);
  }

  /** Runs the auction locally on `text`. Returns at most `max` labeled offers, often none. */
  async findOffers(text: string, max = 1): Promise<SponsoredOffer[]> {
    const catalog = await this.catalog.get();
    if (!catalog || !text.trim()) return [];
    if (catalog !== this.indexed) {
      await this.scorer.index(catalog.offers);
      this.indexed = catalog;
    }
    const relevance = await this.scorer.score(text);
    const winners = runLocalAuction(catalog.offers, relevance, {
      minRelevance: this.options.minRelevance ?? 0.25,
      maxResults: max,
    });
    return winners.map(({ offer, relevance: r }) => ({
      label: 'Sponsored',
      id: offer.id,
      brand: offer.brand,
      title: offer.title,
      claims: offer.claims,
      promoCode: offer.promoCode,
      relevance: Math.round(r * 100) / 100,
      clickUrl: signedClickUrl({
        marketplaceUrl: this.options.marketplaceUrl,
        publisherId: this.options.publisherId,
        publisherSecret: this.options.publisherSecret,
        offerId: offer.id,
      }),
    }));
  }

  /** Call when an offer is actually displayed. Only the per-offer total is ever reported. */
  markShown(offerId: string): void {
    this.impressions.record(offerId);
  }

  flushImpressions(): Promise<void> {
    return this.impressions.flush();
  }

  /**
   * Handles a `get_sponsored_offers` tool call. `content` goes back to the model,
   * `offers` are for your UI to render as cards (the model never sees click URLs).
   */
  async handleToolCall(rawArguments: unknown): Promise<{ content: string; offers: SponsoredOffer[] }> {
    const args = typeof rawArguments === 'string' ? safeParse(rawArguments) : rawArguments;
    const intent = typeof (args as { intent?: unknown })?.intent === 'string' ? (args as { intent: string }).intent.slice(0, 200) : '';
    const offers = await this.findOffers(intent, 3);
    const content = JSON.stringify({
      sponsored: true,
      disclosure: 'Paid placements, shown to the user as cards labeled "Sponsored". If you mention one, say it is sponsored and use only the listed claims.',
      offers: offers.map(({ id, brand, title, claims, promoCode }) => ({ id, brand, title, claims, promoCode })),
    });
    return { content, offers };
  }
}

function safeParse(text: string): unknown {
  try {
    return JSON.parse(text);
  } catch {
    return null;
  }
}
