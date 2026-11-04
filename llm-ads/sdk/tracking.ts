import { randomBytes } from 'node:crypto';
import { canonicalJson, clickMessage, hmacHex } from '../shared/crypto.ts';
import type { ImpressionReport } from '../shared/types.ts';

/** A click link signed by the publisher, so the marketplace knows which app to pay. */
export function signedClickUrl(options: {
  marketplaceUrl: string;
  publisherId: string;
  publisherSecret: string;
  offerId: string;
  nowMs?: number;
}): string {
  const params = {
    offerId: options.offerId,
    publisherId: options.publisherId,
    ts: String(Math.floor((options.nowMs ?? Date.now()) / 1000)),
    nonce: randomBytes(12).toString('hex'),
  };
  const url = new URL(`/c/${encodeURIComponent(options.offerId)}`, options.marketplaceUrl);
  url.searchParams.set('pub', params.publisherId);
  url.searchParams.set('ts', params.ts);
  url.searchParams.set('n', params.nonce);
  url.searchParams.set('sig', hmacHex(options.publisherSecret, clickMessage(params)));
  return url.toString();
}

/** Counts views per offer and reports only the totals. Nothing about who saw what, or when. */
export class ImpressionCounter {
  private counts = new Map<string, number>();
  private periodStart = new Date();
  private readonly options: { marketplaceUrl: string; publisherId: string; publisherSecret: string; fetchImpl?: typeof fetch };

  constructor(options: { marketplaceUrl: string; publisherId: string; publisherSecret: string; fetchImpl?: typeof fetch }) {
    this.options = options;
  }

  record(offerId: string): void {
    this.counts.set(offerId, (this.counts.get(offerId) ?? 0) + 1);
  }

  async flush(): Promise<void> {
    if (this.counts.size === 0) return;
    const counts = this.counts;
    const report: ImpressionReport = {
      publisherId: this.options.publisherId,
      periodStart: this.periodStart.toISOString(),
      periodEnd: new Date().toISOString(),
      counts: Object.fromEntries(counts),
    };
    this.counts = new Map();
    this.periodStart = new Date();
    try {
      const res = await (this.options.fetchImpl ?? fetch)(new URL('/impressions', this.options.marketplaceUrl), {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-signature': hmacHex(this.options.publisherSecret, canonicalJson(report)) },
        body: JSON.stringify(report),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      // Keep the counts for the next flush.
      for (const [id, n] of counts) this.counts.set(id, (this.counts.get(id) ?? 0) + n);
      console.warn(`[llm-ads] impression flush failed: ${err instanceof Error ? err.message : err}`);
    }
  }
}
