import { verifyPayload } from '../shared/crypto.ts';
import type { Catalog, SignedCatalog } from '../shared/types.ts';

export interface CatalogClientOptions {
  marketplaceUrl: string;
  /** Pin this in production. It proves the bids came from the marketplace. */
  publicKeyPem: string;
  refreshMs?: number;
  fetchImpl?: typeof fetch;
}

/** Verifies a signed catalog. Throws if the signature is wrong or the catalog has expired. */
export function verifyCatalog(signed: SignedCatalog, publicKeyPem: string, now = new Date()): Catalog {
  if (!signed?.catalog || typeof signed.signature !== 'string') throw new Error('malformed catalog');
  if (!verifyPayload(signed.catalog, signed.signature, publicKeyPem)) throw new Error('bad catalog signature');
  if (new Date(signed.catalog.expiresAt) <= now) throw new Error('catalog expired');
  return signed.catalog;
}

/**
 * Downloads the whole catalog, so the marketplace learns nothing about what users talk about.
 * Fails open: if the marketplace is down, the app keeps working with the last valid catalog or no ads.
 */
export class CatalogClient {
  private readonly options: Required<Omit<CatalogClientOptions, 'fetchImpl'>> & { fetchImpl: typeof fetch };
  private current: Catalog | null = null;
  private fetchedAt = 0;
  private inflight: Promise<Catalog | null> | null = null;

  constructor(options: CatalogClientOptions) {
    // Explicit ?? rather than spread: callers may pass `undefined` for optional fields.
    this.options = {
      marketplaceUrl: options.marketplaceUrl,
      publicKeyPem: options.publicKeyPem,
      refreshMs: options.refreshMs ?? 60_000,
      fetchImpl: options.fetchImpl ?? fetch,
    };
  }

  async get(): Promise<Catalog | null> {
    const now = Date.now();
    const fresh = this.current && now - this.fetchedAt < this.options.refreshMs && new Date(this.current.expiresAt).getTime() > now;
    if (fresh) return this.current;
    this.inflight ??= this.refresh().finally(() => {
      this.inflight = null;
    });
    return this.inflight;
  }

  private async refresh(): Promise<Catalog | null> {
    try {
      const res = await this.options.fetchImpl(new URL('/catalog', this.options.marketplaceUrl));
      if (!res.ok) throw new Error(`catalog HTTP ${res.status}`);
      this.current = verifyCatalog((await res.json()) as SignedCatalog, this.options.publicKeyPem);
      this.fetchedAt = Date.now();
    } catch (err) {
      console.warn(`[llm-ads] catalog refresh failed: ${err instanceof Error ? err.message : err}`);
      if (this.current && new Date(this.current.expiresAt).getTime() <= Date.now()) this.current = null;
    }
    return this.current;
  }
}
