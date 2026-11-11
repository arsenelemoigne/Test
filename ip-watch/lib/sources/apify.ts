// Apify runs scrapers for platforms without a usable public API; here the Meta Ad Library.
// One synchronous call per search; maxTotalChargeUsd is a technical cap set above the Actor's real price.
const BASE = "https://api.apify.com/v2";
const META_ADS_ACTOR = "automation-lab~facebook-ads-library";

export function apifyConfigured(): boolean {
  return Boolean(process.env.APIFY_TOKEN);
}

async function runSync(actor: string, input: unknown, capUsd: number, timeoutSec = 150): Promise<Record<string, unknown>[]> {
  const token = process.env.APIFY_TOKEN;
  if (!token) throw new Error("APIFY_TOKEN manquant");
  const qs = new URLSearchParams({ timeout: String(timeoutSec), format: "json", clean: "true", maxTotalChargeUsd: String(capUsd) });
  const res = await fetch(`${BASE}/acts/${actor}/run-sync-get-dataset-items?${qs}`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}`, "Content-Type": "application/json" },
    body: JSON.stringify(input),
    signal: AbortSignal.timeout((timeoutSec + 30) * 1000),
  });
  const text = await res.text();
  if (res.status === 408) throw new Error("Apify : délai dépassé");
  if (!res.ok) throw new Error(`Apify ${res.status}: ${text.slice(0, 200)}`);
  const json = JSON.parse(text) as unknown;
  return Array.isArray(json) ? (json as Record<string, unknown>[]) : [];
}

export type MetaAd = { url: string; pageName: string | null; startDate: string | null; endDate: string | null; image: string | null; text: string | null; query: string | null };

const str = (v: unknown): string | null => (typeof v === "string" && v.trim() ? v.trim() : null);
const firstString = (...values: unknown[]): string | null => {
  for (const v of values) {
    if (Array.isArray(v)) {
      const inner = firstString(...v);
      if (inner) return inner;
    } else if (v && typeof v === "object") {
      const o = v as Record<string, unknown>;
      const inner = firstString(o.url, o.original_image_url, o.resized_image_url, o.video_preview_image_url, o.src);
      if (inner) return inner;
    } else if (str(v)) return str(v);
  }
  return null;
};
const isoDate = (v: unknown): string | null => {
  if (typeof v === "number") return new Date(v > 1e12 ? v : v * 1000).toISOString().slice(0, 10);
  const s = str(v);
  if (!s) return null;
  const d = new Date(s);
  return Number.isNaN(d.getTime()) ? s.slice(0, 10) : d.toISOString().slice(0, 10);
};

// Ads whose text matches the queries, shown in the EU over the past year (the library's own limit).
export async function metaAds(queries: string[], opts: { country?: string; maxAds?: number } = {}): Promise<MetaAd[]> {
  const items = await runSync(META_ADS_ACTOR, { searchQueries: queries, country: opts.country ?? "FR", activeStatus: "all", maxAds: opts.maxAds ?? 20 }, 0.3);
  return items
    .map((i): MetaAd | null => {
      const snapshot = (i.snapshot ?? {}) as Record<string, unknown>;
      const id = str(i.adArchiveId) ?? str(i.ad_archive_id) ?? str(i.id) ?? (typeof i.adArchiveId === "number" ? String(i.adArchiveId) : null);
      const url = str(i.adLibraryUrl) ?? str(i.url) ?? (id ? `https://www.facebook.com/ads/library/?id=${id}` : null);
      if (!url) return null;
      return {
        url,
        pageName: str(i.pageName) ?? str(i.page_name) ?? str(snapshot.page_name) ?? null,
        startDate: isoDate(i.startDate ?? i.start_date ?? i.startDateFormatted),
        endDate: isoDate(i.endDate ?? i.end_date),
        image: firstString(i.imageUrls, i.images, i.imageUrl, i.thumbnailUrl, i.videoPreviewUrls, i.videoPreviewImageUrl, snapshot.images, snapshot.videos, snapshot.cards),
        text: str(i.bodyText) ?? str(i.body) ?? str((snapshot.body as Record<string, unknown> | undefined)?.text) ?? str(i.title) ?? null,
        query: str(i.searchQuery) ?? str(i.query) ?? null,
      };
    })
    .filter((x): x is MetaAd => x !== null);
}
