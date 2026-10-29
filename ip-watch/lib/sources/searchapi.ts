// SearchAPI.io: Google Shopping, Google Lens, Amazon, eBay, Google web results as JSON.
const BASE = "https://www.searchapi.io/api/v1/search";

export const QUOTA_MESSAGE = "quota SearchAPI du mois épuisé";

export function searchApiConfigured(): boolean {
  return Boolean(process.env.SEARCHAPI_API_KEY);
}

export type Listing = { title: string; url: string; seller: string | null; price: number | null; currency: string | null; thumbnail: string | null };

async function call(params: Record<string, string>): Promise<Record<string, unknown>> {
  const key = process.env.SEARCHAPI_API_KEY;
  if (!key) throw new Error("SEARCHAPI_API_KEY manquant");
  const res = await fetch(`${BASE}?${new URLSearchParams({ ...params, api_key: key })}`, { headers: { Accept: "application/json" } });
  const text = await res.text();
  if (res.status === 429 && /used all of the searches/i.test(text)) throw new Error(QUOTA_MESSAGE);
  if (!res.ok) throw new Error(`SearchAPI ${params.engine} ${res.status}: ${text.slice(0, 200)}`);
  return JSON.parse(text);
}

function priceOf(item: Record<string, unknown>): { price: number | null; currency: string | null } {
  const p = item.price;
  if (typeof item.extracted_price === "number") return { price: item.extracted_price, currency: currencyOf(String(p ?? "")) };
  if (p && typeof p === "object") {
    const o = p as Record<string, unknown>;
    const value = typeof o.extracted_value === "number" ? o.extracted_value : typeof o.extracted === "number" ? o.extracted : null;
    return { price: value, currency: typeof o.currency === "string" ? o.currency : currencyOf(String(o.value ?? o.raw ?? "")) };
  }
  if (typeof p === "string") {
    const n = Number(p.replace(/[^\d,.]/g, "").replace(/\.(?=\d{3}\b)/g, "").replace(",", "."));
    return { price: Number.isFinite(n) && n > 0 ? n : null, currency: currencyOf(p) };
  }
  return { price: null, currency: null };
}

function currencyOf(s: string): string | null {
  if (s.includes("€") || /EUR/i.test(s)) return "EUR";
  if (s.includes("$") || /USD/i.test(s)) return "USD";
  if (s.includes("£") || /GBP/i.test(s)) return "GBP";
  if (s.includes("¥") || /CNY|JPY/i.test(s)) return s.includes("JPY") ? "JPY" : "CNY";
  return null;
}

function toListings(items: unknown): Listing[] {
  if (!Array.isArray(items)) return [];
  return items
    .map((raw): Listing | null => {
      const i = raw as Record<string, unknown>;
      const url = (i.link ?? i.product_link ?? i.url) as string | undefined;
      const title = (i.title ?? i.name) as string | undefined;
      if (!url || !title) return null;
      const seller = (i.source ?? i.seller ?? i.merchant ?? (i.seller_info as Record<string, unknown> | undefined)?.name ?? null) as string | null;
      return { title, url, seller: typeof seller === "string" ? seller : null, thumbnail: typeof i.thumbnail === "string" ? i.thumbnail : null, ...priceOf(i) };
    })
    .filter((x): x is Listing => x !== null);
}

export async function shopping(q: string) {
  const r = await call({ engine: "google_shopping", q, gl: "fr", hl: "fr" });
  return toListings(r.shopping_results ?? r.inline_shopping_results);
}

export async function lens(imageUrl: string) {
  const r = await call({ engine: "google_lens", url: imageUrl, search_type: "visual_matches", hl: "fr", gl: "fr" });
  return toListings(r.visual_matches);
}

export async function amazon(q: string) {
  const r = await call({ engine: "amazon_search", q, amazon_domain: "amazon.fr" });
  return toListings(r.organic_results);
}

export async function ebay(q: string) {
  const r = await call({ engine: "ebay_search", q, ebay_domain: "ebay.fr" });
  return toListings(r.organic_results);
}

export async function google(q: string) {
  const r = await call({ engine: "google", q, gl: "fr", hl: "fr", num: "10" });
  return toListings(r.organic_results);
}
