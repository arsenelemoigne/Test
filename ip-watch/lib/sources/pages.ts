// Product photos of a public page (og:image and the main product images), used as visual references.
const UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";

export async function pageImages(url: string, hint: string, limit = 4): Promise<string[]> {
  const res = await fetch(url, { headers: { "User-Agent": UA, Accept: "text/html", "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8" }, redirect: "follow", signal: AbortSignal.timeout(12_000) });
  if (!res.ok) throw new Error(`page ${res.status}`);
  const html = (await res.text()).slice(0, 800_000);
  const abs = (src: string) => {
    try {
      const decoded = src.replace(/&#x2F;|&#47;/gi, "/").replace(/&quot;/g, '"').replace(/&amp;/g, "&");
      return new URL(decoded, res.url || url).toString();
    } catch {
      return null;
    }
  };
  const out: string[] = [];
  const push = (src: string | undefined) => {
    const u = src ? abs(src) : null;
    if (u && /^https?:/.test(u) && !/\.svg(\?|$)|logo|icon|sprite|placeholder/i.test(u) && !out.includes(u)) out.push(u);
  };
  for (const m of Array.from(html.matchAll(/<meta[^>]+(?:property|name)=["'](?:og:image|twitter:image)(?::src)?["'][^>]*>/gi))) push(m[0].match(/content=["']([^"']+)["']/i)?.[1]);
  const words = hint.toLowerCase().split(/\s+/).filter((w) => w.length > 2);
  for (const m of Array.from(html.matchAll(/<img[^>]+>/gi))) {
    const tag = m[0];
    const src = tag.match(/\s(?:data-src|src)=["']([^"']+)["']/i)?.[1];
    const alt = (tag.match(/\salt=["']([^"']*)["']/i)?.[1] ?? "").toLowerCase();
    if (src && (words.some((w) => alt.includes(w) || src.toLowerCase().includes(w)) || /product|produit/i.test(src))) push(src);
    if (out.length >= limit) break;
  }
  return out.slice(0, limit);
}
