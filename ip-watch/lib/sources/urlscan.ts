// urlscan.io: dated capture of a page (screenshot + technical data) and search of past scans for lookalike domains.
const BASE = "https://urlscan.io";

export function urlscanConfigured(): boolean {
  return Boolean(process.env.URLSCAN_API_KEY);
}

function key(): string {
  const k = process.env.URLSCAN_API_KEY;
  if (!k) throw new Error("URLSCAN_API_KEY manquant");
  return k;
}

export type Capture = { uuid: string; resultUrl: string; screenshot: ArrayBuffer | null; capturedAt: string; country: string | null; ip: string | null; domain: string | null };

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

// Submits an unlisted scan and waits for the result (usually 10–30 s).
export async function capture(url: string, timeoutMs = 75_000): Promise<Capture | null> {
  const submit = await fetch(`${BASE}/api/v1/scan/`, {
    method: "POST",
    headers: { "API-Key": key(), "Content-Type": "application/json" },
    body: JSON.stringify({ url, visibility: "unlisted" }),
  });
  if (!submit.ok) throw new Error(`urlscan scan ${submit.status}: ${(await submit.text()).slice(0, 200)}`);
  const { uuid } = (await submit.json()) as { uuid: string };
  const deadline = Date.now() + timeoutMs;
  await sleep(10_000);
  while (Date.now() < deadline) {
    const res = await fetch(`${BASE}/api/v1/result/${uuid}/`, { headers: { "API-Key": key() } });
    if (res.status === 200) {
      const r = (await res.json()) as { page?: { country?: string; ip?: string; domain?: string }; task?: { time?: string; screenshotURL?: string } };
      const shot = await fetch(r.task?.screenshotURL ?? `${BASE}/screenshots/${uuid}.png`, { headers: { "API-Key": key() } });
      return {
        uuid,
        resultUrl: `${BASE}/result/${uuid}/`,
        screenshot: shot.ok ? await shot.arrayBuffer() : null,
        capturedAt: r.task?.time ?? new Date().toISOString(),
        country: r.page?.country ?? null,
        ip: r.page?.ip ?? null,
        domain: r.page?.domain ?? null,
      };
    }
    if (res.status !== 404) throw new Error(`urlscan result ${res.status}`);
    await sleep(4_000);
  }
  return null;
}

// Domains containing the brand that urlscan has already seen.
export async function lookalikeDomains(term: string, officialHosts: string[]): Promise<{ domain: string; url: string; country: string | null; seenAt: string }[]> {
  const q = `page.domain:*${term.toLowerCase().replace(/[^a-z0-9-]/g, "")}*`;
  const res = await fetch(`${BASE}/api/v1/search/?${new URLSearchParams({ q, size: "50" })}`, { headers: { "API-Key": key() } });
  if (!res.ok) throw new Error(`urlscan search ${res.status}`);
  const json = (await res.json()) as { results?: { page?: { domain?: string; url?: string; country?: string }; task?: { time?: string } }[] };
  const seen = new Map<string, { domain: string; url: string; country: string | null; seenAt: string }>();
  for (const r of json.results ?? []) {
    const domain = r.page?.domain;
    if (!domain || seen.has(domain) || officialHosts.some((h) => domain === h || domain.endsWith(`.${h}`))) continue;
    seen.set(domain, { domain, url: r.page?.url ?? `https://${domain}`, country: r.page?.country ?? null, seenAt: r.task?.time ?? "" });
  }
  return Array.from(seen.values()).slice(0, 15);
}
