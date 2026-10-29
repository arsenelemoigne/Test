// PISTE gateway (Légifrance, Judilibre): OAuth client credentials, or an API key in the KeyId header as fallback.
const TOKEN_URL = "https://oauth.piste.gouv.fr/api/oauth/token";
export const PISTE_BASE = "https://api.piste.gouv.fr";

let cached: { token: string; expires: number } | null = null;

export function pisteConfigured(): boolean {
  return Boolean((process.env.PISTE_CLIENT_ID && process.env.PISTE_CLIENT_SECRET) || process.env.PISTE_API_KEY);
}

async function authHeaders(): Promise<Record<string, string>> {
  const id = process.env.PISTE_CLIENT_ID;
  const secret = process.env.PISTE_CLIENT_SECRET;
  if (id && secret) {
    if (!cached || cached.expires < Date.now() + 60_000) {
      const res = await fetch(TOKEN_URL, {
        method: "POST",
        headers: { "Content-Type": "application/x-www-form-urlencoded" },
        body: new URLSearchParams({ grant_type: "client_credentials", client_id: id, client_secret: secret, scope: "openid" }),
      });
      if (!res.ok) throw new Error(`PISTE OAuth ${res.status}: ${(await res.text()).slice(0, 200)}`);
      const json = (await res.json()) as { access_token: string; expires_in?: number };
      cached = { token: json.access_token, expires: Date.now() + (json.expires_in ?? 3600) * 1000 };
    }
    return { Authorization: `Bearer ${cached.token}` };
  }
  if (process.env.PISTE_API_KEY) return { KeyId: process.env.PISTE_API_KEY };
  throw new Error("PISTE non configuré (PISTE_CLIENT_ID / PISTE_CLIENT_SECRET)");
}

export async function piste<T>(method: "GET" | "POST", path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${PISTE_BASE}${path}`, {
    method,
    headers: { Accept: "application/json", ...(body ? { "Content-Type": "application/json" } : {}), ...(await authHeaders()) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`PISTE ${path} ${res.status}: ${text.slice(0, 300)}`);
  return JSON.parse(text) as T;
}
