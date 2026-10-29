// Minimal PostgREST client using the server-side secret key; never import from client components.
function config() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;
  return { url: url.replace(/\/$/, ""), key };
}

export function supabaseConfigured(): boolean {
  return config() !== null;
}

export async function insertRows(table: string, rows: Record<string, unknown>[]): Promise<void> {
  const c = config();
  if (!c) throw new Error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY manquants");
  const res = await fetch(`${c.url}/rest/v1/${table}`, {
    method: "POST",
    headers: {
      apikey: c.key,
      Authorization: `Bearer ${c.key}`,
      "Content-Type": "application/json",
      Prefer: "return=minimal",
    },
    body: JSON.stringify(rows),
  });
  if (!res.ok) throw new Error(`Supabase ${res.status}: ${(await res.text()).slice(0, 300)}`);
}
