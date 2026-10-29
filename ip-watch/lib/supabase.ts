// Minimal PostgREST + Storage client using the server-side secret key; never import from client components.
function config() {
  const url = process.env.SUPABASE_URL;
  const key = process.env.SUPABASE_SERVICE_ROLE_KEY;
  if (!url || !key) return null;
  return { url: url.replace(/\/$/, ""), key };
}

export function supabaseConfigured(): boolean {
  return config() !== null;
}

function required() {
  const c = config();
  if (!c) throw new Error("SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY manquants");
  return c;
}

function headers(key: string, extra: Record<string, string> = {}) {
  return { apikey: key, Authorization: `Bearer ${key}`, "Content-Type": "application/json", ...extra };
}

async function check(res: Response, what: string) {
  if (!res.ok) throw new Error(`Supabase ${what} ${res.status}: ${(await res.text()).slice(0, 300)}`);
}

// Filters use PostgREST syntax, e.g. { client_id: "eq.123", order: "found_at.asc" }.
export async function select<T>(table: string, params: Record<string, string> = {}): Promise<T[]> {
  const c = required();
  const qs = new URLSearchParams({ select: "*", ...params });
  const res = await fetch(`${c.url}/rest/v1/${table}?${qs}`, { headers: headers(c.key), cache: "no-store" });
  await check(res, `select ${table}`);
  return (await res.json()) as T[];
}

export async function insert<T>(table: string, rows: Record<string, unknown>[]): Promise<T[]> {
  const c = required();
  const res = await fetch(`${c.url}/rest/v1/${table}`, {
    method: "POST",
    headers: headers(c.key, { Prefer: "return=representation" }),
    body: JSON.stringify(rows),
  });
  await check(res, `insert ${table}`);
  return (await res.json()) as T[];
}

// Only the columns present in the rows are written, so other columns (e.g. edits) are preserved.
export async function upsert(table: string, rows: Record<string, unknown>[], onConflict: string): Promise<void> {
  if (rows.length === 0) return;
  const c = required();
  const res = await fetch(`${c.url}/rest/v1/${table}?on_conflict=${encodeURIComponent(onConflict)}`, {
    method: "POST",
    headers: headers(c.key, { Prefer: "resolution=merge-duplicates,return=minimal" }),
    body: JSON.stringify(rows),
  });
  await check(res, `upsert ${table}`);
}

export async function update<T>(table: string, filters: Record<string, string>, patch: Record<string, unknown>): Promise<T[]> {
  const c = required();
  const res = await fetch(`${c.url}/rest/v1/${table}?${new URLSearchParams(filters)}`, {
    method: "PATCH",
    headers: headers(c.key, { Prefer: "return=representation" }),
    body: JSON.stringify(patch),
  });
  await check(res, `update ${table}`);
  return (await res.json()) as T[];
}

export async function uploadObject(bucket: string, path: string, bytes: ArrayBuffer, contentType: string): Promise<void> {
  const c = required();
  const res = await fetch(`${c.url}/storage/v1/object/${bucket}/${path}`, {
    method: "POST",
    headers: { apikey: c.key, Authorization: `Bearer ${c.key}`, "Content-Type": contentType, "x-upsert": "true" },
    body: bytes,
  });
  await check(res, "upload");
}

export async function downloadObject(bucket: string, path: string): Promise<Response> {
  const c = required();
  return fetch(`${c.url}/storage/v1/object/${bucket}/${path}`, { headers: { apikey: c.key, Authorization: `Bearer ${c.key}` }, cache: "no-store" });
}
