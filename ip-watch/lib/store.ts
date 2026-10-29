import { randomUUID } from "crypto";
import * as sb from "./supabase";
import type { Client, FieldEdit, Finding, HistoryEvent, Mark, Right, Role, RunSummary, WatchRun } from "./types";
import fixture from "@/data/dev-fixture.json";

// Supabase in production; an in-memory store seeded with a fictitious client only for local development.
const memory = !sb.supabaseConfigured() && (process.env.NODE_ENV !== "production" || process.env.IPW_DEV_MEMORY === "1");

type ClientRow = {
  id: string; slug: string; name: string; siren: string | null; company: Client["company"]; holder_names: string[];
  status: Client["status"]; error: string | null; data_updated_at: string | null; imported_at: string | null; created_at: string;
};
type MarkRow = { client_id: string; id: string; data: Right; raw?: unknown; edits: Record<string, FieldEdit> };
type EventRow = { id: number; client_id: string | null; created_at: string; kind: HistoryEvent["kind"]; title: string; summary: string | null; details: Record<string, unknown> | null };
type RunRow = { id: string; client_id: string; mark_id: string; mark_name: string; trigger: WatchRun["trigger"]; status: WatchRun["status"]; started_at: string; finished_at: string | null; stats: WatchRun["stats"]; error: string | null };
type FindingRow = {
  id: number; run_id: string; client_id: string; mark_id: string; found_at: string; kind: Finding["kind"]; source: string; platform: string | null;
  url: string; title: string | null; price: number | null; currency: string | null; country: string | null; seller: string | null;
  company: Finding["company"]; confidence: number | null; reason: string | null; screenshot_path: string | null; sha256: string | null;
  scan_url: string | null; captured_at: string | null; status: string; comment: string | null;
};

const mem = {
  clients: [] as ClientRow[],
  marks: [] as MarkRow[],
  events: [] as EventRow[],
  runs: [] as RunRow[],
  findings: [] as FindingRow[],
  seq: 1,
};

if (memory && mem.clients.length === 0) {
  const id = randomUUID();
  mem.clients.push({ id, slug: "exemple", name: "Maison Exemple (test local)", siren: null, company: null, holder_names: ["MAISON EXEMPLE"], status: "ready", error: null, data_updated_at: null, imported_at: new Date().toISOString(), created_at: new Date().toISOString() });
  for (const r of (fixture as { rights: Right[] }).rights) mem.marks.push({ client_id: id, id: r.id, data: r, edits: {} });
  // One finished search so the live results view can be checked locally.
  type FixtureFinding = { title: string; url: string; platform: string; price: number | null; currency: string | null; country: string | null; seller: string | null; relevance: number; comment: string; kind: string };
  const runId = randomUUID();
  const at = new Date(Date.now() - 3600_000).toISOString();
  const items = (fixture as unknown as { findings: FixtureFinding[] }).findings;
  mem.runs.push({ id: runId, client_id: id, mark_id: "WO-DEMO-0004", mark_name: "ORBIS", trigger: "bouton", status: "done", started_at: at, finished_at: at, error: null, stats: { steps: [{ name: "Google Shopping", status: "done" }, { name: "Évaluation", status: "done" }], total: items.length, high: items.filter((f) => f.relevance >= 4).length, byCountry: {} } });
  for (const f of items) {
    mem.findings.push({
      id: mem.seq++, run_id: runId, client_id: id, mark_id: "WO-DEMO-0004", found_at: at, kind: f.kind === "domaine" ? "domaine" : f.kind === "marketplace" ? "annonce" : "web", source: f.platform,
      platform: f.seller, url: f.url, title: f.title, price: f.price, currency: f.currency, country: f.country, seller: f.seller, company: null,
      confidence: f.relevance * 18 + 5, reason: f.comment || null, screenshot_path: null, sha256: null, scan_url: null, captured_at: null, status: "a_verifier", comment: null,
    });
  }
}

const toClient = (r: ClientRow): Client => ({
  id: r.id, slug: r.slug, name: r.name, status: r.status, company: r.company, holderNames: r.holder_names ?? [],
  dataUpdatedAt: r.data_updated_at, importedAt: r.imported_at, error: r.error,
});

const toRun = (r: RunRow): WatchRun => ({
  id: r.id, clientId: r.client_id, markId: r.mark_id, markName: r.mark_name, trigger: r.trigger, status: r.status,
  startedAt: r.started_at, finishedAt: r.finished_at, stats: r.stats, error: r.error,
});

export const toFinding = (r: FindingRow): Finding => ({
  id: r.id, runId: r.run_id, markId: r.mark_id, foundAt: r.found_at, kind: r.kind, source: r.source, platform: r.platform, url: r.url,
  title: r.title, price: r.price === null ? null : Number(r.price), currency: r.currency, country: r.country, seller: r.seller, company: r.company,
  confidence: r.confidence, reason: r.reason, screenshotPath: r.screenshot_path, sha256: r.sha256, scanUrl: r.scan_url,
  capturedAt: r.captured_at, status: r.status, comment: r.comment,
});

export function storeReady(): boolean {
  return memory || sb.supabaseConfigured();
}

// ---------- Clients

export async function listClients(): Promise<Client[]> {
  const rows = memory ? mem.clients : await sb.select<ClientRow>("clients", { order: "created_at.asc" });
  return rows.map(toClient);
}

export async function getClientBySlug(slug: string): Promise<Client | null> {
  const rows = memory ? mem.clients.filter((c) => c.slug === slug) : await sb.select<ClientRow>("clients", { slug: `eq.${slug}` });
  return rows[0] ? toClient(rows[0]) : null;
}

export async function getClientById(id: string): Promise<Client | null> {
  const rows = memory ? mem.clients.filter((c) => c.id === id) : await sb.select<ClientRow>("clients", { id: `eq.${id}` });
  return rows[0] ? toClient(rows[0]) : null;
}

export async function saveClient(c: { slug: string; name: string; siren?: string | null; company?: Client["company"]; holderNames: string[]; status: Client["status"] }): Promise<Client> {
  const row = { slug: c.slug, name: c.name, siren: c.siren ?? null, company: c.company ?? null, holder_names: c.holderNames, status: c.status, error: null };
  if (memory) {
    const existing = mem.clients.find((x) => x.slug === c.slug);
    if (existing) Object.assign(existing, row);
    else mem.clients.push({ ...row, id: randomUUID(), data_updated_at: null, imported_at: null, created_at: new Date().toISOString() });
    return toClient(mem.clients.find((x) => x.slug === c.slug)!);
  }
  await sb.upsert("clients", [row], "slug");
  return (await getClientBySlug(c.slug))!;
}

export async function updateClient(id: string, patch: Partial<Pick<ClientRow, "status" | "error" | "data_updated_at" | "imported_at" | "holder_names">>): Promise<void> {
  if (memory) {
    Object.assign(mem.clients.find((c) => c.id === id) ?? {}, patch);
    return;
  }
  await sb.update("clients", { id: `eq.${id}` }, patch);
}

// ---------- Marks

function summarize(runs: RunRow[]): Map<string, RunSummary> {
  const out = new Map<string, RunSummary>();
  for (const r of runs) {
    if (out.has(r.mark_id)) continue;
    out.set(r.mark_id, { id: r.id, startedAt: r.started_at, status: r.status, total: r.stats?.total ?? 0, high: r.stats?.high ?? 0 });
  }
  return out;
}

export async function listMarks(clientId: string): Promise<Mark[]> {
  const [rows, runs] = memory
    ? [mem.marks.filter((m) => m.client_id === clientId), mem.runs.filter((r) => r.client_id === clientId).sort((a, b) => b.started_at.localeCompare(a.started_at))]
    : await Promise.all([
        sb.select<MarkRow>("marks", { client_id: `eq.${clientId}`, select: "client_id,id,data,edits" }),
        sb.select<RunRow>("watch_runs", { client_id: `eq.${clientId}`, order: "started_at.desc", limit: "500" }),
      ]);
  const last = summarize(runs);
  return rows.map((r) => ({ ...r.data, edits: r.edits ?? {}, lastRun: last.get(r.id) ?? null }));
}

export async function upsertMarks(clientId: string, items: { right: Right; raw: unknown }[]): Promise<void> {
  if (memory) {
    for (const { right, raw } of items) {
      const existing = mem.marks.find((m) => m.client_id === clientId && m.id === right.id);
      if (existing) Object.assign(existing, { data: right, raw });
      else mem.marks.push({ client_id: clientId, id: right.id, data: right, raw, edits: {} });
    }
    return;
  }
  // Re-imports keep the lawyer's edits: the stored data is the registry value with edits applied on top.
  const existing = await sb.select<MarkRow>("marks", { client_id: `eq.${clientId}`, select: "id,edits" });
  const edits = new Map(existing.map((m) => [m.id, m.edits ?? {}]));
  const rows = items.map(({ right, raw }) => {
    const e = edits.get(right.id) ?? {};
    const data = { ...right, ...Object.fromEntries(Object.entries(e).map(([k, v]) => [k, v.value])) };
    return { client_id: clientId, id: right.id, data, raw, updated_at: new Date().toISOString() };
  });
  for (let i = 0; i < rows.length; i += 200) await sb.upsert("marks", rows.slice(i, i + 200), "client_id,id");
}

// After a full re-import: titles no longer returned by the registry are removed.
export async function removeStaleMarks(clientId: string, keepIds: string[]): Promise<number> {
  const keep = new Set(keepIds);
  if (memory) {
    const before = mem.marks.length;
    mem.marks = mem.marks.filter((m) => m.client_id !== clientId || keep.has(m.id));
    return before - mem.marks.length;
  }
  const existing = await sb.select<{ id: string }>("marks", { client_id: `eq.${clientId}`, select: "id" });
  const stale = existing.map((m) => m.id).filter((id) => !keep.has(id));
  for (let i = 0; i < stale.length; i += 100) {
    const ids = stale.slice(i, i + 100).map((id) => `"${id.replace(/"/g, "")}"`).join(",");
    await sb.remove("marks", { client_id: `eq.${clientId}`, id: `in.(${ids})` });
  }
  return stale.length;
}

// Removes the client with its titles, searches, results and history; screenshots are deleted too.
export async function deleteClient(id: string): Promise<void> {
  if (memory) {
    mem.clients = mem.clients.filter((c) => c.id !== id);
    mem.marks = mem.marks.filter((m) => m.client_id !== id);
    mem.runs = mem.runs.filter((r) => r.client_id !== id);
    mem.findings = mem.findings.filter((f) => f.client_id !== id);
    mem.events = mem.events.filter((e) => e.client_id !== id);
    return;
  }
  await sb.removeObjects("evidence", id).catch(() => undefined);
  await sb.remove("clients", { id: `eq.${id}` });
}

export async function updateMarkField(clientId: string, markId: string, field: keyof Right, value: unknown, role: Role): Promise<{ prev: unknown; mark: Right } | null> {
  const rows = memory
    ? mem.marks.filter((m) => m.client_id === clientId && m.id === markId)
    : await sb.select<MarkRow>("marks", { client_id: `eq.${clientId}`, id: `eq.${markId}`, select: "client_id,id,data,edits" });
  const row = rows[0];
  if (!row) return null;
  const prev = row.data[field];
  const original = row.edits?.[field]?.prev ?? prev;
  const data = { ...row.data, [field]: value } as Right;
  const edits = { ...(row.edits ?? {}), [field]: { value, prev: original, by: role, at: new Date().toISOString() } };
  if (memory) Object.assign(row, { data, edits });
  else await sb.update("marks", { client_id: `eq.${clientId}`, id: `eq.${markId}` }, { data, edits, updated_at: new Date().toISOString() });
  return { prev, mark: data };
}

// ---------- History

export async function addEvent(clientId: string | null, kind: HistoryEvent["kind"], title: string, summary: string | null, details: Record<string, unknown> | null = null): Promise<void> {
  const row = { client_id: clientId, kind, title, summary, details };
  if (memory) mem.events.push({ ...row, id: mem.seq++, created_at: new Date().toISOString() });
  else await sb.insert("events", [row]);
}

export async function listEvents(clientId: string | null, limit = 200): Promise<HistoryEvent[]> {
  const rows = memory
    ? mem.events.filter((e) => !clientId || e.client_id === clientId).sort((a, b) => b.created_at.localeCompare(a.created_at)).slice(0, limit)
    : await sb.select<EventRow>("events", { ...(clientId ? { or: `(client_id.eq.${clientId},client_id.is.null)` } : {}), order: "created_at.desc", limit: String(limit) });
  return rows.map((r) => ({ id: r.id, createdAt: r.created_at, kind: r.kind, title: r.title, summary: r.summary, details: r.details }));
}

export async function emailAlreadyHandled(emailId: string): Promise<boolean> {
  if (memory) return mem.events.some((e) => e.details?.emailId === emailId);
  const rows = await sb.select<EventRow>("events", { "details->>emailId": `eq.${emailId}`, select: "id", limit: "1" });
  return rows.length > 0;
}

// ---------- Infringement searches

export async function createRun(clientId: string, markId: string, markName: string, trigger: WatchRun["trigger"]): Promise<WatchRun> {
  const row = { client_id: clientId, mark_id: markId, mark_name: markName, trigger, status: "running" as const, stats: { steps: [], total: 0, high: 0, byCountry: {} } };
  if (memory) {
    const full: RunRow = { ...row, id: randomUUID(), started_at: new Date().toISOString(), finished_at: null, error: null };
    mem.runs.push(full);
    return toRun(full);
  }
  const [created] = await sb.insert<RunRow>("watch_runs", [row]);
  return toRun(created);
}

export async function updateRun(id: string, patch: Partial<Pick<RunRow, "status" | "stats" | "finished_at" | "error">>): Promise<void> {
  if (memory) {
    Object.assign(mem.runs.find((r) => r.id === id) ?? {}, patch);
    return;
  }
  await sb.update("watch_runs", { id: `eq.${id}` }, patch);
}

export async function getRun(id: string): Promise<{ run: WatchRun; findings: Finding[] } | null> {
  const [runs, findings] = memory
    ? [mem.runs.filter((r) => r.id === id), mem.findings.filter((f) => f.run_id === id)]
    : await Promise.all([sb.select<RunRow>("watch_runs", { id: `eq.${id}` }), sb.select<FindingRow>("findings", { run_id: `eq.${id}`, order: "id.asc" })]);
  if (!runs[0]) return null;
  return { run: toRun(runs[0]), findings: findings.map(toFinding) };
}

export async function latestRuns(clientId: string, limit = 20): Promise<WatchRun[]> {
  const rows = memory
    ? mem.runs.filter((r) => r.client_id === clientId).sort((a, b) => b.started_at.localeCompare(a.started_at)).slice(0, limit)
    : await sb.select<RunRow>("watch_runs", { client_id: `eq.${clientId}`, order: "started_at.desc", limit: String(limit) });
  return rows.map(toRun);
}

export type NewFinding = Omit<FindingRow, "id" | "found_at" | "status" | "comment" | "screenshot_path" | "sha256" | "scan_url" | "captured_at"> & Partial<Pick<FindingRow, "screenshot_path" | "sha256" | "scan_url" | "captured_at">>;

export async function addFindings(rows: NewFinding[]): Promise<Finding[]> {
  if (rows.length === 0) return [];
  if (memory) {
    const created = rows.map((r) => ({ screenshot_path: null, sha256: null, scan_url: null, captured_at: null, ...r, id: mem.seq++, found_at: new Date().toISOString(), status: "a_verifier", comment: null }));
    mem.findings.push(...created);
    return created.map(toFinding);
  }
  return (await sb.insert<FindingRow>("findings", rows)).map(toFinding);
}

export async function updateFinding(id: number, patch: Partial<FindingRow>): Promise<void> {
  if (memory) {
    Object.assign(mem.findings.find((f) => f.id === id) ?? {}, patch);
    return;
  }
  await sb.update("findings", { id: `eq.${id}` }, patch);
}

export async function markFindings(clientId: string, markId: string): Promise<{ finding: Finding; run: WatchRun | null }[]> {
  const [findings, runs] = memory
    ? [mem.findings.filter((f) => f.client_id === clientId && f.mark_id === markId), mem.runs.filter((r) => r.client_id === clientId && r.mark_id === markId)]
    : await Promise.all([
        sb.select<FindingRow>("findings", { client_id: `eq.${clientId}`, mark_id: `eq.${markId}`, order: "found_at.asc" }),
        sb.select<RunRow>("watch_runs", { client_id: `eq.${clientId}`, mark_id: `eq.${markId}` }),
      ]);
  const byId = new Map(runs.map((r) => [r.id, toRun(r)]));
  return findings.map((f) => ({ finding: toFinding(f), run: byId.get(f.run_id) ?? null }));
}

export async function saveRawImport(row: { run_id: string; client_id: string | null; kind: string; query?: string; position?: number; http_status: number | null; payload: unknown; error?: string | null }): Promise<void> {
  if (memory) return;
  await sb.insert("raw_imports", [{ source: "inpi", query: null, position: null, error: null, ...row }]);
}
