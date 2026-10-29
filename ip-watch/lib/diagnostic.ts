import { Resend } from "resend";
import * as sb from "./supabase";
import * as inpi from "./sources/inpi";
import { searchAnnouncements } from "./sources/bodacc";
import { searchCompanies } from "./sources/entreprises";
import { LF } from "./sources/legifrance";
import { pisteRaw } from "./sources/piste";
import { lookupDomain } from "./sources/rdap";
import { startTrace, runInTrace } from "./trace";

// One cheap call per source, to see at a glance what works. Nothing here spends SearchAPI or urlscan credits
// nor LLM tokens; INPI costs two calls of its quota.
export type Check = { group: string; name: string; ok: boolean | null; detail: string; ms: number };

const L712_1 = "LEGIARTI000006279687";
const CPI = "LEGITEXT000006069414";

async function timed(group: string, name: string, fn: () => Promise<{ ok: boolean | null; detail: string }>): Promise<Check> {
  const started = Date.now();
  try {
    return { group, name, ...(await fn()), ms: Date.now() - started };
  } catch (e) {
    return { group, name, ok: false, detail: e instanceof Error ? e.message.slice(0, 300) : String(e), ms: Date.now() - started };
  }
}

const missing = (v: string) => ({ ok: null, detail: `non configuré (${v})` });

async function getJson(url: string, headers: Record<string, string> = {}): Promise<{ status: number; json: Record<string, unknown> | null; text: string }> {
  const res = await fetch(url, { headers: { Accept: "application/json", ...headers } });
  const text = await res.text();
  let json: Record<string, unknown> | null = null;
  try {
    json = JSON.parse(text);
  } catch {
    // keep the raw text
  }
  return { status: res.status, json, text };
}

function openRouterChecks(): Promise<Check>[] {
  const key = process.env.OPENROUTER_API_KEY;
  const models = [
    ["modèle principal", process.env.OPENROUTER_MODEL || "google/gemini-2.5-flash"],
    ["modèle rapide", process.env.OPENROUTER_MODEL_FAST || process.env.OPENROUTER_MODEL || "google/gemini-2.5-flash"],
  ];
  return [
    timed("IA", "OpenRouter · clé et crédit", async () => {
      if (!key) return missing("OPENROUTER_API_KEY");
      const { status, json, text } = await getJson("https://openrouter.ai/api/v1/key", { Authorization: `Bearer ${key}` });
      if (status >= 400) return { ok: false, detail: `${status} ${text.slice(0, 200)}` };
      const d = (json?.data ?? {}) as { usage?: number; limit?: number | null; limit_remaining?: number | null };
      const remaining = d.limit_remaining ?? (d.limit != null && d.usage != null ? d.limit - d.usage : null);
      return { ok: true, detail: `utilisé : ${d.usage?.toFixed(2) ?? "?"} $${remaining != null ? ` · reste : ${remaining.toFixed(2)} $` : " · sans plafond sur la clé"}` };
    }),
    ...models.map(([label, id]) =>
      timed("IA", `OpenRouter · ${label}`, async () => {
        const { status, json } = await getJson(`https://openrouter.ai/api/v1/models/${id}/endpoints`);
        const endpoints = ((json?.data as { endpoints?: { provider_name?: string }[] } | undefined)?.endpoints ?? []).map((e) => e.provider_name).filter(Boolean);
        if (status >= 400 || endpoints.length === 0) return { ok: false, detail: `${id} : identifiant inconnu d'OpenRouter (${status})` };
        return { ok: true, detail: `${id} · servi par ${endpoints.slice(0, 3).join(", ")}` };
      }),
    ),
  ];
}

function pisteChecks(): Promise<Check>[] {
  const configured = (process.env.PISTE_CLIENT_ID && process.env.PISTE_CLIENT_SECRET) || process.env.PISTE_API_KEY;
  const raw = (group: string, name: string, method: "GET" | "POST", path: string, body?: unknown) =>
    timed(group, name, async () => {
      if (!configured) return missing("PISTE_CLIENT_ID / PISTE_CLIENT_SECRET");
      const r = await pisteRaw(method, path, body);
      return { ok: r.status < 400, detail: `${r.status} · ${r.mode} · ${r.text.replace(/\s+/g, " ").slice(0, 220) || "(réponse vide)"}` };
    });
  const today = Date.now();
  const numArticle = { typeChamp: "NUM_ARTICLE", criteres: [{ typeRecherche: "EXACTE", valeur: "L712-1", operateur: "ET" }], operateur: "ET" };
  const page = { pageNumber: 1, pageSize: 3, operateur: "ET", sort: "PERTINENCE", typePagination: "ARTICLE" };
  return [
    raw("Légifrance", "article par identifiant (L712-1)", "POST", `${LF}/consult/getArticle`, { id: L712_1 }),
    raw("Légifrance", "article par code et numéro", "POST", `${LF}/consult/getArticleWithIdandNum`, { id: CPI, num: "L712-1" }),
    raw("Légifrance", "recherche · CODE_DATE + nom du code", "POST", `${LF}/search`, {
      fond: "CODE_DATE",
      recherche: { ...page, champs: [numArticle], filtres: [{ facette: "NOM_CODE", valeurs: ["Code de la propriété intellectuelle"] }, { facette: "DATE_VERSION", singleDate: today }] },
    }),
    raw("Légifrance", "recherche · CODE_ETAT mots-clés", "POST", `${LF}/search`, {
      fond: "CODE_ETAT",
      recherche: { ...page, typePagination: "DEFAUT", champs: [{ typeChamp: "ALL", criteres: [{ typeRecherche: "UN_DES_MOTS", valeur: "marque renouvelable", operateur: "ET" }], operateur: "ET" }], filtres: [] },
    }),
    raw("Judilibre", "disponibilité", "GET", "/cassation/judilibre/v1.0/healthcheck"),
    raw("Judilibre", "recherche « contrefaçon de marque »", "GET", `/cassation/judilibre/v1.0/search?${new URLSearchParams({ query: "contrefaçon de marque", page_size: "1" })}`),
  ];
}

export async function runDiagnostic(): Promise<{ traceId: string; checks: Check[] }> {
  const trace = await startTrace("diagnostic", "Diagnostic des sources", null);
  const checks = await runInTrace(trace, async () => {
    const all = await Promise.all([
      timed("Données", "Supabase", async () => {
        if (!sb.supabaseConfigured()) return missing("SUPABASE_URL");
        const rows = await sb.select<{ id: string }>("clients", { select: "id", limit: "1" });
        return { ok: true, detail: `base joignable · ${rows.length ? "au moins un client" : "aucun client"}` };
      }),
      ...openRouterChecks(),
      timed("Registres", "INPI · connexion et recherche", async () => {
        if (!inpi.inpiConfigured()) return missing("INPI_USERNAME / INPI_PASSWORD");
        const res = await inpi.search("marques", "[Mark_Exp=QUATRE] ET [DEPOSANT=BOUCHERON]", ["FR"], 0, 1, inpi.MARK_FIELDS);
        return { ok: true, detail: `connecté · ${res.hits.length ? "marque QUATRE trouvée" : "aucun résultat"} (2 appels de quota)` };
      }),
      timed("Registres", "Recherche d'entreprises", async () => {
        const c = await searchCompanies("boucheron", 1);
        return { ok: c.length > 0, detail: c[0] ? `${c[0].name} · SIREN ${c[0].siren}` : "aucun résultat" };
      }),
      timed("Registres", "BODACC", async () => {
        const a = await searchAnnouncements("330160367");
        return { ok: true, detail: `${a.length} annonce(s) · dernière : ${a[0]?.date ?? "—"}` };
      }),
      ...pisteChecks(),
      timed("Droit de l'UE", "EUR-Lex (lien du règlement 2017/1001)", async () => {
        const res = await fetch("https://eur-lex.europa.eu/legal-content/FR/TXT/?uri=CELEX:32017R1001", { method: "HEAD" });
        return { ok: res.status < 400, detail: `${res.status} · recherche faite par le web (OpenRouter), liens vérifiés` };
      }),
      timed("Contrefaçons", "SearchAPI · compte", async () => {
        const key = process.env.SEARCHAPI_API_KEY;
        if (!key) return missing("SEARCHAPI_API_KEY");
        const { status, json, text } = await getJson(`https://www.searchapi.io/api/v1/me?${new URLSearchParams({ api_key: key })}`);
        if (status >= 400) return { ok: false, detail: `${status} ${text.slice(0, 200)}` };
        const acc = (json?.account ?? json ?? {}) as Record<string, unknown>;
        const left = typeof acc.remaining_credits === "number" ? acc.remaining_credits : null;
        // Without a subscription the free searches are not reported here (-1); the SearchAPI dashboard shows them.
        if (left === null || left < 0) return { ok: true, detail: `clé valide · offre gratuite, crédits visibles sur le tableau de bord SearchAPI · ${acc.current_month_usage ?? 0} recherche(s) ce mois-ci` };
        return { ok: true, detail: `crédits restants : ${left}` };
      }),
      timed("Contrefaçons", "Google Vision (recherche par image)", async () => {
        const key = process.env.GOOGLE_VISION_API_KEY;
        if (!key) return { ok: null, detail: "non configuré (GOOGLE_VISION_API_KEY) : la recherche visuelle passe par Google Lens (SearchAPI) et la comparaison des captures" };
        // An empty request is free and tells whether the key is valid and the API enabled.
        const res = await fetch(`https://vision.googleapis.com/v1/images:annotate?key=${encodeURIComponent(key)}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ requests: [] }) });
        const text = await res.text();
        return { ok: res.status < 400, detail: res.status < 400 ? "clé valide, API activée" : `${res.status} ${text.replace(/\s+/g, " ").slice(0, 200)}` };
      }),
      timed("Contrefaçons", "urlscan · quotas", async () => {
        const key = process.env.URLSCAN_API_KEY;
        if (!key) return missing("URLSCAN_API_KEY");
        const { status, json, text } = await getJson("https://urlscan.io/user/quotas/", { "API-Key": key });
        if (status >= 400) return { ok: false, detail: `${status} ${text.slice(0, 200)}` };
        const limits = (json?.limits ?? {}) as Record<string, { day?: { used?: number; limit?: number } }>;
        const scans = limits.private ?? limits.unlisted ?? limits.public;
        return { ok: true, detail: scans?.day ? `scans aujourd'hui : ${scans.day.used ?? 0}/${scans.day.limit ?? "?"}` : "clé valide" };
      }),
      timed("Contrefaçons", "RDAP (titulaire d'un domaine)", async () => {
        const d = await lookupDomain("boucheron.com");
        return { ok: true, detail: `boucheron.com · bureau d'enregistrement : ${d.registrar ?? "—"}` };
      }),
      timed("Email", "Resend · domaine d'envoi", async () => {
        const key = process.env.RESEND_API_KEY;
        if (!key) return missing("RESEND_API_KEY");
        const { data, error } = await new Resend(key).domains.list();
        if (error) return { ok: false, detail: error.message };
        const domains = (data?.data ?? []).map((d) => `${d.name} (${d.status})`);
        return { ok: domains.length > 0, detail: domains.join(", ") || "aucun domaine" };
      }),
      timed("Email", "Réception (webhook)", async () => {
        if (!process.env.RESEND_WEBHOOK_SECRET) return missing("RESEND_WEBHOOK_SECRET");
        const rows = await sb.select<{ created_at: string }>("events", { select: "created_at", kind: "eq.email", order: "created_at.desc", limit: "1" });
        return { ok: true, detail: rows[0] ? `dernier email traité : ${new Date(rows[0].created_at).toLocaleString("fr-FR", { timeZone: "Europe/Paris" })}` : "secret présent, aucun email reçu pour l'instant" };
      }),
    ]);
    return all;
  });
  await trace.finish(checks.some((c) => c.ok === false) ? "error" : "done", { checks });
  return { traceId: trace.id, checks };
}
