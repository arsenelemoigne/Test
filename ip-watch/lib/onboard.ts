import { randomUUID } from "crypto";
import { extractJson } from "./llm";
import { traced } from "./log";
import { currentTrace, traceStep } from "./trace";
import { searchCompanies } from "./sources/entreprises";
import * as inpi from "./sources/inpi";
import * as store from "./store";
import { isActiveRight } from "./deadlines";
import type { Client, Company, Right } from "./types";

const PAGE_SIZE = 100;
const MAX_PAGES = 5;
// Full records cost one INPI call each (quota ≈ 100): only the most useful marks are enriched.
const MAX_NOTICES = 20;
const LEGAL_FORMS = /\b(SASU|SAS|SARL|EURL|SNC|SCA|SA|SE|GMBH|LTD|LIMITED|INC|LLC|SPA|BV)\b\.?/gi;

export function slugify(s: string): string {
  return s.normalize("NFD").replace(/[̀-ͯ]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "").slice(0, 40) || "client";
}

const holderName = (s: string) => s.replace(LEGAL_FORMS, "").replace(/[[\]]/g, "").replace(/\s+/g, " ").trim().toUpperCase();

type Extracted = { societe?: string | null; siren?: string | null };

// Reads a free-form message (or a forwarded email) and identifies the client company.
async function identify(text: string): Promise<{ name: string; siren: string | null }> {
  const sirenInText = text.match(/\b(\d{3})\s?(\d{3})\s?(\d{3})\b/);
  const ai = await extractJson<Extracted>(
    "Tu lis un message d'avocat, parfois un email transféré. Identifie la société cliente dont il faut suivre les marques et dessins. " +
      'Réponds en JSON : {"societe": "nom commercial ou dénomination, sans forme juridique", "siren": "9 chiffres ou null"}. N\'invente rien.',
    text.slice(0, 6000),
    "identify-client",
  );
  const siren = (ai?.siren ?? (sirenInText ? sirenInText.slice(1).join("") : null))?.replace(/\s/g, "") ?? null;
  const name = ai?.societe?.trim() || text.replace(/https?:\/\/\S+/g, "").trim().split(/\n/)[0].slice(0, 60);
  return { name, siren: siren && /^\d{9}$/.test(siren) ? siren : null };
}

async function findCompany(name: string, siren: string | null): Promise<Company | null> {
  try {
    const results = await traced("source", "entreprises", { q: siren ?? name }, () => searchCompanies(siren ?? name));
    if (siren) return results.find((c) => c.siren === siren) ?? results[0] ?? null;
    const n = name.toUpperCase();
    return results.find((c) => c.name.toUpperCase().includes(n)) ?? results[0] ?? null;
  } catch {
    return null;
  }
}

// Searches INPI for every title filed under the holder names and stores the raw answers.
async function importTitles(client: Client, holders: string[], runId: string): Promise<{ right: Right; raw: unknown }[]> {
  const found = new Map<string, { right: Right; raw: unknown }>();
  const searches = [
    { kind: "marques" as const, type: "marque" as const, collections: ["FR", "EU", "WO"], fields: inpi.MARK_FIELDS },
    { kind: "modeles" as const, type: "dessin" as const, collections: ["FR", "WO"], fields: ["*"] },
  ];
  for (const holder of holders) {
    for (const s of searches) {
      const query = `[DEPOSANT=${holder}]`;
      for (let page = 0; page < MAX_PAGES; page++) {
        const position = page * PAGE_SIZE;
        const res = await inpi.search(s.kind, query, s.collections, position, PAGE_SIZE, s.fields);
        await store.saveRawImport({ run_id: runId, client_id: client.id, kind: s.kind, query, position, http_status: res.status, payload: res.data });
        if (res.status >= 400) throw new Error(`INPI ${s.kind} ${res.status}`);
        for (const hit of res.hits) {
          const right = inpi.normalize(hit, s.type);
          if (right && !found.has(right.id)) found.set(right.id, { right, raw: hit });
        }
        if (res.hits.length < PAGE_SIZE) break;
      }
    }
  }
  const all = Array.from(found.values());
  traceStep({ kind: "etape", name: "Titres trouvés à l'INPI", output: { marques: all.filter((i) => i.right.type === "marque").length, dessins: all.filter((i) => i.right.type === "dessin").length }, ok: true });
  const kept = await classifyHolders(all, client, holders);

  // Details read on a previous import are reused, so full records are only fetched for what is still missing.
  const known = await store.knownDetails(client.id);
  let reused = 0;
  for (const { right } of kept) {
    right.origin = "inpi";
    const k = known.get(right.id);
    if (!k) continue;
    const before = JSON.stringify([right.filingDate, right.expiryDate, right.classes, right.territories]);
    right.filingDate ??= k.filingDate;
    right.registrationDate ??= k.registrationDate;
    right.expiryDate ??= k.expiryDate;
    if (!right.classes.length) right.classes = k.classes ?? [];
    if (!right.territories.length) right.territories = k.territories ?? [];
    if (JSON.stringify([right.filingDate, right.expiryDate, right.classes, right.territories]) !== before) reused++;
  }
  if (reused) traceStep({ kind: "etape", name: "Détails déjà connus réutilisés", output: { titres: reused }, ok: true });
  await enrichFromNotices(client, kept, runId);
  return kept;
}

const LEGAL_WORDS = new Set(["SA", "SAS", "SASU", "SARL", "EURL", "SE", "SNC", "SOCIETE", "SOCIE", "ANONYME", "PAR", "ACTIONS", "SIMPLIFIEE", "A", "DIRECTOIRE", "ET", "CONSEIL", "DE", "SURVEILLANCE", "STE"]);
// Dots are dropped first so "S.A." and "S.A.R.L." read as SA and SARL.
const normName = (s: string) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").replace(/\./g, "").toUpperCase().replace(/[^A-Z0-9 ]/g, " ").replace(/\s+/g, " ").trim();
// Words a group company adds to the brand ("BOUCHERON HOLDING", "LES MONTRES BOUCHERON"); anything else suggests a namesake.
const GROUP_WORDS = new Set(["HOLDING", "PARFUMS", "PARFUM", "LES", "MONTRES", "JOAILLERIE", "BIJOUX", "HORLOGERIE", "INTERNATIONAL", "GROUPE", "GROUP", "PARIS", "FRANCE", "DISTRIBUTION"]);
// Name without legal-form words: "BOUCHERON, Société par actions simplifiée" and "S.A. BOUCHERON" both give "BOUCHERON".
const coreName = (s: string) => normName(s).split(" ").filter((w) => w && !LEGAL_WORDS.has(w)).join(" ");

type HolderClass = "client" | "groupe" | "homonyme";

// A holder search returns the client, its group (holding, subsidiaries) and namesakes: only the client counts by default.
export async function classifyHolders(items: { right: Right; raw: unknown }[], client: Client, holders: string[]) {
  const counts = new Map<string, number>();
  for (const i of items) counts.set(i.right.owner, (counts.get(i.right.owner) ?? 0) + 1);
  const owners = Array.from(counts.keys()).filter(Boolean);
  const target = coreName(client.company?.legalName ?? holders[0] ?? client.name);
  const brand = coreName(holders[0] ?? client.name);
  // Strict rule: one of the co-holders is exactly the client's name; otherwise group if the brand appears.
  const strict = (o: string): HolderClass => {
    if (/\d{5}/.test(o)) return "homonyme";
    const parts = o.split(/,|;| et /i).map(coreName).filter(Boolean);
    if (parts.includes(target)) return "client";
    const isGroup = parts.some((p) => {
      const words = p.split(" ");
      return words.includes(brand) && words.every((w) => w === brand || GROUP_WORDS.has(w));
    });
    return isGroup ? "groupe" : "homonyme";
  };
  const decided = new Map<string, HolderClass>(owners.map((o) => [o, strict(o)]));

  const ai = await extractJson<{ client?: string[]; groupe?: string[]; homonymes?: string[] }>(
    [
      `Société cliente : ${client.company?.legalName ?? client.name}${client.company ? `, SIREN ${client.company.siren}, siège ${client.company.address ?? "n.c."}, activité ${client.company.activity ?? "n.c."}` : ""}.`,
      "Voici les titulaires (avec leur nombre de titres) trouvés à l'INPI par une recherche sur le nom de déposant.",
      "Classe chaque libellé : « client » = la société cliente elle-même, quelle que soit la façon d'écrire sa forme juridique, y compris en cotitularité ;",
      "« groupe » = sa holding, ses filiales ou sociétés sœurs (ex. « … HOLDING », « … PARFUMS ») ; « homonymes » = personnes physiques ou sociétés sans lien (autre adresse, autre forme ancienne sans rapport).",
      'Réponds en JSON : {"client": ["libellé exact", ...], "groupe": [...], "homonymes": [...]}.',
    ].join("\n"),
    JSON.stringify(owners.map((o) => ({ titulaire: o, titres: counts.get(o) }))),
    "classification des titulaires",
    "strong",
  );
  if (ai) {
    for (const o of ai.client ?? []) if (decided.has(o)) decided.set(o, "client");
    for (const o of ai.groupe ?? []) if (decided.has(o)) decided.set(o, "groupe");
    for (const o of ai.homonymes ?? []) if (decided.has(o)) decided.set(o, "homonyme");
  }
  const by = (c: HolderClass) => owners.filter((o) => decided.get(o) === c).map((o) => `${o} (${counts.get(o)})`);
  traceStep({ kind: "etape", name: "Classement des titulaires", detail: ai ? "règle stricte puis vérification par l'IA" : "règle stricte (IA indisponible)", output: { client: by("client"), groupe: by("groupe"), homonymes_ecartes: by("homonyme") }, ok: true });

  return items
    .filter((i) => decided.get(i.right.owner) !== "homonyme")
    .map((i) => ({ ...i, right: { ...i.right, clientOwned: decided.get(i.right.owner) === "client" } }));
}

const isActive = (status: string) => !/expir|retir|radi|rejet|refus|abandon|non renouvel|annul|caduc/i.test(status);

// Fills dates, classes and designated countries from full records, active international marks first.
async function enrichFromNotices(client: Client, items: { right: Right; raw: unknown }[], runId: string) {
  const missing = items.filter((i) => i.right.type === "marque" && i.right.clientOwned !== false && (!i.right.expiryDate || (i.right.office === "WO" && i.right.territories.length === 0)) && isActive(i.right.status));
  const rank = (r: Right) => (r.office === "WO" ? 0 : r.office === "EU" ? 1 : 2);
  const seenNames = new Set<string>();
  const picked = missing
    .sort((a, b) => rank(a.right) - rank(b.right))
    .filter((i) => {
      const key = `${i.right.office}:${i.right.name}`;
      if (seenNames.has(key)) return false;
      seenNames.add(key);
      return true;
    })
    .slice(0, MAX_NOTICES);
  traceStep({ kind: "etape", name: "Fiches complètes à lire", detail: `${picked.length} sur ${missing.length} titres incomplets (limite ${MAX_NOTICES}, quota INPI)`, ok: true });
  for (const item of picked) {
    const path = inpi.noticePath(item.raw as Record<string, unknown>);
    if (!path) continue;
    try {
      const res = await inpi.notice(path);
      await store.saveRawImport({ run_id: runId, client_id: client.id, kind: "notice", query: path, http_status: res.status, payload: typeof res.data === "string" ? { text: res.data.slice(0, 200_000) } : res.data });
      if (res.status === 429) break;
      if (res.status >= 400) continue;
      const d = inpi.parseNotice(res.data);
      const r = item.right;
      r.filingDate ??= d.filingDate;
      r.registrationDate ??= d.registrationDate;
      r.expiryDate ??= d.expiryDate;
      if (!r.classes.length) r.classes = d.classes;
      if (r.office === "WO" && !r.territories.length) r.territories = d.countries;
      if (!r.status && d.status) r.status = d.status;
    } catch {
      // A missing record must not stop the import.
    }
  }
}

export type OnboardResult = { client: Client; marks: number; designs: number; summary: string };

export async function onboard(text: string, trigger: "email" | "app"): Promise<OnboardResult> {
  const { name, siren } = await identify(text);
  traceStep({ kind: "etape", name: "Client identifié dans le message", output: { societe: name, siren }, ok: true });
  const company = await findCompany(name, siren);
  traceStep({ kind: "etape", name: "Fiche société", output: company ? { denomination: company.legalName ?? company.name, siren: company.siren, siege: company.address } : "aucune société trouvée", ok: Boolean(company) });
  const brand = holderName(name);
  const holders = Array.from(new Set([brand, company?.legalName ? holderName(company.legalName) : ""].filter(Boolean))).slice(0, 2);
  const client = await store.saveClient({ slug: slugify(brand), name: brand.charAt(0) + brand.slice(1).toLowerCase(), siren: company?.siren ?? siren, company, holderNames: holders, status: "importing" });

  try {
    const runId = randomUUID();
    const items = await importTitles(client, holders, runId);
    await store.upsertMarks(client.id, items);
    if (items.length) await store.removeStaleMarks(client.id, items.map((i) => i.right.id));
    const updatedAt = await inpi.metadata();
    await store.updateClient(client.id, { status: "ready", error: null, imported_at: new Date().toISOString(), data_updated_at: updatedAt, holder_names: holders });

    const own = items.filter((i) => i.right.clientOwned !== false);
    const marks = own.filter((i) => i.right.type === "marque");
    const designs = own.length - marks.length;
    const active = own.filter((i) => isActiveRight(i.right));
    const group = items.length - own.length;
    const byOffice = (o: string) => marks.filter((m) => m.right.office === o).length;
    const summary =
      `Titres déposés par ${company?.legalName ?? holders[0]}${company ? ` (SIREN ${company.siren})` : ""} : ` +
      `${marks.length} marques (France ${byOffice("FR")}, UE ${byOffice("EU")}, international ${byOffice("WO")}) et ${designs} dessins & modèles, dont ${active.length} en vigueur.` +
      (group ? ` ${group} titres de sociétés du groupe (holding, filiales) conservés à part.` : "");
    await store.addEvent(client.id, "import", `Import INPI — ${marks.length} marques, ${designs} dessins`, summary, { runId, holders, trigger, siren: company?.siren ?? siren, traceId: currentTrace()?.id });
    return { client: (await store.getClientById(client.id))!, marks: marks.length, designs, summary };
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    await store.updateClient(client.id, { status: "error", error: message });
    await store.addEvent(client.id, "import", "Import INPI échoué", message, { trigger });
    throw e;
  }
}

// Adds specific marks to an existing client, searched by name under the client's holder names.
export async function addMarks(client: Client, markNames: string[]): Promise<number> {
  const added = new Map<string, { right: Right; raw: unknown }>();
  for (const markName of markNames.slice(0, 5)) {
    for (const holder of client.holderNames.length ? client.holderNames : [holderName(client.name)]) {
      const res = await inpi.search("marques", `[Mark_Exp=${holderName(markName)}] ET [DEPOSANT=${holder}]`, ["FR", "EU", "WO"], 0, 50);
      for (const hit of res.hits) {
        const right = inpi.normalize(hit, "marque");
        if (right) added.set(right.id, { right, raw: hit });
      }
    }
  }
  await store.upsertMarks(client.id, Array.from(added.values()));
  await store.addEvent(client.id, "import", `Ajout de marques — ${markNames.join(", ")}`, `${added.size} titre(s) trouvé(s) à l'INPI.`, { markNames });
  return added.size;
}
