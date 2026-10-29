import { randomUUID } from "crypto";
import { extractJson } from "./llm";
import { traced } from "./log";
import { searchCompanies } from "./sources/entreprises";
import * as inpi from "./sources/inpi";
import * as store from "./store";
import { isActiveRight } from "./deadlines";
import type { Client, Company, Right } from "./types";

const PAGE_SIZE = 100;
const MAX_PAGES = 5;
// Full records cost one INPI call each (quota ≈ 100): only the most useful marks are enriched.
const MAX_NOTICES = 30;
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
  const kept = await keepGroupHolders(all, client, holders);
  await enrichFromNotices(client, kept, runId);
  return kept;
}

const CORPORATE_WORDS = new Set(["SA", "SAS", "SASU", "SARL", "SE", "HOLDING", "GROUPE", "GROUP", "PARFUMS", "PARIS", "LES", "MONTRES", "JOAILLERIE", "INTERNATIONAL", "SOCIETE", "ANONYME", "PAR", "ACTIONS", "SIMPLIFIEE", "ET", "CIE", "FRANCE"]);
const normName = (s: string) => s.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toUpperCase().replace(/[^A-Z0-9 ]/g, " ").replace(/\s+/g, " ").trim();

// A holder search also returns namesakes (people or companies sharing the name): keep the client's group only.
async function keepGroupHolders(items: { right: Right; raw: unknown }[], client: Client, holders: string[]) {
  const owners = Array.from(new Set(items.map((i) => i.right.owner).filter(Boolean)));
  if (owners.length === 0) return items;
  const brands = holders.map(normName);
  // Strict fallback: besides the brand, only corporate words (legal forms, "holding", "parfums"…).
  const strict = (o: string) => {
    const words = normName(o).split(" ");
    return brands.some((b) => normName(o).includes(b)) && words.every((w) => brands.some((b) => b.split(" ").includes(w)) || CORPORATE_WORDS.has(w));
  };
  const ai = await extractJson<{ retenus?: string[] }>(
    [
      `Société cliente : ${client.name}${client.company ? ` (SIREN ${client.company.siren}, ${client.company.address ?? ""}, activité ${client.company.activity ?? "n.c."})` : ""}.`,
      "Voici les titulaires trouvés à l'INPI pour une recherche par nom de déposant.",
      "Retiens uniquement ceux qui appartiennent au groupe du client : la société, ses filiales, ses holdings, ses anciennes dénominations. Écarte les homonymes : personnes physiques, autres sociétés sans lien.",
      'Réponds en JSON : {"retenus": ["libellé exact", ...]}.',
    ].join("\n"),
    JSON.stringify(owners),
    "holder-filter",
  );
  const allowed = new Set((ai?.retenus ?? owners.filter(strict)).map(normName));
  return items.filter((i) => allowed.has(normName(i.right.owner)));
}

const isActive = (status: string) => !/expir|retir|radi|rejet|refus|abandon|non renouvel|annul|caduc/i.test(status);

// Fills dates, classes and designated countries from full records, active international marks first.
async function enrichFromNotices(client: Client, items: { right: Right; raw: unknown }[], runId: string) {
  const missing = items.filter((i) => i.right.type === "marque" && (!i.right.expiryDate || (i.right.office === "WO" && i.right.territories.length === 0)) && isActive(i.right.status));
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
  const company = await findCompany(name, siren);
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

    const marks = items.filter((i) => i.right.type === "marque");
    const designs = items.length - marks.length;
    const active = items.filter((i) => isActiveRight(i.right));
    const byOffice = (o: string) => marks.filter((m) => m.right.office === o).length;
    const summary =
      `Recherche INPI au nom de ${holders.join(" et ")}${company ? ` (SIREN ${company.siren})` : ""} : ` +
      `${marks.length} marques (France ${byOffice("FR")}, UE ${byOffice("EU")}, international ${byOffice("WO")}) et ${designs} dessins & modèles, dont ${active.length} titres en vigueur.`;
    await store.addEvent(client.id, "import", `Import INPI — ${marks.length} marques, ${designs} dessins`, summary, { runId, holders, trigger, siren: company?.siren ?? siren });
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
