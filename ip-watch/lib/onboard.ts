import { randomUUID } from "crypto";
import { extractJson } from "./llm";
import { traced } from "./log";
import { searchCompanies } from "./sources/entreprises";
import * as inpi from "./sources/inpi";
import * as store from "./store";
import type { Client, Company, Right } from "./types";

const PAGE_SIZE = 100;
const MAX_PAGES = 5;
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
    { kind: "marques" as const, type: "marque" as const, collections: ["FR", "EU", "WO"] },
    { kind: "modeles" as const, type: "dessin" as const, collections: ["FR", "WO"] },
  ];
  for (const holder of holders) {
    for (const s of searches) {
      const query = `[DEPOSANT=${holder}]`;
      for (let page = 0; page < MAX_PAGES; page++) {
        const position = page * PAGE_SIZE;
        const res = await inpi.search(s.kind, query, s.collections, position, PAGE_SIZE);
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
  return Array.from(found.values());
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
    const updatedAt = await inpi.metadata();
    await store.updateClient(client.id, { status: "ready", error: null, imported_at: new Date().toISOString(), data_updated_at: updatedAt, holder_names: holders });

    const marks = items.filter((i) => i.right.type === "marque");
    const designs = items.length - marks.length;
    const byOffice = (o: string) => marks.filter((m) => m.right.office === o).length;
    const summary =
      `Recherche INPI au nom de ${holders.join(" et ")}${company ? ` (SIREN ${company.siren})` : ""} : ` +
      `${marks.length} marques (France ${byOffice("FR")}, UE ${byOffice("EU")}, international ${byOffice("WO")}) et ${designs} dessins & modèles.`;
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
