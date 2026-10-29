import { piste } from "./piste";

const LF = "/dila/legifrance/lf-engine-app";

export type LawArticle = { id: string; num: string | null; code: string | null; text: string; url: string };

export const articleUrl = (id: string) => `https://www.legifrance.gouv.fr/codes/article_lc/${id}`;

function stripHtml(html: string): string {
  return html.replace(/<br\s*\/?>/gi, "\n").replace(/<\/p>/gi, "\n").replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/\n{3,}/g, "\n\n").trim();
}

// Collects every article hit (LEGIARTI ids) wherever it sits in the search response.
function collectArticles(node: unknown, code: string | null, out: Map<string, { id: string; num: string | null; code: string | null; excerpt: string }>) {
  if (Array.isArray(node)) {
    node.forEach((n) => collectArticles(n, code, out));
    return;
  }
  if (!node || typeof node !== "object") return;
  const o = node as Record<string, unknown>;
  let nextCode = code;
  const titles = o.titles as { title?: string }[] | undefined;
  if (Array.isArray(titles) && titles[0]?.title) nextCode = titles[0].title;
  if (typeof o.id === "string" && o.id.startsWith("LEGIARTI") && !out.has(o.id)) {
    const values = Array.isArray(o.values) ? (o.values as string[]).join(" … ") : typeof o.text === "string" ? o.text : "";
    out.set(o.id, { id: o.id, num: typeof o.num === "string" ? o.num : null, code: nextCode, excerpt: stripHtml(values).slice(0, 500) });
  }
  for (const v of Object.values(o)) if (v && typeof v === "object") collectArticles(v, nextCode, out);
}

async function searchCodes(champs: unknown[], code?: string) {
  const filtres: unknown[] = [{ facette: "DATE_VERSION", singleDate: Date.now() }];
  if (code) filtres.push({ facette: "NOM_CODE", valeurs: [code] });
  const recherche = { champs, filtres, pageNumber: 1, pageSize: 8, operateur: "ET", sort: "PERTINENCE", typePagination: "ARTICLE" };
  try {
    return await piste<unknown>("POST", `${LF}/search`, { fond: "CODE_DATE", recherche });
  } catch {
    // Some gateway versions only accept the in-force fund with a legal status filter.
    const etat = { ...recherche, filtres: [{ facette: "TEXT_LEGAL_STATUS", valeurs: ["VIGUEUR"] }, ...(code ? [{ facette: "NOM_CODE", valeurs: [code] }] : [])] };
    return piste<unknown>("POST", `${LF}/search`, { fond: "CODE_ETAT", recherche: etat });
  }
}

export async function searchLaw(query: string, code?: string) {
  const data = await searchCodes([{ typeChamp: "ALL", criteres: [{ typeRecherche: "UN_DES_MOTS", valeur: query, operateur: "ET" }], operateur: "ET" }], code);
  const out = new Map();
  collectArticles(data, null, out);
  return Array.from(out.values()).slice(0, 8).map((a) => ({ ...a, url: articleUrl(a.id) }));
}

export async function getArticle(id: string): Promise<LawArticle> {
  const data = await piste<{ article?: Record<string, unknown> }>("POST", `${LF}/consult/getArticle`, { id });
  const a = data.article ?? {};
  const html = (a.texteHtml as string) ?? (a.texte as string) ?? "";
  const textTitles = a.textTitles as { titre?: string }[] | undefined;
  return { id, num: (a.num as string) ?? null, code: textTitles?.[0]?.titre ?? null, text: stripHtml(html).slice(0, 3000), url: articleUrl(id) };
}

// Finds an article by its number, e.g. ("Code de la propriété intellectuelle", "L713-2").
export async function findArticle(code: string, num: string): Promise<LawArticle | null> {
  const clean = num.replace(/\s+/g, "").replace(/^art(icle)?\.?/i, "").replace(/^([LRD])\.?/i, "$1");
  const data = await searchCodes([{ typeChamp: "NUM_ARTICLE", criteres: [{ typeRecherche: "EXACTE", valeur: clean, operateur: "ET" }], operateur: "ET" }], code);
  const out = new Map();
  collectArticles(data, null, out);
  const hits = Array.from(out.values()) as { id: string; num: string | null }[];
  const exact = hits.find((h) => h.num?.replace(/\s|\./g, "").toUpperCase() === clean.replace(/\./g, "").toUpperCase()) ?? hits[0];
  return exact ? getArticle(exact.id) : null;
}
