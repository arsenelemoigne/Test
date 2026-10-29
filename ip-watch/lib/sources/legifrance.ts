import { webSearch } from "../llm";
import { piste } from "./piste";

export const LF = "/dila/legifrance/lf-engine-app";

export type LawArticle = { id: string; num: string | null; code: string | null; text: string; url: string; via?: string };

// LEGITEXT ids of the codes a trademark lawyer cites most, for direct lookup by article number.
export const CODE_IDS: Record<string, string> = {
  "code de la propriété intellectuelle": "LEGITEXT000006069414",
  "code civil": "LEGITEXT000006070721",
  "code de commerce": "LEGITEXT000005634379",
  "code pénal": "LEGITEXT000006070719",
  "code de la consommation": "LEGITEXT000006069565",
  "code de procédure civile": "LEGITEXT000006070716",
  "code des douanes": "LEGITEXT000006071570",
};

const codeId = (code: string) => CODE_IDS[code.trim().toLowerCase().replace(/^cpi$/, "code de la propriété intellectuelle")] ?? null;
const LEGIARTI = /LEGIARTI\d{12}/;

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
  try {
    const data = await searchCodes([{ typeChamp: "ALL", criteres: [{ typeRecherche: "UN_DES_MOTS", valeur: query, operateur: "ET" }], operateur: "ET" }], code);
    const out = new Map();
    collectArticles(data, null, out);
    return Array.from(out.values()).slice(0, 8).map((a) => ({ ...a, url: articleUrl(a.id) }));
  } catch (e) {
    // The search endpoint refused: find Légifrance pages by web search, keep only official article links.
    const web = await webSearch(`${query} ${code ?? ""} site:legifrance.gouv.fr`, "legifrance-web");
    const hits = web.results.filter((r) => r.url.includes("legifrance.gouv.fr") && LEGIARTI.test(r.url));
    if (hits.length === 0) throw e;
    return hits.slice(0, 6).map((r) => ({ id: r.url.match(LEGIARTI)![0], num: null, code: code ?? null, excerpt: r.snippet, url: articleUrl(r.url.match(LEGIARTI)![0]), via: "liens Légifrance trouvés par recherche web" }));
  }
}

function toArticle(id: string, a: Record<string, unknown>): LawArticle {
  const html = (a.texteHtml as string) ?? (a.texte as string) ?? "";
  const textTitles = a.textTitles as { titre?: string }[] | undefined;
  return { id, num: (a.num as string) ?? null, code: textTitles?.[0]?.titre ?? null, text: stripHtml(html).slice(0, 3000), url: articleUrl(id) };
}

export async function getArticle(id: string): Promise<LawArticle> {
  const data = await piste<{ article?: Record<string, unknown> }>("POST", `${LF}/consult/getArticle`, { id });
  return toArticle(id, data.article ?? {});
}

// Finds an article by its number, e.g. ("Code de la propriété intellectuelle", "L713-2").
// Direct lookup first, then the search endpoint, then a web search used only to find the official id.
export async function findArticle(code: string, num: string): Promise<LawArticle | null> {
  const clean = num.replace(/\s+/g, "").replace(/^art(icle)?\.?/i, "").replace(/^([LRD])\.?/i, "$1");
  const errors: string[] = [];
  const id = codeId(code);
  if (id) {
    try {
      const data = await piste<{ article?: Record<string, unknown> }>("POST", `${LF}/consult/getArticleWithIdandNum`, { id, num: clean });
      if (data.article?.id) return toArticle(String(data.article.id), data.article);
    } catch (e) {
      errors.push(e instanceof Error ? e.message : String(e));
    }
  }
  try {
    const data = await searchCodes([{ typeChamp: "NUM_ARTICLE", criteres: [{ typeRecherche: "EXACTE", valeur: clean, operateur: "ET" }], operateur: "ET" }], code);
    const out = new Map();
    collectArticles(data, null, out);
    const hits = Array.from(out.values()) as { id: string; num: string | null }[];
    const exact = hits.find((h) => h.num?.replace(/\s|\./g, "").toUpperCase() === clean.replace(/\./g, "").toUpperCase()) ?? hits[0];
    if (exact) return getArticle(exact.id);
  } catch (e) {
    errors.push(e instanceof Error ? e.message : String(e));
  }
  const web = await webSearch(`article ${clean} ${code} site:legifrance.gouv.fr`, "legifrance-web");
  const hit = web.results.find((r) => r.url.includes("legifrance.gouv.fr") && LEGIARTI.test(r.url));
  if (!hit) {
    if (errors.length) throw new Error(errors.join(" | ").slice(0, 400));
    return null;
  }
  const legiarti = hit.url.match(LEGIARTI)![0];
  try {
    const article = await getArticle(legiarti);
    return { ...article, code: article.code ?? code, num: article.num ?? clean, via: "identifiant trouvé par recherche web, texte lu sur Légifrance" };
  } catch {
    return { id: legiarti, num: clean, code, text: hit.snippet, url: articleUrl(legiarti), via: "extrait de recherche web (Légifrance indisponible) : à vérifier" };
  }
}
