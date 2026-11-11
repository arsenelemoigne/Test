import { randomUUID } from "crypto";
import sharp from "sharp";
import { APP_NAME, FIRM_NAME } from "./config";
import * as files from "./files";
import { extractJson, llmConfigured, webSearch, type ContentPart } from "./llm";
import { traced } from "./log";
import * as apify from "./sources/apify";
import * as searchapi from "./sources/searchapi";
import * as vision from "./sources/vision";
import * as store from "./store";
import { runInTrace, startTrace, traceStep } from "./trace";
import type { Client } from "./types";
import { buildVisualPdf, buildVisualXlsx } from "./visual-report";
import { pageImages } from "./sources/pages";
import { VISUAL_STEPS } from "./visual-steps";
import { levelCounts, MODE_LABEL, type VisualAnalysis, type VisualLevel, type VisualMode, type VisualResult, type VisualSearch, type VisualStep } from "./visual-types";

const [S_ANALYSE, S_VISION, S_LENS, S_META, S_SHOPPING, S_IMAGES, S_WEB, S_COPY, S_COMPARE, S_REPORT] = VISUAL_STEPS;

const MAX_RESULTS = 120;
const MAX_COMPARED = 90;
const IMAGE_BATCH = 8;
const TEXT_BATCH = 30;
const UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36";

type Queries = { annonces: string[]; produits: string; images: string[]; web: string };
type Done = (search: VisualSearch, pdf: Uint8Array | null) => Promise<void>;

const appUrl = () => (process.env.APP_URL || "https://pi.getmanda.eu").replace(/\/$/, "");
const folderOf = (clientId: string | null, id: string) => `${clientId ?? "sans-client"}/visual/${id}`;
const dataUri = (bytes: Buffer) => `data:image/jpeg;base64,${bytes.toString("base64")}`;
const LEVELS: VisualLevel[] = ["tres_proche", "proche", "meme_theme", "ecarte"];
const levelRank = (l: VisualLevel | null) => (l ? LEVELS.indexOf(l) : 4);

export function sortResults(results: VisualResult[]): VisualResult[] {
  return [...results].sort((a, b) => levelRank(a.level) - levelRank(b.level) || (b.score ?? -1) - (a.score ?? -1) || a.n - b.n);
}

// Any uploaded format becomes a JPEG of reasonable size; the original orientation is kept.
export async function normalizeImage(bytes: Buffer, max = 1600): Promise<Buffer> {
  return sharp(bytes, { failOn: "none" }).rotate().resize({ width: max, height: max, fit: "inside", withoutEnlargement: true }).flatten({ background: "#ffffff" }).jpeg({ quality: 85 }).toBuffer();
}

async function fetchImage(url: string): Promise<Buffer | null> {
  if (url.startsWith("data:image/")) {
    const comma = url.indexOf(",");
    return comma > 0 ? Buffer.from(url.slice(comma + 1), "base64") : null;
  }
  const tryUrl = async (u: string) => {
    try {
      const res = await fetch(u, { headers: { "User-Agent": UA, Accept: "image/*,*/*;q=0.8" }, signal: AbortSignal.timeout(12_000) });
      const type = res.headers.get("content-type") ?? "";
      if (!res.ok || !/^image\//.test(type)) return null;
      const buf = Buffer.from(await res.arrayBuffer());
      return buf.byteLength > 200 && buf.byteLength < 8_000_000 ? buf : null;
    } catch {
      return null;
    }
  };
  return tryUrl(url);
}

// Runs async work with a fixed number of parallel tasks.
function limiter(n: number) {
  let active = 0;
  const queue: (() => void)[] = [];
  return async <T>(fn: () => Promise<T>): Promise<T> => {
    if (active >= n) await new Promise<void>((r) => queue.push(r));
    active++;
    try {
      return await fn();
    } finally {
      active--;
      queue.shift()?.();
    }
  };
}

const normalizeUrl = (u: string) => {
  try {
    const x = new URL(u);
    x.hash = "";
    for (const k of Array.from(x.searchParams.keys())) if (/^(utm_|fbclid|gclid|srsltid)/.test(k)) x.searchParams.delete(k);
    return `${x.hostname.replace(/^www\./, "")}${x.pathname.replace(/\/$/, "")}${x.search}`.toLowerCase();
  } catch {
    return u.toLowerCase();
  }
};

export async function startVisualSearch(opts: { client: Client | null; mode: VisualMode; note: string | null; image: Buffer; origin: "depot" | "email"; onDone?: Done }): Promise<VisualSearch> {
  const id = randomUUID();
  const jpeg = await normalizeImage(opts.image);
  const imagePath = `${folderOf(opts.client?.id ?? null, id)}/image.jpg`;
  await files.putFile(imagePath, jpeg, "image/jpeg");
  const trace = await startTrace("visuel", `Recherche visuelle — ${MODE_LABEL[opts.mode]}`, opts.client?.id ?? null);
  const search = await store.createVisualSearch({ id, client_id: opts.client?.id ?? null, mode: opts.mode, origin: opts.origin, note: opts.note, image_path: imagePath, trace_id: trace.id });
  // Runs in the background; the page polls the search for steps and results.
  void runInTrace(trace, () => execute(search, opts.client, jpeg, opts.onDone))
    .then(() => trace.finish("done", { visualSearchId: id }))
    .catch(async (e) => {
      const message = e instanceof Error ? e.message : String(e);
      await store.updateVisualSearch(id, { status: "error", error: message, finished_at: new Date().toISOString() }).catch(() => undefined);
      await trace.finish("error", { error: message });
    });
  return search;
}

class Skip extends Error {}

async function execute(search: VisualSearch, client: Client | null, jpeg: Buffer, onDone?: Done) {
  const steps: VisualStep[] = [];
  const results: VisualResult[] = [];
  const seen = new Set<string>();
  let analysis: VisualAnalysis | null = null;
  const folder = folderOf(client?.id ?? null, search.id);

  // Saves are chained so an older snapshot never overwrites a newer one.
  let saving: Promise<void> = Promise.resolve();
  const save = () => {
    const patch = { steps: steps.map((s) => ({ ...s })), results: results.map((r) => ({ ...r })), analysis };
    saving = saving.then(() => store.updateVisualSearch(search.id, patch)).catch(() => undefined);
    return saving;
  };
  const setStep = (name: string, patch: Partial<VisualStep>) => {
    const s = steps.find((x) => x.name === name);
    if (s) Object.assign(s, patch);
    else steps.push({ name, status: "running", ...patch });
    void save();
  };
  const step = async (name: string, fn: () => Promise<string | void>) => {
    setStep(name, { status: "running" });
    const started = Date.now();
    try {
      const detail = (await fn()) ?? undefined;
      setStep(name, { status: "done", detail });
      traceStep({ kind: "etape", name, detail, ms: Date.now() - started, ok: true });
    } catch (e) {
      const skipped = e instanceof Skip;
      const detail = e instanceof Error ? e.message : String(e);
      setStep(name, { status: skipped ? "skipped" : "error", detail: detail.slice(0, 200) });
      traceStep({ kind: "etape", name, detail, ms: Date.now() - started, ok: skipped });
    }
  };
  const add = (items: Omit<VisualResult, "n" | "imagePath" | "level" | "score" | "similar" | "differs" | "category">[]): VisualResult[] => {
    const added: VisualResult[] = [];
    for (const it of items) {
      if (results.length >= MAX_RESULTS) break;
      const key = normalizeUrl(it.url);
      if (!it.url || seen.has(key)) continue;
      seen.add(key);
      const r: VisualResult = { ...it, title: (it.title || it.url).slice(0, 300), n: results.length + 1, imagePath: null, level: null, score: null, similar: null, differs: null, category: null };
      results.push(r);
      added.push(r);
    }
    void save();
    return added;
  };

  // ---- 1. What the image shows, and the queries it suggests.
  const context = await clientContext(client, search.mode);
  const analysisDone = (async () => {
    let queries: Queries | null = null;
    await step(S_ANALYSE, async () => {
      const out = await analyse(jpeg, search.mode, search.note, context);
      analysis = out.analysis;
      queries = out.queries;
      void save();
      return `${analysis.titre} · ${analysis.elements.length} élément(s)`;
    });
    return queries ?? fallbackQueries(search.note);
  })();

  // ---- 2. Downloads and visual comparison run as soon as each source answers.
  const download = limiter(8);
  const llm = limiter(3);
  let copied = 0;
  let compared = 0;
  const work: Promise<void>[] = [];
  const feed = (items: VisualResult[]) => {
    if (!items.length) return;
    work.push(
      (async () => {
        setStep(S_COPY, { status: "running", detail: `${copied} visuel(s) copié(s)` });
        await Promise.all(
          items.filter((r) => r.imageUrl).map((r) =>
            download(async () => {
              const raw = await fetchImage(r.imageUrl!);
              if (!raw) return;
              try {
                const meta = await sharp(raw, { failOn: "none" }).metadata();
                if ((meta.width ?? 0) < 40 || (meta.height ?? 0) < 40) return;
                const thumb = await normalizeImage(raw, 480);
                const path = `${folder}/r${r.n}.jpg`;
                await files.putFile(path, thumb, "image/jpeg");
                r.imagePath = path;
                copied++;
              } catch {
                // An unreadable image is simply left out of the comparison.
              }
            }),
          ),
        );
        setStep(S_COPY, { status: "running", detail: `${copied} visuel(s) copié(s)` });
        await analysisDone;
        await compare(items, llm);
        compared = results.filter((r) => r.level).length;
        setStep(S_COMPARE, { status: "running", detail: `${compared} résultat(s) évalué(s)` });
      })().catch(() => undefined),
    );
  };

  async function compare(items: VisualResult[], run: ReturnType<typeof limiter>) {
    // An identical image found by Vision is, by definition, a very close match.
    for (const r of items) {
      if (r.source === "Google Vision" && r.foundBy.includes("identique") && !r.imagePath) {
        Object.assign(r, { level: "tres_proche", score: 95, similar: "Image identique publiée sur cette page (Google Vision)", differs: null });
      }
    }
    const pending = items.filter((r) => !r.level);
    const withImage = pending.filter((r) => r.imagePath).slice(0, Math.max(0, MAX_COMPARED - compared));
    const textOnly = pending.filter((r) => !r.imagePath);
    const batches: VisualResult[][] = [];
    for (let i = 0; i < withImage.length; i += IMAGE_BATCH) batches.push(withImage.slice(i, i + IMAGE_BATCH));
    const textBatches: VisualResult[][] = [];
    for (let i = 0; i < textOnly.length; i += TEXT_BATCH) textBatches.push(textOnly.slice(i, i + TEXT_BATCH));
    setStep(S_COMPARE, { status: "running", detail: `${compared} résultat(s) évalué(s)` });
    await Promise.all([
      ...batches.map((b) => run(() => rateBatch(b, true))),
      ...textBatches.map((b) => run(() => rateBatch(b, false))),
    ]);
    void save();
  }

  async function rateBatch(batch: VisualResult[], withImages: boolean) {
    if (!llmConfigured()) return;
    const elements = (analysis?.elements ?? []).map((e) => `${e.id} : ${e.label}`).join("\n");
    const parts: ContentPart[] = [{ type: "text", text: `${ratingIntro(search.mode, context)}\n\nÉléments de l'image 0 :\n${elements || "(non décrits)"}\n${search.note ? `Précisions de l'avocat : ${search.note}\n` : ""}\nImage 0 (référence) :` }, { type: "image_url", image_url: { url: dataUri(jpeg) } }];
    for (const r of batch) {
      parts.push({ type: "text", text: `Résultat n°${r.n} — ${r.source}${r.advertiser ? ` — ${r.advertiser}` : ""} — ${r.title.slice(0, 140)} — ${hostOf(r.url)}${r.date ? ` — ${r.date}` : ""}` });
      if (withImages && r.imagePath) {
        const f = await files.getFile(r.imagePath);
        if (f) parts.push({ type: "image_url", image_url: { url: dataUri(f.bytes) } });
      }
    }
    if (!withImages) parts.push({ type: "text", text: "Ces résultats n'ont pas d'image exploitable : juge sur le titre et le site, et ne dépasse pas « meme_theme » sauf si le titre désigne clairement la même campagne ou le même produit." });
    const out = await extractJson<{ resultats?: { n: number; niveau?: string; score?: number; similaire?: string; differe?: string; categorie?: string }[] }>(
      [
        "Tu réponds uniquement en JSON.",
        'Format : {"resultats":[{"n":12,"niveau":"tres_proche|proche|meme_theme|ecarte","score":0,"similaire":"15 mots max","differe":"15 mots max","categorie":"copie|usage_du_signe|officiel|revente|sans_rapport|null"}]}',
        "Un objet par résultat reçu, avec le même numéro n. Écris en français.",
      ].join("\n"),
      parts,
      withImages ? "visual-compare" : "visual-compare-text",
      "strong",
    );
    for (const o of out?.resultats ?? []) {
      const r = batch.find((x) => x.n === Number(o.n));
      if (!r) continue;
      let level = (LEVELS as string[]).includes(o.niveau ?? "") ? (o.niveau as VisualLevel) : "meme_theme";
      if (!withImages && (level === "tres_proche" || level === "proche") && !(r.source === "Google Vision")) level = level === "tres_proche" ? "proche" : "meme_theme";
      if (search.mode === "contrefacon" && (o.categorie === "officiel" || o.categorie === "sans_rapport")) level = "ecarte";
      r.level = level;
      r.score = typeof o.score === "number" ? Math.max(0, Math.min(100, Math.round(o.score))) : null;
      r.similar = o.similaire?.trim() || null;
      r.differs = o.differe?.trim() || null;
      r.category = search.mode === "contrefacon" && o.categorie && o.categorie !== "null" ? o.categorie.replace(/_/g, " ") : null;
    }
  }

  // ---- 3. Sources. Vision and Lens only need the image; the others wait for the queries.
  const sources: Promise<void>[] = [];
  sources.push(
    step(S_VISION, async () => {
      if (!vision.visionConfigured()) throw new Skip("clé Google Vision absente");
      const ab = jpeg.buffer.slice(jpeg.byteOffset, jpeg.byteOffset + jpeg.byteLength) as ArrayBuffer;
      const r = await traced("source", "vision.web", { image: "image déposée" }, () => vision.webDetection({ bytes: ab }), (x) => ({ pages: x.pages.length, similaires: x.similarImages.length, devine: x.guess }));
      const pages = add(
        r.pages.slice(0, 25).map((p) => ({ source: "Google Vision", kind: "image" as const, title: p.title, url: p.url, imageUrl: p.image, advertiser: hostOf(p.url), date: null, country: null, price: null, currency: null, foundBy: `image déposée (correspondance ${p.match})` })),
      );
      const similar = add(
        r.similarImages.slice(0, 10).map((u) => ({ source: "Google Vision", kind: "image" as const, title: `Image visuellement proche — ${hostOf(u)}`, url: u, imageUrl: u, advertiser: hostOf(u), date: null, country: null, price: null, currency: null, foundBy: "image déposée (image similaire)" })),
      );
      feed([...pages, ...similar]);
      return `${r.pages.length} page(s) avec la même image, ${r.similarImages.length} image(s) similaire(s)${r.guess ? ` · « ${r.guess} »` : ""}`;
    }),
  );
  sources.push(
    step(S_LENS, async () => {
      if (!searchapi.searchApiConfigured()) throw new Skip("clé SearchAPI absente");
      const url = await files.temporaryUrl(`${folder}/image.jpg`, 900);
      if (!url) throw new Skip("pas de lien temporaire pour l'image (stockage local)");
      const hits = await traced("source", "searchapi.lens", { image: "image déposée" }, () => searchapi.lens(url), (x) => ({ resultats: x.length }));
      feed(add(hits.slice(0, 30).map((h) => ({ source: "Google Lens", kind: h.price !== null ? ("produit" as const) : ("image" as const), title: h.title, url: h.url, imageUrl: h.thumbnail, advertiser: h.seller, date: null, country: null, price: h.price, currency: h.currency, foundBy: "image déposée (Lens)" }))));
      return `${hits.length} image(s) ressemblante(s)`;
    }),
  );

  const queries = await analysisDone;
  sources.push(
    step(S_META, async () => {
      if (!apify.apifyConfigured()) throw new Skip("jeton Apify absent");
      const qs = queries.annonces.slice(0, 2);
      if (!qs.length) throw new Skip("aucune requête");
      const ads = await traced("source", "apify.meta-ads", { requetes: qs.join(" | ") }, () => apify.metaAds(qs, { country: "FR", maxAds: 20 }), (x) => ({ publicites: x.length }));
      feed(
        add(
          ads.slice(0, 30).map((a) => ({
            source: "Meta Ad Library",
            kind: "publicite" as const,
            title: (a.text ?? a.pageName ?? "Publicité Meta").replace(/\s+/g, " ").slice(0, 160),
            url: a.url,
            imageUrl: a.image,
            advertiser: a.pageName,
            date: a.startDate,
            country: "FR",
            price: null,
            currency: null,
            foundBy: a.query ?? qs.join(" | "),
          })),
        ),
      );
      return `${ads.length} publicité(s) · ${qs.map((q) => `« ${q} »`).join(", ")}`;
    }),
  );
  sources.push(
    step(S_SHOPPING, async () => {
      if (!searchapi.searchApiConfigured()) throw new Skip("clé SearchAPI absente");
      if (!queries.produits) throw new Skip("aucune requête");
      const hits = await traced("source", "searchapi.shopping", { q: queries.produits }, () => searchapi.shopping(queries.produits), (x) => ({ resultats: x.length }));
      feed(add(hits.slice(0, 15).map((h) => ({ source: "Google Shopping", kind: "produit" as const, title: h.title, url: h.url, imageUrl: h.thumbnail, advertiser: h.seller, date: null, country: null, price: h.price, currency: h.currency, foundBy: queries.produits }))));
      return `${hits.length} produit(s) · « ${queries.produits} »`;
    }),
  );
  sources.push(
    step(S_IMAGES, async () => {
      if (!searchapi.searchApiConfigured()) throw new Skip("clé SearchAPI absente");
      const qs = queries.images.slice(0, 2);
      if (!qs.length) throw new Skip("aucune requête");
      let n = 0;
      for (const q of qs) {
        const hits = await traced("source", "searchapi.images", { q }, () => searchapi.images(q), (x) => ({ resultats: x.length }));
        n += hits.length;
        feed(add(hits.slice(0, 15).map((h) => ({ source: "Google Images", kind: "image" as const, title: h.title, url: h.url, imageUrl: h.image, advertiser: h.source, date: null, country: null, price: null, currency: null, foundBy: q }))));
      }
      return `${n} visuel(s) · ${qs.map((q) => `« ${q} »`).join(", ")}`;
    }),
  );
  // Press articles and campaign pages found by a web search; each page's preview image is used for the comparison.
  sources.push(
    step(S_WEB, async () => {
      if (!llmConfigured()) throw new Skip("IA non configurée");
      const r = await webSearch(queries.web, "visual-web", {
        maxResults: 10,
        system: "Trouve des campagnes publicitaires, articles de presse spécialisée ou produits existants qui correspondent à la description. Réponds en 3 phrases maximum, en français.",
      });
      const pages = add(r.results.map((w) => ({ source: "Web", kind: "page" as const, title: w.title, url: w.url, imageUrl: null, advertiser: hostOf(w.url), date: null, country: null, price: null, currency: null, foundBy: queries.web })));
      const preview = limiter(5);
      await Promise.all(
        pages.map((p) =>
          preview(async () => {
            const imgs = await pageImages(p.url, "", 1).catch(() => [] as string[]);
            if (imgs[0]) {
              p.imageUrl = imgs[0];
              p.kind = "image";
            }
          }),
        ),
      );
      feed(pages);
      return `${r.results.length} page(s), ${pages.filter((p) => p.imageUrl).length} avec visuel · « ${queries.web} »`;
    }),
  );

  await Promise.all(sources);
  // Work may be added while earlier batches finish; wait until nothing is left.
  for (let i = 0; i < work.length; i++) await work[i];
  setStep(S_COPY, { status: results.some((r) => r.imageUrl) ? "done" : "skipped", detail: `${copied} visuel(s) copié(s) dans le dossier` });
  const counts = levelCounts(results);
  setStep(S_COMPARE, { status: llmConfigured() ? "done" : "skipped", detail: `${results.filter((r) => r.level).length} évalué(s) : ${counts.tres_proche} très proche(s), ${counts.proche} proche(s), ${counts.meme_theme} même thème` });
  results.splice(0, results.length, ...sortResults(results));

  // ---- 4. Reports, filed with the case.
  let pdf: Uint8Array | null = null;
  let pdfPath: string | null = null;
  let xlsxPath: string | null = null;
  await step(S_REPORT, async () => {
    const snapshot: VisualSearch = { ...search, status: "done", analysis, steps, results, finishedAt: new Date().toISOString() };
    const ctx = {
      clientName: client?.name ?? "Sans client",
      appName: APP_NAME,
      firmName: FIRM_NAME,
      appUrl: appUrl(),
      sourceImage: jpeg,
      getImage: async (path: string) => (await files.getFile(path))?.bytes ?? null,
    };
    const built = await buildVisualPdf(snapshot, ctx);
    pdf = built;
    pdfPath = `${folder}/rapport.pdf`;
    await files.putFile(pdfPath, built, "application/pdf");
    const xlsx = await buildVisualXlsx(snapshot, ctx);
    xlsxPath = `${folder}/resultats.xlsx`;
    await files.putFile(xlsxPath, xlsx, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
    return "rapport PDF et tableur Excel ajoutés au dossier";
  });

  await save();
  await store.updateVisualSearch(search.id, { status: "done", finished_at: new Date().toISOString(), report_pdf_path: pdfPath, report_xlsx_path: xlsxPath, steps, results, analysis });
  const title = (analysis as VisualAnalysis | null)?.titre ?? "image";
  await store.addEvent(
    client?.id ?? null,
    "recherche",
    `Recherche visuelle — ${title}`,
    `${MODE_LABEL[search.mode]} : ${results.length} résultat(s), ${counts.tres_proche} très proche(s), ${counts.proche} proche(s), ${counts.meme_theme} même thème.`,
    { visualSearchId: search.id, traceId: search.traceId, pdf: pdfPath, origin: search.origin },
  );
  if (onDone) {
    const final = await store.getVisualSearch(search.id);
    if (final) await onDone(final, pdf).catch(() => undefined);
  }
}

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url.slice(0, 40);
  }
}

type Context = { client: string | null; holders: string[]; marks: string[] };

async function clientContext(client: Client | null, mode: VisualMode): Promise<Context> {
  if (!client) return { client: null, holders: [], marks: [] };
  const marks = mode === "contrefacon" ? await store.listMarks(client.id).catch(() => []) : [];
  const names = Array.from(new Set(marks.filter((m) => m.clientOwned !== false).map((m) => m.name.trim()).filter(Boolean))).slice(0, 25);
  return { client: client.name, holders: client.holderNames, marks: names };
}

function fallbackQueries(note: string | null): Queries {
  const base = (note ?? "").split(/[.,;\n]/)[0].trim().slice(0, 60) || "campagne publicitaire";
  return { annonces: [base], produits: base, images: [`${base} campagne`], web: `${base} campagne publicitaire` };
}

async function analyse(jpeg: Buffer, mode: VisualMode, note: string | null, ctx: Context): Promise<{ analysis: VisualAnalysis; queries: Queries }> {
  const fallback = fallbackQueries(note);
  const empty: VisualAnalysis = { titre: note?.slice(0, 60) || "Image déposée", type: "autre", description: note ?? "", elements: [{ id: "E0", label: note ?? "image déposée" }], motsCles: [], keywordsEn: [], marquesVisibles: [], texteVisible: null };
  if (!llmConfigured()) return { analysis: empty, queries: fallback };
  type Out = {
    titre?: string; type?: string; description?: string; elements?: { id?: string; label?: string }[]; mots_cles?: string[]; keywords_en?: string[];
    marques_visibles?: string[]; texte_visible?: string | null; requetes?: { annonces?: string[]; produits?: string; images?: string[]; web?: string };
  };
  const out = await extractJson<Out>(
    [
      "Tu assistes un avocat en propriété intellectuelle. On te montre une image : une publicité, un visuel de campagne ou une photo de produit.",
      mode === "anteriorite"
        ? "Le client prépare un lancement : il faut trouver ce qui existe déjà en ligne (campagnes, publicités, produits, visuels) avec la même idée ou la même mise en scène."
        : `Il faut trouver en ligne des copies ou imitations de ce produit ou visuel du client${ctx.client ? ` (${ctx.client}${ctx.marks.length ? `, marques : ${ctx.marks.slice(0, 12).join(", ")}` : ""})` : ""}.`,
      "Décris l'image de façon factuelle et découpe-la en éléments : E0 est le concept global en une phrase ; E1 à E6 : sujet principal, décor, saison et lumière, cadrage et composition, texte ou slogan, personnes ou lieux (n'en garde que ce qui est visible).",
      "Prépare des requêtes courtes (2 à 6 mots), une idée chacune :",
      "- annonces : 2 requêtes comme les mots d'un texte publicitaire (la bibliothèque publicitaire de Meta cherche dans le TEXTE des annonces, pas dans l'image), une en français puis une en anglais ;",
      "- produits : une requête Google Shopping décrivant l'objet (forme, matière, motif) ;",
      `- images : 2 requêtes Google Images visant des campagnes ou visuels semblables${mode === "anteriorite" ? ", sans nom de marque" : " (une avec la marque du client, une décrivant l'objet)"} ;`,
      "- web : une requête web pour trouver des campagnes ou articles de presse semblables.",
      'Réponds en JSON : {"titre":"6 mots maximum","type":"publicite|produit|autre","description":"2 phrases","elements":[{"id":"E0","label":"..."}],"mots_cles":["..."],"keywords_en":["..."],"marques_visibles":[],"texte_visible":null,"requetes":{"annonces":["",""],"produits":"","images":["",""],"web":""}}',
    ].join("\n"),
    [
      { type: "text", text: note ? `Précisions de l'avocat : ${note}` : "Pas de précision de l'avocat." },
      { type: "image_url", image_url: { url: dataUri(jpeg) } },
    ],
    "visual-analyse",
    "strong",
  );
  if (!out) return { analysis: empty, queries: fallback };
  const clean = (a: unknown, n: number) => (Array.isArray(a) ? a.filter((x): x is string => typeof x === "string" && x.trim().length > 0).map((x) => x.trim()).slice(0, n) : []);
  const elements = (out.elements ?? []).filter((e) => e?.label).map((e, i) => ({ id: e.id?.trim() || `E${i}`, label: String(e.label).trim() })).slice(0, 8);
  const analysis: VisualAnalysis = {
    titre: out.titre?.trim() || empty.titre,
    type: out.type === "publicite" || out.type === "produit" ? out.type : "autre",
    description: out.description?.trim() || "",
    elements: elements.length ? elements : empty.elements,
    motsCles: clean(out.mots_cles, 8),
    keywordsEn: clean(out.keywords_en, 8),
    marquesVisibles: clean(out.marques_visibles, 6),
    texteVisible: typeof out.texte_visible === "string" && out.texte_visible.trim() ? out.texte_visible.trim() : null,
  };
  const q = out.requetes ?? {};
  const queries: Queries = {
    annonces: clean(q.annonces, 2).length ? clean(q.annonces, 2) : fallback.annonces,
    produits: typeof q.produits === "string" && q.produits.trim() ? q.produits.trim() : analysis.motsCles.slice(0, 3).join(" ") || fallback.produits,
    images: clean(q.images, 2).length ? clean(q.images, 2) : fallback.images,
    web: typeof q.web === "string" && q.web.trim() ? q.web.trim() : fallback.web,
  };
  traceStep({ kind: "etape", name: "Requêtes préparées", output: queries, ok: true });
  return { analysis, queries };
}

function ratingIntro(mode: VisualMode, ctx: Context): string {
  if (mode === "anteriorite") {
    return [
      "Tu aides un avocat en propriété intellectuelle à vérifier, avant un lancement, ce qui existe déjà. L'image 0 est le projet du client (publicité, visuel ou produit).",
      "Compare chaque résultat numéroté à l'image 0 et classe-le :",
      "- tres_proche : même concept ET même mise en scène (même sujet dans le même décor), ou même slogan, ou même produit / même design ;",
      "- proche : au moins deux éléments en commun, ou même concept avec une autre exécution ;",
      "- meme_theme : un seul élément en commun ;",
      "- ecarte : aucun élément en commun, page sans rapport ou image illisible.",
      "En cas de doute entre deux niveaux, choisis le plus élevé. score = ressemblance de 0 à 100. similaire / differe : composition, cadrage, lumière, décor, sujet, couleurs, texte.",
    ].join("\n");
  }
  return [
    `Tu aides un avocat en propriété intellectuelle à repérer des copies. L'image 0 montre le produit ou le visuel du client${ctx.client ? ` (${ctx.client})` : ""}${ctx.marks.length ? ` ; ses marques : ${ctx.marks.slice(0, 15).join(", ")}` : ""}.`,
    "Compare chaque résultat numéroté à l'image 0 et classe-le :",
    "- tres_proche : copie quasi identique, ou même signe sur un produit semblable ;",
    "- proche : reprise des caractéristiques essentielles (forme, motif, combinaison de couleurs) ;",
    "- meme_theme : même type de produit, ressemblance faible ;",
    "- ecarte : sans rapport, ou produit officiel du client.",
    "categorie : copie, usage_du_signe (le signe du client utilisé par un tiers), officiel (le client ou un revendeur agréé), revente (produit authentique d'occasion), sans_rapport.",
    "score = ressemblance de 0 à 100. similaire / differe : forme, motif, matière, couleurs, signe, présentation.",
  ].join("\n");
}
