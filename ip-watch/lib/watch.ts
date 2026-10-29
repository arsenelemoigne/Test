import { createHash } from "crypto";
import { extractJson, llmConfigured, webSearch, type ContentPart } from "./llm";
import { traced } from "./log";
import { slugify } from "./onboard";
import { pageImages } from "./sources/pages";
import * as searchapi from "./sources/searchapi";
import * as urlscan from "./sources/urlscan";
import * as vision from "./sources/vision";
import { lookupDomain, registrableDomain } from "./sources/rdap";
import * as store from "./store";
import { uploadObject } from "./supabase";
import { runInTrace, startTrace, traceStep } from "./trace";
import type { Client, Finding, FindingKind, Mark, RunStats, WatchRun } from "./types";

type Candidate = { kind: FindingKind; source: string; title: string; url: string; seller: string | null; price: number | null; currency: string | null; country: string | null; snippet?: string | null; thumbnail?: string | null };

// What the model understood of the protected object, and how it plans to look for copies.
type Plan = {
  produit: string;
  description_visuelle: string;
  pages_officielles: string[];
  domaines_officiels: string[];
  titulaires_officiels: string[];
  prix_officiel: string;
  requetes: string[];
};

const MARKETPLACE_COUNTRY: [RegExp, string][] = [
  [/aliexpress|alibaba|dhgate|temu|joom|made-in-china/, "CN"],
  [/shein/, "CN"],
  [/wish\.com/, "US"],
];
const TLD_COUNTRY: Record<string, string> = {
  fr: "FR", de: "DE", it: "IT", es: "ES", uk: "GB", cn: "CN", hk: "HK", jp: "JP", kr: "KR", ru: "RU", be: "BE", ch: "CH", nl: "NL", pl: "PL",
  pt: "PT", ca: "CA", au: "AU", br: "BR", in: "IN", tr: "TR", ae: "AE", sg: "SG", us: "US", mx: "MX", at: "AT", se: "SE", dk: "DK", ie: "IE", tw: "TW", th: "TH", vn: "VN",
};
// Registrars that brands use to hold defensive registrations of their own name.
const BRAND_REGISTRARS = /markmonitor|csc corporate|corporation service company|com laude|safenames|nameshield|lexsynergy|brandsight|ebrand|instra/i;
const MAX_QUERIES = 6;
const MAX_CAPTURES = 6;

function hostOf(url: string): string {
  try {
    return new URL(url).hostname.toLowerCase();
  } catch {
    return "";
  }
}

function countryOf(url: string): string | null {
  const host = hostOf(url);
  for (const [re, c] of MARKETPLACE_COUNTRY) if (re.test(host)) return c;
  const tld = host.split(".").pop() ?? "";
  return TLD_COUNTRY[tld] ?? null;
}

// First-pass score, refined by the model when available.
function heuristic(c: Candidate, brand: string, markName: string): number {
  const text = `${c.title} ${c.url} ${c.snippet ?? ""}`.toLowerCase();
  let s = 30;
  if (/(replica|r[ée]plique|dupe|copie|imitation|inspir|style |type |look|fake|faux)/.test(text)) s += 30;
  if (text.includes(markName.toLowerCase())) s += 10;
  if (text.includes(brand.toLowerCase())) s += 5;
  if (MARKETPLACE_COUNTRY.some(([re]) => re.test(hostOf(c.url)))) s += 15;
  if (c.price !== null && c.price < 150) s += 10;
  return Math.max(5, Math.min(95, s));
}

function searchTerm(client: Client, mark: Mark): string {
  const brand = client.holderNames[0] ?? client.name.toUpperCase();
  return mark.name.toUpperCase().includes(brand) ? mark.name : `${client.name} ${mark.name}`;
}

async function download(url: string): Promise<{ bytes: ArrayBuffer; type: string } | null> {
  try {
    const res = await fetch(url, { headers: { "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36" }, signal: AbortSignal.timeout(12_000) });
    const type = res.headers.get("content-type") ?? "";
    if (!res.ok || !/^image\/(png|jpe?g|webp|gif)/.test(type)) return null;
    const bytes = await res.arrayBuffer();
    return bytes.byteLength > 0 && bytes.byteLength < 4_500_000 ? { bytes, type: type.split(";")[0] } : null;
  } catch {
    return null;
  }
}

const dataUri = (img: { bytes: ArrayBuffer; type: string }) => `data:${img.type};base64,${Buffer.from(img.bytes).toString("base64")}`;

export async function startWatch(client: Client, mark: Mark, trigger: WatchRun["trigger"]): Promise<WatchRun> {
  const run = await store.createRun(client.id, mark.id, mark.name, trigger);
  const trace = await startTrace("recherche", `Recherche de contrefaçons — ${mark.name}`, client.id);
  // Runs in the background; the dashboard polls the run for results.
  void runInTrace(trace, () => execute(run, client, mark, trace.id))
    .then(() => trace.finish("done", { runId: run.id }))
    .catch(async (e) => {
      await store.updateRun(run.id, { status: "error", error: e instanceof Error ? e.message : String(e), finished_at: new Date().toISOString() });
      await trace.finish("error", { error: e instanceof Error ? e.message : String(e) });
    });
  return run;
}

// Understands what the sign protects in practice (the product, its look, its price) before searching.
async function makePlan(client: Client, mark: Mark, portfolio: Mark[], term: string): Promise<Plan> {
  const brand = client.holderNames[0] ?? client.name.toUpperCase();
  const fallback: Plan = {
    produit: term,
    description_visuelle: "",
    pages_officielles: [],
    domaines_officiels: [`${slugify(brand).replace(/-/g, "")}.com`, `${slugify(brand).replace(/-/g, "")}.fr`],
    titulaires_officiels: client.holderNames.length ? client.holderNames : [client.name],
    prix_officiel: "",
    requetes: [`${term} réplique`, `${term} dupe`, `${term} replica`],
  };
  if (!llmConfigured()) return fallback;
  const web = await webSearch(`${term} : produit, collection, page officielle et prix`, "watch-plan-web", { maxResults: 6 }).catch(() => ({ summary: "", results: [] }));
  const siblings = portfolio.filter((m) => m.name.toUpperCase() === mark.name.toUpperCase()).map((m) => `${m.id} ${m.type} cl. ${m.classes.join(",")}`);
  const plan = await extractJson<Partial<Plan>>(
    [
      "Tu prépares une recherche de contrefaçons en ligne pour un avocat en propriété intellectuelle.",
      "À partir de la marque et des résultats web, identifie le produit réellement commercialisé sous ce signe par le titulaire, décris son apparence de façon à reconnaître une copie, et prépare des requêtes de recherche.",
      "Les requêtes doivent viser les copies et imitations, pas le site officiel : « dupe », « inspired », « style », « réplique », « pas cher », places de marché (AliExpress, DHgate, Temu, Etsy, Amazon, eBay, Vinted), en français et en anglais, et au moins une requête décrivant l'objet sans la marque.",
      `Donne au plus ${MAX_QUERIES} requêtes. N'utilise pour pages_officielles que des URL présentes dans les résultats web.`,
      'Réponds en JSON : {"produit":"","description_visuelle":"","pages_officielles":[],"domaines_officiels":[],"titulaires_officiels":[],"prix_officiel":"","requetes":[]}.',
    ].join("\n"),
    JSON.stringify({
      titulaire: client.name,
      noms_du_titulaire: client.holderNames,
      marque: mark.name,
      titre: `${mark.id} · ${mark.type} · ${mark.office} · classes ${mark.classes.join(", ") || "n.c."}`,
      autres_titres_meme_nom: siblings,
      resultats_web: web.results.map((r) => ({ url: r.url, titre: r.title, extrait: r.snippet.slice(0, 300) })),
    }),
    "watch-plan",
    "strong",
  );
  const urls = new Set(web.results.map((r) => r.url));
  return {
    produit: plan?.produit?.trim() || fallback.produit,
    description_visuelle: plan?.description_visuelle?.trim() ?? "",
    pages_officielles: (plan?.pages_officielles ?? []).filter((u) => urls.has(u)).slice(0, 3),
    domaines_officiels: Array.from(new Set([...fallback.domaines_officiels, ...(plan?.domaines_officiels ?? []).map((d) => d.toLowerCase().replace(/^www\./, ""))])),
    titulaires_officiels: Array.from(new Set([...fallback.titulaires_officiels, ...(plan?.titulaires_officiels ?? [])])),
    prix_officiel: plan?.prix_officiel ?? "",
    requetes: (plan?.requetes ?? []).filter((q) => typeof q === "string" && q.trim()).slice(0, MAX_QUERIES).concat(plan?.requetes?.length ? [] : fallback.requetes),
  };
}

async function execute(run: WatchRun, client: Client, mark: Mark, traceId: string) {
  const stats: RunStats = { steps: [], total: 0, high: 0, byCountry: {}, traceId };
  const findings: Finding[] = [];
  const extra = new Map<number, { snippet: string | null; thumbnail: string | null }>();
  const seen = new Set<string>();
  const brand = client.holderNames[0] ?? client.name.toUpperCase();
  const term = searchTerm(client, mark);
  let plan: Plan | null = null;
  let official: string[] = [`${slugify(brand).replace(/-/g, "")}.com`, `${slugify(brand).replace(/-/g, "")}.fr`];
  const references: { bytes: ArrayBuffer; type: string; url: string | null }[] = [];
  let searchApiOut = !searchapi.searchApiConfigured();

  const save = () => store.updateRun(run.id, { stats: { ...stats, total: findings.length, high: findings.filter((f) => (f.confidence ?? 0) >= 70).length, byCountry: countByCountry(findings) } });

  const step = async (name: string, fn: () => Promise<string | void>) => {
    const s: RunStats["steps"][number] = { name, status: "running" };
    stats.steps.push(s);
    await save();
    try {
      const detail = await fn();
      s.status = "done";
      if (detail) s.detail = detail;
    } catch (e) {
      s.status = "error";
      s.detail = e instanceof Error ? e.message.slice(0, 160) : String(e);
    }
    traceStep({ kind: "etape", name, detail: s.detail, ok: s.status === "done" });
    await save();
  };

  const isOfficial = (url: string) => {
    const host = hostOf(url).replace(/^www\./, "");
    return official.some((o) => host === o || host.endsWith(`.${o}`));
  };

  const add = async (cands: Candidate[]): Promise<string> => {
    let skipped = 0;
    const fresh = cands.filter((c) => {
      const host = hostOf(c.url);
      if (!host || seen.has(c.url)) return false;
      seen.add(c.url);
      if (isOfficial(c.url)) {
        skipped++;
        return false;
      }
      return true;
    });
    const created = await store.addFindings(
      fresh.map((c) => ({
        run_id: run.id, client_id: client.id, mark_id: mark.id, kind: c.kind, source: c.source, platform: c.seller ?? hostOf(c.url),
        url: c.url, title: c.title, price: c.price, currency: c.currency, country: c.country ?? countryOf(c.url), seller: c.seller,
        company: null, confidence: heuristic(c, brand, mark.name), reason: null,
      })),
    );
    created.forEach((f, i) => extra.set(f.id, { snippet: fresh[i]?.snippet ?? null, thumbnail: fresh[i]?.thumbnail ?? null }));
    findings.push(...created);
    return `${created.length} nouveau(x) résultat(s)${skipped ? `, ${skipped} site(s) officiel(s) écarté(s)` : ""}`;
  };

  const listing = (kind: FindingKind, source: string) => (items: searchapi.Listing[]): Candidate[] =>
    items.slice(0, 20).map((i) => ({ kind, source, title: i.title, url: i.url, seller: i.seller, price: i.price, currency: i.currency, country: null, thumbnail: i.thumbnail }));

  // SearchAPI calls stop at the first "quota exhausted" answer instead of failing one by one.
  const viaSearchApi = async (name: string, fn: () => Promise<Candidate[]>) => {
    if (searchApiOut) return;
    await step(name, async () => {
      try {
        return await add(await fn());
      } catch (e) {
        if (e instanceof Error && e.message === searchapi.QUOTA_MESSAGE) searchApiOut = true;
        throw e;
      }
    });
  };

  // 1. What is protected, concretely.
  await step("Analyse de l'objet protégé", async () => {
    const portfolio = await store.listMarks(client.id);
    plan = await makePlan(client, mark, portfolio, term);
    official = plan.domaines_officiels;
    traceStep({
      kind: "etape",
      name: "Plan de recherche",
      output: plan,
      ok: true,
      summary: `${plan.produit}${plan.description_visuelle ? ` — ${plan.description_visuelle}` : ""}. Requêtes : ${plan.requetes.map((q) => `« ${q} »`).join(", ")}.`,
      links: plan.pages_officielles.map((u) => ({ url: u, title: "page officielle" })),
    });
    return `${plan.produit}${plan.prix_officiel ? ` · prix officiel ${plan.prix_officiel}` : ""} · ${plan.requetes.length} requête(s) préparée(s)`;
  });
  const p = plan as Plan | null;

  // 2. Reference photos of the genuine product: the one set on the title, else the official pages.
  await step("Photos de référence", async () => {
    const urls: string[] = mark.imageUrl ? [mark.imageUrl] : [];
    for (const page of p?.pages_officielles ?? []) {
      if (urls.length >= 3) break;
      const found = await traced("source", "page.images", { page }, () => pageImages(page, `${mark.name} ${p?.produit ?? ""}`, 3)).catch(() => [] as string[]);
      urls.push(...found.filter((u) => !urls.includes(u)));
    }
    for (const url of urls.slice(0, 3)) {
      const img = await download(url);
      if (img) references.push({ ...img, url });
    }
    if (references.length === 0) return "aucune photo trouvée : ajoutez-en une dans la fiche de la marque";
    traceStep({ kind: "api", name: "Photos retenues", ok: true, links: references.map((r) => ({ url: r.url!, title: hostOf(r.url!) })) });
    return `${references.length} photo(s) du produit authentique`;
  });

  // 3. Visual search from those photos.
  if (vision.visionConfigured() && references.length) {
    await step("Recherche visuelle (Google Vision)", async () => {
      const cands: Candidate[] = [];
      for (const ref of references.slice(0, 2)) {
        const r = await traced("source", "vision.web", { image: ref.url }, () => vision.webDetection({ bytes: ref.bytes }), (x) => ({ pages: x.pages.length, devine: x.guess }));
        cands.push(...r.pages.map((m) => ({ kind: "image" as const, source: `Google Vision (${m.match})`, title: m.title, url: m.url, seller: null, price: null, currency: null, country: null, thumbnail: m.image })));
      }
      return add(cands);
    });
  }
  const lensRef = references.find((r) => r.url);
  if (lensRef) await viaSearchApi("Google Lens", async () => listing("image", "Google Lens")(await traced("source", "searchapi.lens", { image: lensRef.url }, () => searchapi.lens(lensRef.url!))));

  // 4. Marketplaces and the open web, with the queries the model prepared.
  const queries = p?.requetes.length ? p.requetes : [term];
  await viaSearchApi("Google Shopping", async () => listing("annonce", "Google Shopping")(await traced("source", "searchapi.shopping", { q: queries[0] }, () => searchapi.shopping(queries[0]))));
  await viaSearchApi("Amazon", async () => listing("annonce", "Amazon")(await traced("source", "searchapi.amazon", { q: term }, () => searchapi.amazon(term))));
  await viaSearchApi("eBay", async () => listing("annonce", "eBay")(await traced("source", "searchapi.ebay", { q: term }, () => searchapi.ebay(term))));
  if (searchApiOut && searchapi.searchApiConfigured()) traceStep({ kind: "etape", name: "SearchAPI", detail: `${searchapi.QUOTA_MESSAGE} : recherche web à la place`, ok: false });

  if (llmConfigured()) {
    await step(`Recherche web (${queries.length} requêtes)`, async () => {
      const system = "Trouve des pages qui vendent ce produit ou une imitation : annonces, boutiques en ligne, places de marché. Écarte le site officiel et la presse. Réponds en une phrase.";
      const answers = await Promise.all(queries.map((q, i) => webSearch(q, `watch-web-${i + 1}`, { system, maxResults: 8 }).catch(() => ({ summary: "", results: [] }))));
      const cands = answers.flatMap((a) => a.results).map((r) => ({ kind: "web" as const, source: "Recherche web", title: r.title, url: r.url, seller: null, price: null, currency: null, country: null, snippet: r.snippet }));
      return add(cands);
    });
  }

  // 5. Lookalike domains, minus the brand's own defensive registrations.
  if (urlscan.urlscanConfigured()) {
    await step("Domaines similaires", async () => {
      const domains = await traced("source", "urlscan.search", { brand }, () => urlscan.lookalikeDomains(slugify(brand).replace(/-/g, ""), official));
      const owners = (p?.titulaires_officiels ?? client.holderNames).map((h) => h.toLowerCase());
      const checked = await Promise.all(
        domains.map(async (d) => {
          const r = await lookupDomain(registrableDomain(d.domain)).catch(() => null);
          const own = Boolean(r && ((r.registrar && BRAND_REGISTRARS.test(r.registrar)) || (r.holder && owners.some((o) => r.holder!.toLowerCase().includes(o)))));
          return { d, own, registrar: r?.registrar ?? null };
        }),
      );
      const own = checked.filter((c) => c.own);
      traceStep({ kind: "etape", name: "Domaines du titulaire écartés", detail: `${own.length} sur ${domains.length}`, output: own.map((c) => `${c.d.domain} (${c.registrar})`), ok: true });
      const kept = await add(checked.filter((c) => !c.own).map(({ d }) => ({ kind: "domaine" as const, source: "urlscan", title: d.domain, url: d.url, seller: null, price: null, currency: null, country: d.country })));
      return `${kept} · ${own.length} domaine(s) détenu(s) par le titulaire écarté(s)`;
    });
  }

  // 6. The model sorts copies from the genuine, the press and the unrelated.
  if (llmConfigured() && findings.length) await step("Évaluation", () => score(findings, extra, client, mark, p));

  // 7. Dated captures of the strongest results, then a visual comparison with the genuine product.
  if (urlscan.urlscanConfigured()) {
    await step("Captures et comparaison visuelle", async () => {
      const targets = [...findings].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)).filter((f) => (f.confidence ?? 0) >= 40).slice(0, MAX_CAPTURES);
      let done = 0;
      let compared = 0;
      for (const f of targets) {
        try {
          const cap = await traced("source", "urlscan.capture", { url: f.url }, () => urlscan.capture(f.url), (c) => (c ? { uuid: c.uuid, country: c.country } : null));
          if (!cap) continue;
          let path: string | null = null;
          let sha: string | null = null;
          if (cap.screenshot) {
            sha = createHash("sha256").update(Buffer.from(cap.screenshot)).digest("hex");
            path = `${client.id}/${mark.id}/${run.id}/${f.id}.png`;
            // The urlscan link and the hash stay even if storing the image fails.
            await uploadObject("evidence", path, cap.screenshot, "image/png").catch(() => (path = null));
          }
          const patch = { screenshot_path: path, sha256: sha, scan_url: cap.resultUrl, captured_at: cap.capturedAt, country: f.country ?? cap.country };
          await store.updateFinding(f.id, patch);
          Object.assign(f, { screenshotPath: path, sha256: sha, scanUrl: cap.resultUrl, capturedAt: cap.capturedAt, country: patch.country });
          done++;
          if (cap.screenshot && llmConfigured() && (await compare(f, { bytes: cap.screenshot, type: "image/png" }, references, mark, p))) compared++;
          await save();
        } catch {
          // A failed capture must not stop the run.
        }
      }
      return `${done}/${targets.length} capture(s), ${compared} comparée(s) au produit authentique`;
    });
  }

  await step("Titulaires des sites", async () => {
    const targets = [...findings].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)).slice(0, 8);
    let found = 0;
    for (const f of targets) {
      const domain = registrableDomain(hostOf(f.url));
      if (!domain || /amazon|ebay|aliexpress|google|etsy|temu|alibaba|dhgate|vinted|shein/.test(domain)) continue;
      try {
        const r = await lookupDomain(domain);
        const company = { name: r.holder ?? `Titulaire masqué (bureau d'enregistrement : ${r.registrar ?? "inconnu"})`, source: `RDAP ${domain}` };
        await store.updateFinding(f.id, { company });
        f.company = company;
        if (r.holder) found++;
      } catch {
        // RDAP is not available for every extension.
      }
    }
    return `${found} titulaire(s) identifié(s)`;
  });

  const high = findings.filter((f) => (f.confidence ?? 0) >= 70);
  const countries = Object.entries(countByCountry(high)).sort((a, b) => b[1] - a[1]).map(([c, n]) => `${c} ${n}`).join(", ");
  await store.updateRun(run.id, { status: "done", finished_at: new Date().toISOString(), stats: { ...stats, total: findings.length, high: high.length, byCountry: countByCountry(findings) } });
  await store.addEvent(
    client.id,
    "recherche",
    `Recherche de contrefaçons — ${mark.name}`,
    `${findings.length} résultat(s), dont ${high.length} à forte confiance${countries ? ` (${countries})` : ""}.`,
    { runId: run.id, markId: mark.id, trigger: run.trigger, traceId },
  );
}

// Looks at the captured page next to the genuine product and adjusts the confidence.
async function compare(f: Finding, shot: { bytes: ArrayBuffer; type: string }, refs: { bytes: ArrayBuffer; type: string }[], mark: Mark, plan: Plan | null): Promise<boolean> {
  const parts: ContentPart[] = [
    { type: "text", text: `Objet protégé : ${plan?.produit ?? mark.name}. ${plan?.description_visuelle ?? ""}\nPage examinée : ${f.url}\n${refs.length ? `Les ${refs.length} première(s) image(s) montrent le produit authentique ; la dernière est la capture de la page.` : "L'image est la capture de la page ; compare-la à la description."}` },
    ...refs.slice(0, 2).map((r) => ({ type: "image_url" as const, image_url: { url: dataUri(r) } })),
    { type: "image_url", image_url: { url: dataUri(shot) } },
  ];
  const out = await extractJson<{ ressemblance?: number; signe_repris?: boolean; categorie?: string; motif?: string }>(
    [
      "Tu examines une page web pour un avocat en propriété intellectuelle.",
      "Indique à quel point le produit montré reprend l'apparence du produit authentique (0-100), si le signe ou la marque est repris, et la nature de la page : copie, revente, officiel, presse, sans_rapport, inaccessible.",
      'Réponds en JSON : {"ressemblance":0,"signe_repris":false,"categorie":"","motif":"15 mots maximum"}.',
    ].join("\n"),
    parts,
    "watch-vision",
    "strong",
  );
  if (!out || typeof out.ressemblance !== "number") return false;
  const visual = Math.max(0, Math.min(100, Math.round(out.ressemblance)));
  const dismissed = ["officiel", "presse", "sans_rapport", "inaccessible"].includes(out.categorie ?? "");
  const confidence = dismissed ? Math.min(f.confidence ?? 0, 15) : Math.round(0.4 * (f.confidence ?? 0) + 0.6 * Math.max(visual, out.signe_repris ? 60 : 0));
  const reason = `${f.reason ? `${f.reason} · ` : ""}vue : ${out.motif ?? ""}`.slice(0, 220);
  await store.updateFinding(f.id, { confidence, reason });
  f.confidence = confidence;
  f.reason = reason;
  return true;
}

function countByCountry(findings: Finding[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const f of findings) if (f.country) out[f.country] = (out[f.country] ?? 0) + 1;
  return out;
}

const SCORE_BATCH = 50;
const MAX_SCORED = 200;

// Every result is judged by the model, in parallel batches.
async function score(findings: Finding[], extra: Map<number, { snippet: string | null; thumbnail: string | null }>, client: Client, mark: Mark, plan: Plan | null): Promise<string> {
  const chunks: Finding[][] = [];
  for (let i = 0; i < Math.min(findings.length, MAX_SCORED); i += SCORE_BATCH) chunks.push(findings.slice(i, i + SCORE_BATCH));
  const results = await Promise.all(chunks.map((chunk, k) => scoreBatch(chunk, extra, client, mark, plan, `watch-score-${k + 1}`)));
  const n = results.reduce((a, r) => a + r.n, 0);
  const dismissed = results.reduce((a, r) => a + r.dismissed, 0);
  return `${n} résultat(s) évalué(s), ${dismissed} écarté(s) (officiel, presse, sans rapport)`;
}

async function scoreBatch(findings: Finding[], extra: Map<number, { snippet: string | null; thumbnail: string | null }>, client: Client, mark: Mark, plan: Plan | null, label: string): Promise<{ n: number; dismissed: number }> {
  const batch = findings.map((f, i) => ({ i, titre: f.title, site: hostOf(f.url), url: f.url.slice(0, 160), extrait: extra.get(f.id)?.snippet?.slice(0, 240) ?? null, prix: f.price, devise: f.currency, vendeur: f.seller, source: f.source }));
  const out = await extractJson<{ scores?: { i: number; confiance: number; categorie?: string; motif: string }[] }>(
    [
      `Tu assistes un avocat en propriété intellectuelle. Marque surveillée : « ${mark.name} » (${mark.type}, classes ${mark.classes.join(", ") || "n.c."}), titulaire ${client.name}.`,
      plan ? `Produit protégé : ${plan.produit}. Apparence : ${plan.description_visuelle || "n.c."}. Prix officiel : ${plan.prix_officiel || "n.c."}. Sites officiels : ${plan.domaines_officiels.join(", ")}.` : "",
      "Pour chaque résultat, donne une catégorie et la probabilité (0-100) qu'il s'agisse d'une atteinte à vérifier.",
      "Catégories : copie (imitation, « dupe », « inspired », réplique), usage_du_signe (le signe utilisé par un tiers pour vendre), revente (occasion d'un produit authentique, épuisement des droits), officiel (titulaire ou revendeur agréé), presse, sans_rapport.",
      "Élevée : copie explicite, prix très inférieur au prix officiel, place de marché ou boutique inconnue. Mets 10 au plus pour officiel, presse et sans_rapport.",
      "Un produit d'une autre catégorie que le produit protégé (par exemple un parfum du même nom quand il s'agit d'un bijou) est sans_rapport.",
      'Réponds en JSON : {"scores":[{"i":0,"confiance":80,"categorie":"copie","motif":"12 mots maximum"}]}.',
    ].filter(Boolean).join("\n"),
    JSON.stringify(batch),
    label,
    "strong",
  );
  let n = 0;
  let dismissed = 0;
  for (const s of out?.scores ?? []) {
    const f = findings[s.i];
    if (!f || typeof s.confiance !== "number") continue;
    const low = ["officiel", "presse", "sans_rapport"].includes(s.categorie ?? "");
    const confidence = Math.max(0, Math.min(low ? 10 : 100, Math.round(s.confiance)));
    if (low) dismissed++;
    const reason = `${s.categorie ? `${s.categorie.replace(/_/g, " ")} : ` : ""}${String(s.motif ?? "")}`.slice(0, 160);
    await store.updateFinding(f.id, { confidence, reason });
    f.confidence = confidence;
    f.reason = reason;
    n++;
  }
  return { n, dismissed };
}
