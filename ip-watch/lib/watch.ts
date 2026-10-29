import { createHash } from "crypto";
import { extractJson, llmConfigured, webSearch } from "./llm";
import { traced } from "./log";
import { slugify } from "./onboard";
import * as searchapi from "./sources/searchapi";
import * as urlscan from "./sources/urlscan";
import { lookupDomain, registrableDomain } from "./sources/rdap";
import * as store from "./store";
import { uploadObject } from "./supabase";
import type { Client, Finding, FindingKind, Mark, RunStats, WatchRun } from "./types";

type Candidate = { kind: FindingKind; source: string; title: string; url: string; seller: string | null; price: number | null; currency: string | null; country: string | null };

const MARKETPLACE_COUNTRY: [RegExp, string][] = [
  [/aliexpress|alibaba|dhgate|temu|joom|made-in-china/, "CN"],
  [/shein/, "CN"],
  [/wish\.com/, "US"],
];
const TLD_COUNTRY: Record<string, string> = {
  fr: "FR", de: "DE", it: "IT", es: "ES", uk: "GB", cn: "CN", hk: "HK", jp: "JP", kr: "KR", ru: "RU", be: "BE", ch: "CH", nl: "NL", pl: "PL",
  pt: "PT", ca: "CA", au: "AU", br: "BR", in: "IN", tr: "TR", ae: "AE", sg: "SG", us: "US", mx: "MX", at: "AT", se: "SE", dk: "DK", ie: "IE", tw: "TW", th: "TH", vn: "VN",
};

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
  const text = `${c.title} ${c.url}`.toLowerCase();
  let s = 35;
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

export async function startWatch(client: Client, mark: Mark, trigger: WatchRun["trigger"]): Promise<WatchRun> {
  const run = await store.createRun(client.id, mark.id, mark.name, trigger);
  // Runs in the background; the dashboard polls the run for results.
  void execute(run, client, mark).catch(async (e) => {
    await store.updateRun(run.id, { status: "error", error: e instanceof Error ? e.message : String(e), finished_at: new Date().toISOString() });
  });
  return run;
}

async function execute(run: WatchRun, client: Client, mark: Mark) {
  const stats: RunStats = { steps: [], total: 0, high: 0, byCountry: {} };
  const findings: Finding[] = [];
  const seen = new Set<string>();
  const brand = client.holderNames[0] ?? client.name.toUpperCase();
  const official = [`${slugify(brand).replace(/-/g, "")}.com`, `${slugify(brand).replace(/-/g, "")}.fr`];
  const term = searchTerm(client, mark);

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
    await save();
  };

  const add = async (cands: Candidate[]): Promise<string> => {
    const fresh = cands.filter((c) => {
      const host = hostOf(c.url);
      if (!host || seen.has(c.url) || official.some((o) => host === o || host.endsWith(`.${o}`))) return false;
      seen.add(c.url);
      return true;
    });
    const created = await store.addFindings(
      fresh.map((c) => ({
        run_id: run.id, client_id: client.id, mark_id: mark.id, kind: c.kind, source: c.source, platform: c.seller ?? hostOf(c.url),
        url: c.url, title: c.title, price: c.price, currency: c.currency, country: c.country ?? countryOf(c.url), seller: c.seller,
        company: null, confidence: heuristic(c, brand, mark.name), reason: null,
      })),
    );
    findings.push(...created);
    return `${created.length} résultat(s)`;
  };

  const listing = (kind: FindingKind, source: string) => (items: searchapi.Listing[]): Candidate[] =>
    items.slice(0, 20).map((i) => ({ kind, source, title: i.title, url: i.url, seller: i.seller, price: i.price, currency: i.currency, country: null }));

  if (searchapi.searchApiConfigured()) {
    await step("Google Shopping", async () => add(listing("annonce", "Google Shopping")(await traced("source", "searchapi.shopping", { term }, () => searchapi.shopping(term)))));
    await step("Amazon", async () => add(listing("annonce", "Amazon")(await traced("source", "searchapi.amazon", { term }, () => searchapi.amazon(term)))));
    await step("eBay", async () => add(listing("annonce", "eBay")(await traced("source", "searchapi.ebay", { term }, () => searchapi.ebay(term)))));
    if (mark.imageUrl) {
      await step("Google Lens", async () => add(listing("image", "Google Lens")(await traced("source", "searchapi.lens", { image: mark.imageUrl }, () => searchapi.lens(mark.imageUrl!)))));
    }
    await step("Google (répliques)", async () =>
      add(listing("web", "Google")(await traced("source", "searchapi.google", { term }, () => searchapi.google(`"${term}" (réplique OR replica OR dupe OR imitation OR "pas cher")`)))),
    );
  } else if (llmConfigured()) {
    await step("Recherche web (repli)", async () => {
      const { results } = await webSearch(`Annonces en ligne vendant des copies, répliques ou imitations de « ${term} » (marketplaces, boutiques).`, "watch-web");
      return add(results.map((r) => ({ kind: "web" as const, source: "Recherche web", title: r.title, url: r.url, seller: null, price: null, currency: null, country: null })));
    });
  }

  if (urlscan.urlscanConfigured()) {
    await step("Domaines similaires", async () => {
      const domains = await traced("source", "urlscan.search", { brand }, () => urlscan.lookalikeDomains(slugify(brand).replace(/-/g, ""), official));
      return add(domains.map((d) => ({ kind: "domaine" as const, source: "urlscan", title: d.domain, url: d.url, seller: null, price: null, currency: null, country: d.country })));
    });
  }

  if (llmConfigured() && findings.length) await step("Évaluation", () => score(findings, client, mark));

  if (urlscan.urlscanConfigured()) {
    await step("Captures", async () => {
      const targets = [...findings].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)).filter((f) => (f.confidence ?? 0) >= 50).slice(0, 6);
      let done = 0;
      for (const f of targets) {
        try {
          const cap = await traced("source", "urlscan.capture", { url: f.url }, () => urlscan.capture(f.url), (c) => (c ? { uuid: c.uuid, country: c.country } : null));
          if (!cap) continue;
          let path: string | null = null;
          let sha: string | null = null;
          if (cap.screenshot) {
            sha = createHash("sha256").update(Buffer.from(cap.screenshot)).digest("hex");
            path = `${client.id}/${mark.id}/${run.id}/${f.id}.png`;
            await uploadObject("evidence", path, cap.screenshot, "image/png");
          }
          const patch = { screenshot_path: path, sha256: sha, scan_url: cap.resultUrl, captured_at: cap.capturedAt, country: f.country ?? cap.country };
          await store.updateFinding(f.id, patch);
          Object.assign(f, { screenshotPath: path, sha256: sha, scanUrl: cap.resultUrl, capturedAt: cap.capturedAt, country: patch.country });
          done++;
          await save();
        } catch {
          // A failed capture must not stop the run.
        }
      }
      return `${done}/${targets.length} capture(s)`;
    });
  }

  await step("Titulaires des sites", async () => {
    const targets = [...findings].sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)).slice(0, 8);
    let found = 0;
    for (const f of targets) {
      const domain = registrableDomain(hostOf(f.url));
      if (!domain || /amazon|ebay|aliexpress|google|etsy|temu|alibaba/.test(domain)) continue;
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
    { runId: run.id, markId: mark.id, trigger: run.trigger },
  );
}

function countByCountry(findings: Finding[]): Record<string, number> {
  const out: Record<string, number> = {};
  for (const f of findings) if (f.country) out[f.country] = (out[f.country] ?? 0) + 1;
  return out;
}

async function score(findings: Finding[], client: Client, mark: Mark): Promise<string> {
  const batch = findings.slice(0, 60).map((f, i) => ({ i, titre: f.title, site: hostOf(f.url), prix: f.price, devise: f.currency, vendeur: f.seller, type: f.kind }));
  const out = await extractJson<{ scores?: { i: number; confiance: number; motif: string }[] }>(
    [
      `Tu assistes un avocat en propriété intellectuelle. Marque surveillée : « ${mark.name} » (${mark.type}, classes ${mark.classes.join(", ") || "n.c."}), titulaire ${client.name}.`,
      "Pour chaque résultat, estime la probabilité (0-100) qu'il s'agisse d'une atteinte à vérifier (contrefaçon, imitation, usage non autorisé du signe).",
      "Élevée : copie ou « dupe » explicite, usage du signe par un tiers non autorisé, prix anormalement bas sur une place de marché.",
      "Faible : site officiel ou revendeur agréé, article de presse, revente d'occasion d'un produit authentique (épuisement des droits), homonyme sans rapport.",
      'Réponds en JSON : {"scores":[{"i":0,"confiance":80,"motif":"12 mots maximum"}]}.',
    ].join("\n"),
    JSON.stringify(batch),
    "watch-score",
    "strong",
  );
  let n = 0;
  for (const s of out?.scores ?? []) {
    const f = findings[s.i];
    if (!f || typeof s.confiance !== "number") continue;
    const confidence = Math.max(0, Math.min(100, Math.round(s.confiance)));
    await store.updateFinding(f.id, { confidence, reason: String(s.motif ?? "").slice(0, 160) });
    f.confidence = confidence;
    f.reason = s.motif;
    n++;
  }
  return `${n} résultat(s) évalué(s)`;
}
