import type { Office, Right, RightType, Territory } from "../types";

// INPI "API diffusion". Login needs a cookie jar and an anti-CSRF token that changes on each response.
const BASE = "https://api-gateway.inpi.fr";
const SESSION_TTL_MS = 4 * 60 * 1000;

export class InpiSession {
  private jar = new Map<string, string>();
  readonly createdAt = Date.now();

  private remember(res: Response) {
    for (const line of res.headers.getSetCookie()) {
      const [pair] = line.split(";");
      const i = pair.indexOf("=");
      if (i > 0) this.jar.set(pair.slice(0, i).trim(), pair.slice(i + 1).trim());
    }
  }

  private headers(json: boolean): Record<string, string> {
    const h: Record<string, string> = { Accept: "application/json" };
    if (json) h["Content-Type"] = "application/json";
    if (this.jar.size) h.Cookie = Array.from(this.jar.entries()).map(([k, v]) => `${k}=${v}`).join("; ");
    const xsrf = this.jar.get("XSRF-TOKEN");
    if (xsrf) h["X-XSRF-TOKEN"] = decodeURIComponent(xsrf);
    return h;
  }

  async request(method: "GET" | "POST", path: string, body?: unknown): Promise<{ status: number; data: unknown }> {
    const res = await fetch(`${BASE}${path}`, { method, headers: this.headers(body !== undefined), body: body === undefined ? undefined : JSON.stringify(body), redirect: "manual" });
    this.remember(res);
    const text = await res.text();
    let data: unknown = text;
    try {
      data = text ? JSON.parse(text) : null;
    } catch {
      // Keep non-JSON bodies (HTML error pages) as text.
    }
    return { status: res.status, data };
  }

  // Returns the login HTTP status; the body is discarded because it may contain tokens.
  async login(username: string, password: string): Promise<number> {
    await this.request("GET", "/services/uaa/api/authenticate");
    const { status } = await this.request("POST", "/auth/login", { username, password, rememberMe: true });
    return status;
  }
}

let shared: InpiSession | null = null;

export function inpiConfigured(): boolean {
  return Boolean(process.env.INPI_USERNAME && process.env.INPI_PASSWORD);
}

export async function inpiSession(): Promise<InpiSession> {
  if (shared && Date.now() - shared.createdAt < SESSION_TTL_MS) return shared;
  if (!inpiConfigured()) throw new Error("INPI_USERNAME / INPI_PASSWORD manquants");
  const s = new InpiSession();
  const status = await s.login(process.env.INPI_USERNAME!, process.env.INPI_PASSWORD!.trim().replace(/^(["'])(.*)\1$/, "$2"));
  if (status >= 400) throw new Error(`Connexion INPI refusée (${status})`);
  shared = s;
  return s;
}

// The largest array of objects in a response is the list of hits, whatever the envelope is called.
export function hitsOf(data: unknown): Record<string, unknown>[] {
  let best: Record<string, unknown>[] = [];
  const visit = (node: unknown, depth: number) => {
    if (depth > 4 || !node || typeof node !== "object") return;
    if (Array.isArray(node)) {
      if (node.length > best.length && node.every((x) => x && typeof x === "object" && !Array.isArray(x))) best = node as Record<string, unknown>[];
      return;
    }
    for (const v of Object.values(node)) visit(v, depth + 1);
  };
  visit(data, 0);
  return best;
}

export async function search(kind: "marques" | "modeles", query: string, collections: string[], position = 0, size = 100) {
  const s = await inpiSession();
  const res = await s.request("POST", `/services/apidiffusion/api/${kind}/search`, { collections, query, fields: ["*"], position, size });
  if (res.status === 401 || res.status === 403) {
    shared = null;
    const retry = await (await inpiSession()).request("POST", `/services/apidiffusion/api/${kind}/search`, { collections, query, fields: ["*"], position, size });
    return { ...retry, hits: hitsOf(retry.data) };
  }
  return { ...res, hits: hitsOf(res.data) };
}

export async function metadata(): Promise<string | null> {
  try {
    const s = await inpiSession();
    const res = await s.request("GET", "/services/apidiffusion/api/marques/metadata");
    const flat = flatten(res.data);
    const date = pick(flat, ["lastupdate", "lastupdatedate", "updatedate", "datemaj", "date"]);
    return date ? String(date) : null;
  } catch {
    return null;
  }
}

// ---------- Mapping a raw hit to a portfolio title

// Flattens nested objects and {name, value} field lists into one lowercase-keyed map.
export function flatten(item: unknown): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  const isPairs = (a: unknown[]) => a.length > 0 && a.every((x) => x && typeof x === "object" && "name" in (x as object) && "value" in (x as object));
  const visit = (node: unknown, depth: number) => {
    if (depth > 5 || !node || typeof node !== "object") return;
    if (Array.isArray(node)) {
      if (isPairs(node)) for (const p of node as { name: string; value: unknown }[]) if (!(p.name.toLowerCase() in out)) out[p.name.toLowerCase()] = p.value;
      return;
    }
    for (const [k, v] of Object.entries(node as Record<string, unknown>)) {
      const key = k.toLowerCase().replace(/[^a-z0-9]/g, "");
      if (Array.isArray(v) && isPairs(v)) visit(v, depth + 1);
      else if (v && typeof v === "object" && !Array.isArray(v)) visit(v, depth + 1);
      if (!(key in out)) out[key] = v;
    }
  };
  visit(item, 0);
  return out;
}

function pick(flat: Record<string, unknown>, candidates: string[]): unknown {
  for (const c of candidates) {
    const v = flat[c];
    if (v !== undefined && v !== null && v !== "") return v;
  }
  return undefined;
}

function str(v: unknown): string | null {
  if (v === undefined || v === null) return null;
  if (Array.isArray(v)) return v.map(str).filter(Boolean).join(", ") || null;
  if (typeof v === "object") return str(Object.values(v as object)[0]);
  const s = String(v).trim();
  return s || null;
}

export function parseDate(v: unknown): string | null {
  if (v === undefined || v === null || v === "") return null;
  if (typeof v === "number") return new Date(v > 1e11 ? v : v * 1000).toISOString().slice(0, 10);
  const s = String(Array.isArray(v) ? v[0] : v).trim();
  let m = s.match(/^(\d{4})-?(\d{2})-?(\d{2})/);
  if (m) return `${m[1]}-${m[2]}-${m[3]}`;
  m = s.match(/^(\d{2})[/.-](\d{2})[/.-](\d{4})/);
  if (m) return `${m[3]}-${m[2]}-${m[1]}`;
  return null;
}

function parseClasses(v: unknown): number[] {
  const nums = new Set<number>();
  const visit = (x: unknown) => {
    if (x === null || x === undefined) return;
    if (Array.isArray(x)) return x.forEach(visit);
    if (typeof x === "object") return Object.values(x as object).forEach(visit);
    for (const n of String(x).match(/\d{1,2}/g) ?? []) {
      const k = Number(n);
      if (k >= 1 && k <= 45) nums.add(k);
    }
  };
  visit(v);
  return Array.from(nums).sort((a, b) => a - b);
}

function parseCountries(v: unknown): Territory[] {
  const out = new Map<string, Territory>();
  const visit = (x: unknown) => {
    if (x === null || x === undefined) return;
    if (Array.isArray(x)) return x.forEach(visit);
    if (typeof x === "object") {
      const o = x as Record<string, unknown>;
      const code = str(o.code ?? o.countrycode ?? o.countryCode ?? o.pays ?? o.country);
      if (code && /^[A-Z]{2}$/.test(code)) out.set(code, { code, status: str(o.status ?? o.decision ?? o.statut) ?? undefined });
      return;
    }
    for (const c of String(x).toUpperCase().match(/\b[A-Z]{2}\b/g) ?? []) out.set(c, { code: c });
  };
  visit(v);
  out.delete("WO");
  out.delete("EM");
  return Array.from(out.values());
}

function officeOf(flat: Record<string, unknown>, number: string | null, fallback: string | undefined): Office {
  const hint = `${str(pick(flat, ["ukey", "collection", "office", "origin", "source", "registrationofficecode"])) ?? ""} ${number ?? ""} ${fallback ?? ""}`.toUpperCase();
  if (/\b(EU|EM|EUIPO)/.test(hint) || /^EU/.test(number ?? "")) return "EU";
  if (/\b(WO|WIPO|OMPI)/.test(hint) || /^WO/.test(number ?? "")) return "WO";
  return "FR";
}

export function normalize(raw: Record<string, unknown>, type: RightType, collectionHint?: string): Right | null {
  const flat = flatten(raw);
  const number = str(pick(flat, ["applicationnumber", "numerodepot", "registrationnumber", "numeroenregistrement", "number", "numero", "st13", "id"]));
  if (!number) return null;
  const office = officeOf(flat, number, collectionHint);
  const cleanNumber = number.replace(/^(FR|EU|EM|WO)/i, "");
  const name =
    str(pick(flat, type === "marque" ? ["mark", "markverbalelementtext", "verbalelement", "marque", "markexp", "denomination", "title"] : ["title", "titre", "designtitle", "objet", "designation", "productindication", "name"])) ??
    (type === "marque" ? "(marque figurative)" : "(dessin ou modèle)");
  const countries = office === "WO" ? parseCountries(pick(flat, ["designatedcountries", "designatedcountrycode", "designatedcountry", "designations", "countries", "pays"])) : [];
  return {
    id: `${office}${cleanNumber}`,
    type,
    office,
    number: cleanNumber,
    name,
    owner: str(pick(flat, ["applicantname", "applicant", "deposant", "holdername", "holder", "titulaire", "applicants", "ownername"])) ?? "",
    classes: parseClasses(pick(flat, type === "marque" ? ["classnumber", "classes", "niceclass", "classdescriptiondetails", "nice"] : ["locarnoclass", "locarno", "classnumber", "classes"])),
    filingDate: parseDate(pick(flat, ["applicationdate", "filingdate", "datedepot", "depositdate"])),
    registrationDate: parseDate(pick(flat, ["registrationdate", "dateenregistrement", "publicationdate", "datepublication"])),
    expiryDate: parseDate(pick(flat, ["expirydate", "expirationdate", "dateexpiration", "renewaldate", "endofprotectiondate", "expiry"])),
    status: str(pick(flat, ["markcurrentstatuscode", "currentstatus", "status", "statut", "legalstatus", "etat"])) ?? "",
    territories: office === "EU" ? [{ code: "EU" }] : office === "FR" ? [{ code: "FR" }] : countries,
    imageUrl: str(pick(flat, ["markimageurl", "imageurl", "image", "thumbnail", "imagelink"])),
    sourceUrl: `https://data.inpi.fr/${type === "marque" ? "marques" : "dessins_modeles"}/${office}${cleanNumber}`,
  };
}
