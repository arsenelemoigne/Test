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

// Mark searches only return five fields by default; dates and classes have to be asked for.
export const MARK_FIELDS = ["ApplicationNumber", "Mark", "MarkCurrentStatusCode", "DEPOSANT", "ukey", "ApplicationDate", "RegistrationDate", "ExpiryDate", "ClassNumber", "PublicationDate"];

export async function search(kind: "marques" | "modeles", query: string, collections: string[], position = 0, size = 100, fields: string[] = ["*"]) {
  const run = async (f: string[]) => (await inpiSession()).request("POST", `/services/apidiffusion/api/${kind}/search`, { collections, query, fields: f, position, size });
  let res = await run(fields);
  if (res.status === 401 || res.status === 403) {
    shared = null;
    res = await run(fields);
  }
  // An unknown field name is rejected: fall back to the default field set.
  if (res.status >= 400 && fields[0] !== "*") res = await run(["*"]);
  return { ...res, hits: hitsOf(res.data) };
}

export type NoticeDetails = { filingDate: string | null; registrationDate: string | null; expiryDate: string | null; classes: number[]; countries: Territory[]; status: string | null };

// Reads dates, classes and designated countries from a full record, whether it comes back as XML or JSON.
export function parseNotice(body: unknown): NoticeDetails {
  const text = typeof body === "string" ? body : JSON.stringify(body ?? "");
  const one = (names: string[]) => {
    for (const n of names) {
      const xml = text.match(new RegExp(`<(?:\\w+:)?${n}[^>]*>\\s*([^<]+?)\\s*</`, "i"));
      if (xml) return xml[1];
      const json = text.match(new RegExp(`"${n}"\\s*:\\s*"([^"]+)"`, "i"));
      if (json) return json[1];
    }
    return null;
  };
  const all = (names: string[]) =>
    names.flatMap((n) => [
      ...Array.from(text.matchAll(new RegExp(`<(?:\\w+:)?${n}[^>]*>\\s*([^<]+?)\\s*</`, "gi"))).map((m) => m[1]),
      ...Array.from(text.matchAll(new RegExp(`"${n}"\\s*:\\s*"([^"]+)"`, "gi"))).map((m) => m[1]),
    ]);
  return {
    filingDate: parseDate(one(["ApplicationDate", "FilingDate"])),
    registrationDate: parseDate(one(["RegistrationDate", "PublicationDate"])),
    expiryDate: parseDate(one(["ExpiryDate", "ExpirationDate", "RenewalDate"])),
    classes: parseClasses(all(["ClassNumber"])),
    countries: parseCountries(all(["DesignatedCountryCode", "DesignatedCountry"])),
    status: one(["MarkCurrentStatusCode", "MarkCurrentStatus"]),
  };
}

export async function notice(path: string): Promise<{ status: number; data: unknown }> {
  const s = await inpiSession();
  return s.request("GET", path);
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
  const isPairs = (a: unknown[]) => a.length > 0 && a.every((x) => x && typeof x === "object" && "name" in (x as object) && ("value" in (x as object) || "values" in (x as object)));
  const visit = (node: unknown, depth: number) => {
    if (depth > 5 || !node || typeof node !== "object") return;
    if (Array.isArray(node)) {
      if (isPairs(node)) {
        for (const p of node as { name: string; value?: unknown; values?: unknown }[]) {
          const key = p.name.toLowerCase();
          if (!(key in out)) out[key] = p.value ?? p.values;
        }
      }
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

// ukey collections seen: FMARK (France), CTMARK (EU), TMINT (international), DMF / DMI (French / international designs).
function officeOf(raw: Record<string, unknown>, flat: Record<string, unknown>, fallback: string | undefined): Office {
  const href = String((raw.xml as { href?: string } | undefined)?.href ?? "");
  const prefix = href.match(/notice\/([A-Z]+)/)?.[1] ?? "";
  const ukey = (str(pick(flat, ["ukey"])) ?? "").split("|")[0].toUpperCase();
  if (prefix === "EU" || ukey === "CTMARK" || fallback === "EU") return "EU";
  if (prefix.startsWith("WO") || ukey === "TMINT" || ukey === "DMI" || fallback === "WO") return "WO";
  return "FR";
}

// Path of the full record for a hit, e.g. /services/apidiffusion/api/marques/notice/WO1921267.
export function noticePath(raw: Record<string, unknown>): string | null {
  const href = (raw.xml as { href?: string } | undefined)?.href;
  if (!href) return null;
  try {
    const u = new URL(href);
    return `${u.pathname}${u.search}`;
  } catch {
    return null;
  }
}

function designTitle(v: unknown): string | null {
  const values = (Array.isArray(v) ? v : [v]).map(str).filter((s): s is string => Boolean(s));
  return values.length ? Array.from(new Set(values)).join(" / ") : null;
}

export function normalize(raw: Record<string, unknown>, type: RightType, collectionHint?: string): Right | null {
  const flat = flatten(raw);
  const number = str(pick(flat, ["applicationnumber", "designapplicationnumber", "numerodepot", "registrationnumber", "numeroenregistrement", "number", "numero", "st13", "documentid", "id"]));
  if (!number) return null;
  const office = officeOf(raw, flat, collectionHint);
  const ref = type === "dessin" ? str(pick(flat, ["designreference"])) : null;
  const cleanNumber = `${number.replace(/^(FR|EU|EM|WO)/i, "").replace(/[^\w]/g, "")}${ref ? `-${ref}` : ""}`;
  const name =
    (type === "marque"
      ? str(pick(flat, ["mark", "markverbalelementtext", "verbalelement", "marque", "markexp", "denomination", "title"]))
      : designTitle(pick(flat, ["designtitle", "title", "titre", "objet", "designation", "productindication"]))) ??
    (type === "marque" ? "(marque figurative)" : "(dessin ou modèle)");
  const countries = office === "WO" ? parseCountries(pick(flat, ["designatedcountries", "designatedcountrycode", "designatedcountry", "designations", "countries", "pays"])) : [];
  return {
    id: `${office}${cleanNumber}`,
    type,
    office,
    number: cleanNumber,
    name,
    owner: str(pick(flat, ["applicantname", "applicant", "deposant", "holdername", "holder", "titulaire", "applicants", "ownername"])) ?? "",
    classes:
      type === "marque"
        ? parseClasses(pick(flat, ["classnumber", "classes", "niceclass", "classdescriptiondetails", "nice"]))
        : (str(pick(flat, ["classnumber", "locarnoclass", "locarno"])) ?? "").match(/^\d{2}/) ? [Number((str(pick(flat, ["classnumber", "locarnoclass", "locarno"])) ?? "").slice(0, 2))] : [],
    filingDate: parseDate(pick(flat, ["applicationdate", "designapplicationdate", "filingdate", "datedepot", "depositdate"])),
    registrationDate: parseDate(pick(flat, ["registrationdate", "dateenregistrement", "publicationdate", "datepublication"])),
    expiryDate: parseDate(pick(flat, ["expirydate", "expirationdate", "dateexpiration", "renewaldate", "endofprotectiondate", "expiry"])),
    status: str(pick(flat, ["markcurrentstatuscode", "currentstatus", "status", "statut", "legalstatus", "etat"])) ?? "",
    territories: office === "EU" ? [{ code: "EU" }] : office === "FR" ? [{ code: "FR" }] : countries,
    // INPI image links need an authenticated session, so they cannot be passed to public image search.
    imageUrl: null,
    sourceUrl: `https://data.inpi.fr/${type === "marque" ? "marques" : "dessins_modeles"}/${office}${cleanNumber}`,
  };
}
