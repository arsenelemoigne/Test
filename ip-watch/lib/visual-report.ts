// Reports for a visual search: a PDF for the client's file, an Excel workbook of the results,
// and the client's register of visual searches. Server-only (pdf-lib, exceljs, sharp).
import ExcelJS from "exceljs";
import {
  PDFDocument,
  PDFHexString,
  StandardFonts,
  popGraphicsState,
  pushGraphicsState,
  rgb,
  setCharacterSpacing,
  type PDFFont,
  type PDFImage,
  type PDFPage,
  type RGB,
} from "pdf-lib";
import { LEVEL_LABEL, MODE_LABEL, levelCounts, type VisualLevel, type VisualResult, type VisualSearch, type VisualStep } from "./visual-types";

export type ReportContext = {
  clientName: string;
  appName: string;
  firmName: string;
  appUrl: string;
  // The uploaded image (JPEG; other formats are converted when possible).
  sourceImage: Buffer | null;
  // Stored result thumbnails (JPEG), by VisualResult.imagePath. May return null or throw.
  getImage: (path: string) => Promise<Buffer | null>;
};

// ---------------------------------------------------------------------------
// Shared vocabulary

type Level = VisualLevel | "none";

const LEVEL_ORDER: Level[] = ["tres_proche", "proche", "meme_theme", "ecarte", "none"];
const LEVEL_TEXT: Record<Level, string> = { ...LEVEL_LABEL, none: "Non évalué" };
const LEVEL_HEX: Record<Level, string> = {
  tres_proche: "#c0392b",
  proche: "#e67e22",
  meme_theme: "#2f6f9f",
  ecarte: "#9aa0a6",
  none: "#c4c8cc",
};
const LEVEL_MEANING: Record<Level, string> = {
  tres_proche: "même concept et plusieurs éléments visuels repris.",
  proche: "concept voisin ou éléments marquants en commun.",
  meme_theme: "même sujet ou même univers, sans reprise caractérisée.",
  ecarte: "rapprochement jugé non pertinent.",
  none: "résultat que l’IA n’a pas noté.",
};

const KIND_LABEL: Record<VisualResult["kind"], string> = { publicite: "Publicité", produit: "Produit", image: "Image", page: "Page web" };
const STEP_WORD: Record<VisualStep["status"], string> = { done: "fait", skipped: "ignoré", error: "erreur", running: "en cours" };
const STATUS_WORD: Record<VisualSearch["status"], string> = { running: "en cours", done: "terminée", error: "erreur" };
const ORIGIN_WORD: Record<VisualSearch["origin"], string> = { depot: "dépôt", email: "email" };
const SHORT_SOURCE: Record<string, string> = {
  "Google Vision": "Vision",
  "Google Lens": "Lens",
  "Meta Ad Library": "Meta Ads",
  "Google Shopping": "Shopping",
  "Google Images": "Images",
};

const levelOf = (r: VisualResult): Level => r.level ?? "none";
const levelRank = (r: VisualResult) => LEVEL_ORDER.indexOf(levelOf(r));
const byRelevance = (a: VisualResult, b: VisualResult) => levelRank(a) - levelRank(b) || (b.score ?? -1) - (a.score ?? -1) || a.n - b.n;
const isClose = (r: VisualResult) => r.level === "tres_proche" || r.level === "proche";

const TZ = "Europe/Paris";

function validDate(iso: string | null | undefined): Date | null {
  if (!iso) return null;
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? null : d;
}

// "11 novembre 2026, 14:32" (Paris time); "1er" for the first day of the month.
function frenchDate(iso: string | null | undefined, withTime = true): string {
  const d = validDate(iso);
  if (!d) return iso ?? "—";
  const parts = new Intl.DateTimeFormat("fr-FR", { timeZone: TZ, day: "numeric", month: "long", year: "numeric", hour: "2-digit", minute: "2-digit", hourCycle: "h23" }).formatToParts(d);
  const get = (t: string) => parts.find((p) => p.type === t)?.value ?? "";
  const day = get("day") === "1" ? "1er" : get("day");
  return `${day} ${get("month")} ${get("year")}${withTime ? `, ${get("hour")}:${get("minute")}` : ""}`;
}

// Excel stores dates without a time zone: give it the Paris wall-clock time.
function excelDate(iso: string | null | undefined): Date | null {
  const d = validDate(iso);
  if (!d) return null;
  const parts = new Intl.DateTimeFormat("en-GB", { timeZone: TZ, year: "numeric", month: "2-digit", day: "2-digit", hour: "2-digit", minute: "2-digit", second: "2-digit", hourCycle: "h23" }).formatToParts(d);
  const g = (t: string) => Number(parts.find((p) => p.type === t)?.value ?? 0);
  return new Date(Date.UTC(g("year"), g("month") - 1, g("day"), g("hour"), g("minute"), g("second")));
}

// Result dates are ISO (YYYY-MM-DD) or short texts that may contain ISO dates: show them as DD/MM/YYYY.
function resultDate(value: string | null): string | null {
  if (!value) return null;
  return value.replace(/(\d{4})-(\d{2})-(\d{2})(?:T[\d:.]+(?:Z|[+-]\d{2}:?\d{2})?)?/g, "$3/$2/$1");
}

const regionNames = (() => {
  try {
    return new Intl.DisplayNames(["fr"], { type: "region" });
  } catch {
    return null;
  }
})();

function countryLabel(value: string | null): string | null {
  const t = value?.trim();
  if (!t) return null;
  if (/^[A-Za-z]{2}$/.test(t)) {
    try {
      return regionNames?.of(t.toUpperCase()) ?? t;
    } catch {
      return t;
    }
  }
  return t;
}

function priceLabel(price: number | null, currency: string | null): string | null {
  if (price === null || price === undefined || !Number.isFinite(price)) return null;
  const cur = (currency ?? "").trim();
  if (/^[A-Za-z]{3}$/.test(cur)) {
    try {
      return new Intl.NumberFormat("fr-FR", { style: "currency", currency: cur.toUpperCase() }).format(price);
    } catch {
      // Unknown code: plain number below.
    }
  }
  const num = new Intl.NumberFormat("fr-FR", { maximumFractionDigits: 2 }).format(price);
  return cur ? `${num} ${cur}` : num;
}

// Scores are 0–100; tolerate 0–1.
function scoreValue(score: number | null): number | null {
  if (score === null || score === undefined || !Number.isFinite(score)) return null;
  return Math.round(score > 0 && score < 1 ? score * 100 : score);
}

function stepSummary(s: VisualStep): string {
  const parts: string[] = [];
  if (typeof s.count === "number") parts.push(`${s.count} résultat${s.count > 1 ? "s" : ""}`);
  if (s.detail) parts.push(s.detail);
  return parts.join(" · ");
}

function webUrl(url: string | null | undefined): string | null {
  const u = url?.trim();
  if (!u || !/^https?:\/\//i.test(u)) return null;
  return u;
}

function domainOf(url: string): string {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url.replace(/^https?:\/\//i, "").split("/")[0];
  }
}

const plural = (n: number, one: string, many = `${one}s`) => `${n} ${n > 1 ? many : one}`;

// ---------------------------------------------------------------------------
// Images

type SharpFn = (input?: Buffer, options?: { failOn?: "none" | "truncated" | "error" | "warning" }) => import("sharp").Sharp;
let sharpLoader: Promise<SharpFn | null> | null = null;

// Loaded lazily so a missing native binary only costs the image conversions, never the report.
function loadSharp(): Promise<SharpFn | null> {
  if (!sharpLoader) {
    sharpLoader = import("sharp")
      .then((m) => ((m as unknown as { default?: SharpFn }).default ?? (m as unknown as SharpFn)))
      .catch(() => null);
  }
  return sharpLoader;
}

async function toJpeg(buf: Buffer, maxPx: number, quality = 85): Promise<{ data: Buffer; width: number; height: number } | null> {
  const sharp = await loadSharp();
  if (!sharp) return null;
  try {
    const { data, info } = await sharp(buf, { failOn: "none" })
      .rotate()
      .resize({ width: maxPx, height: maxPx, fit: "inside", withoutEnlargement: true })
      .flatten({ background: "#ffffff" })
      .jpeg({ quality })
      .toBuffer({ resolveWithObject: true });
    return { data, width: info.width, height: info.height };
  } catch {
    return null;
  }
}

// Uploaded photos may carry an EXIF rotation, be huge, CMYK or not JPEG at all.
async function normalizeSource(buf: Buffer): Promise<Buffer | null> {
  const sharp = await loadSharp();
  if (!sharp) return null;
  try {
    const m = await sharp(buf, { failOn: "none" }).metadata();
    const big = buf.length > 2_500_000 || Math.max(m.width ?? 0, m.height ?? 0) > 2400;
    if (m.format === "jpeg" && (m.orientation ?? 1) <= 1 && !big && m.space !== "cmyk") return null;
    return (await toJpeg(buf, 2000))?.data ?? null;
  } catch {
    return null;
  }
}

async function embedJpeg(doc: PDFDocument, buf: Buffer | null, normalize: boolean): Promise<PDFImage | null> {
  if (!buf || !buf.length) return null;
  let data = buf;
  if (normalize) data = (await normalizeSource(buf)) ?? buf;
  else if (buf.length > 1_500_000) data = (await toJpeg(buf, 1200))?.data ?? buf;
  try {
    return await doc.embedJpg(data);
  } catch {
    const converted = await toJpeg(buf, 2000);
    if (!converted) return null;
    try {
      return await doc.embedJpg(converted.data);
    } catch {
      return null;
    }
  }
}

function jpegSize(b: Buffer): { width: number; height: number } | null {
  if (b.length < 4 || b[0] !== 0xff || b[1] !== 0xd8) return null;
  let i = 2;
  while (i + 9 < b.length) {
    if (b[i] !== 0xff) {
      i++;
      continue;
    }
    const m = b[i + 1];
    if (m === 0xff) {
      i++;
      continue;
    }
    if (m === 0xd8 || m === 0x01 || (m >= 0xd0 && m <= 0xd7)) {
      i += 2;
      continue;
    }
    if (m >= 0xc0 && m <= 0xcf && m !== 0xc4 && m !== 0xc8 && m !== 0xcc) return { height: b.readUInt16BE(i + 5), width: b.readUInt16BE(i + 7) };
    i += 2 + b.readUInt16BE(i + 2);
  }
  return null;
}

async function loadImage(getImage: ReportContext["getImage"], path: string): Promise<Buffer | null> {
  try {
    return await getImage(path);
  } catch {
    return null;
  }
}

async function mapPool<T, R>(items: T[], limit: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out = new Array<R>(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const k = next++;
      out[k] = await fn(items[k]);
    }
  };
  await Promise.all(Array.from({ length: Math.min(limit, items.length) }, worker));
  return out;
}

// ---------------------------------------------------------------------------
// Text that Helvetica (WinAnsi) can draw

// Filled from the embedded font; the fallback (printable ASCII + Latin-1) is only a safety net.
const CHARSET = new Set<number>();
function ensureCharset(font?: PDFFont) {
  if (CHARSET.size) return;
  if (font) for (const c of font.getCharacterSet()) CHARSET.add(c);
  if (!CHARSET.size) for (let c = 0x20; c <= 0xff; c++) if (c < 0x7f || c >= 0xa0) CHARSET.add(c);
}

const NBSP = String.fromCharCode(0xa0);
const FOREIGN = String.fromCharCode(0xe000); // placeholder for a letter of another script (CJK, Cyrillic, Arabic…)

// Replacements for characters WinAnsi lacks, by code point (kept numeric: several are invisible or look alike).
const CHAR_MAP = new Map<number, string>([
  [0x00ad, ""], // soft hyphen
  // dashes, primes, modifier apostrophes
  [0x2010, "-"], [0x2011, "-"], [0x2012, "–"], [0x2015, "—"], [0x2212, "-"], [0x2043, "-"], [0xfe58, "—"],
  [0x2032, "'"], [0x2033, '"'], [0x2035, "'"], [0x02b9, "'"], [0x02bc, "’"], [0x02bb, "‘"],
  // dots and ellipses
  [0x2219, "·"], [0x22c5, "·"], [0x2027, "·"], [0x30fb, "·"], [0xff65, "·"], [0x2981, "·"], [0x22ef, "…"], [0x2025, ".."],
  // CJK punctuation and brackets
  [0x3001, ", "], [0x3002, ". "], [0xff61, ". "], [0xff64, ", "],
  [0x300c, "“"], [0x300d, "”"], [0x300e, "“"], [0x300f, "”"], [0x301d, "“"], [0x301e, "”"],
  [0x3010, "["], [0x3011, "]"], [0x3014, "["], [0x3015, "]"], [0x3016, "["], [0x3017, "]"], [0x3018, "["], [0x3019, "]"], [0x301a, "["], [0x301b, "]"],
  [0x300a, "«"], [0x300b, "»"], [0x3008, "‹"], [0x3009, "›"], [0x301c, "~"], [0x2053, "~"],
  // arrows, comparisons, stars
  [0x2190, "<-"], [0x2192, "->"], [0x2194, "<->"], [0x21d2, "=>"], [0x27a1, "->"], [0x2264, "<="], [0x2265, ">="], [0x2260, "!="], [0x2248, "~"],
  [0x2605, "*"], [0x2606, "*"], [0x2731, "*"], [0x2733, "*"],
  // letters without a decomposition
  [0x0141, "L"], [0x0142, "l"], [0x0110, "D"], [0x0111, "d"], [0x0126, "H"], [0x0127, "h"], [0x0131, "i"], [0x0166, "T"], [0x0167, "t"], [0x1e9e, "SS"], [0x0138, "k"],
  [0x2116, "N°"], [0x2103, "°C"], [0x2109, "°F"],
]);

function encodable(s: string): boolean {
  for (const ch of s) if (!CHARSET.has(ch.codePointAt(0)!)) return false;
  return true;
}

const charCache = new Map<string, string>();

function safeChar(ch: string): string {
  const cp = ch.codePointAt(0)!;
  if (ch === "\n") return "\n";
  const mapped = CHAR_MAP.get(cp);
  if (mapped !== undefined) return mapped;
  if (CHARSET.has(cp)) return ch;
  const cached = charCache.get(ch);
  if (cached !== undefined) return cached;
  let out: string;
  if (cp === 0x202f || cp === 0x2007) out = CHARSET.has(0xa0) ? NBSP : " ";
  else if (/\s/u.test(ch)) out = " ";
  else if (cp >= 0xff01 && cp <= 0xff5e) out = String.fromCharCode(cp - 0xfee0);
  else {
    const nfkc = ch.normalize("NFKC");
    const nfd = ch.normalize("NFD").replace(/\p{M}/gu, "");
    if (nfkc !== ch && nfkc.trim() && encodable(nfkc)) out = nfkc;
    else if (nfd && nfd !== ch && encodable(nfd)) out = nfd;
    else if (/[\p{L}\p{N}]/u.test(ch)) out = FOREIGN;
    else out = ""; // emoji, symbols, marks, controls
  }
  charCache.set(ch, out);
  return out;
}

// Returns text Helvetica can encode. Typographic WinAnsi characters (’ “ ” – — … « » € œ, nbsp) are kept,
// accents of other Latin alphabets are dropped, emoji removed, other scripts shown as "[…]". Keeps "\n".
function safe(text: string | null | undefined): string {
  if (!text) return "";
  ensureCharset();
  let out = "";
  for (const ch of String(text).normalize("NFC").replace(/\r\n?/g, "\n")) out += safeChar(ch);
  return (
    out
      .replace(new RegExp(`${FOREIGN}(?:[\\s,.;:·\\-–—/|_+&()\\[\\]]*${FOREIGN})*`, "g"), "[…]")
      .replace(/[ \t]+/g, " ")
      .replace(/ *\n */g, "\n")
      .replace(/\(\s*\)|\[\s*\]/g, "")
      .replace(/ {2,}/g, " ")
      // French spacing: never break a line before : ; ! ? » or after «.
      .replace(/« /g, `«${NBSP}`)
      .replace(/ (»|[:;!?](?=\s|$))/g, `${NBSP}$1`)
      .trim()
  );
}

const oneLine = (s: string) => s.replace(/\s*\n\s*/g, " ");

// For short names and titles: say so when nothing readable remains.
function safeOr(text: string | null | undefined, fallback: string): string {
  const s = safe(text);
  return /[A-Za-z0-9À-ÿ]/.test(s) ? s : s ? `${fallback} ${s === "[…]" ? "" : s}`.trim() : "";
}

// ---------------------------------------------------------------------------
// PDF drawing kit

const W = 842;
const H = 595;
const M = 40;
const X1 = W - M;
const CW = W - 2 * M;
const BOTTOM = 50; // lowest point for page content (footer below)

const hex = (h: string): RGB => {
  const n = parseInt(h.slice(1), 16);
  return rgb(((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255);
};

const C = {
  ink: hex("#1f2328"),
  muted: hex("#6b6f76"),
  rule: hex("#d0d4d9"),
  hair: hex("#e7e9ec"),
  accent: hex("#24527a"),
  panel: hex("#f4f5f7"),
  white: rgb(1, 1, 1),
  error: hex("#c0392b"),
};
const LEVEL_RGB = Object.fromEntries(LEVEL_ORDER.map((l) => [l, hex(LEVEL_HEX[l])])) as Record<Level, RGB>;
const STEP_RGB: Record<VisualStep["status"], RGB> = { done: C.accent, skipped: C.muted, error: C.error, running: C.muted };

type TextOpts = { size?: number; bold?: boolean; color?: RGB; maxWidth?: number; align?: "left" | "right" | "center"; spacing?: number };
type ParaOpts = { size?: number; bold?: boolean; color?: RGB; lh?: number; maxLines?: number };

class Kit {
  constructor(
    readonly doc: PDFDocument,
    readonly regular: PDFFont,
    readonly bold: PDFFont,
    readonly ctx: ReportContext,
    readonly search: VisualSearch,
  ) {}

  font(bold?: boolean) {
    return bold ? this.bold : this.regular;
  }

  width(s: string, size: number, bold = false) {
    return this.font(bold).widthOfTextAtSize(s, size);
  }

  // Longest prefix that fits with an ellipsis. Text must already be safe.
  private cut(s: string, size: number, bold: boolean, maxWidth: number): string {
    const ell = "…";
    if (this.width(ell, size, bold) > maxWidth) return "";
    let lo = 0;
    let hi = s.length;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (this.width(s.slice(0, mid).trimEnd() + ell, size, bold) <= maxWidth) lo = mid;
      else hi = mid - 1;
    }
    return s.slice(0, lo).replace(/[\s,;:.\-–—·]+$/, "") + ell;
  }

  fit(text: string, size: number, bold: boolean, maxWidth: number): string {
    const s = oneLine(safe(text));
    return this.width(s, size, bold) <= maxWidth ? s : this.cut(s, size, bold, maxWidth);
  }

  wrap(text: string, size: number, bold: boolean, maxWidth: number, maxLines = 1000): string[] {
    const s = safe(text);
    if (!s) return [];
    const lines: string[] = [];
    let overflow = false;
    for (const paragraph of s.split("\n")) {
      let line = "";
      for (const word of paragraph.split(" ").filter(Boolean)) {
        const candidate = line ? `${line} ${word}` : word;
        if (this.width(candidate, size, bold) <= maxWidth) {
          line = candidate;
          continue;
        }
        if (line) lines.push(line);
        line = word;
        // A word longer than the line (URL, run of characters): break it.
        while (this.width(line, size, bold) > maxWidth && line.length > 1) {
          let k = line.length - 1;
          while (k > 1 && this.width(line.slice(0, k), size, bold) > maxWidth) k--;
          lines.push(line.slice(0, k));
          line = line.slice(k);
        }
        if (lines.length > maxLines) break;
      }
      if (line) lines.push(line);
      if (lines.length > maxLines) break;
    }
    if (lines.length > maxLines) overflow = true;
    const kept = lines.slice(0, maxLines);
    if (overflow && kept.length) {
      const last = kept[kept.length - 1].replace(/[\s,;:.\-–—·]+$/, "");
      kept[kept.length - 1] = this.width(`${last}…`, size, bold) <= maxWidth ? `${last}…` : this.cut(last, size, bold, maxWidth);
    }
    return kept;
  }

  // Draws one line; returns its width.
  text(page: PDFPage, text: string, x: number, y: number, o: TextOpts = {}): number {
    const size = o.size ?? 9;
    const bold = o.bold ?? false;
    const s = o.maxWidth !== undefined ? this.fit(text, size, bold, o.maxWidth) : oneLine(safe(text));
    if (!s) return 0;
    const spacing = o.spacing ?? 0;
    const w = this.width(s, size, bold) + spacing * Math.max(0, s.length - 1);
    const dx = o.align === "right" ? -w : o.align === "center" ? -w / 2 : 0;
    if (spacing) page.pushOperators(pushGraphicsState(), setCharacterSpacing(spacing));
    page.drawText(s, { x: x + dx, y, size, font: this.font(bold), color: o.color ?? C.ink });
    if (spacing) page.pushOperators(popGraphicsState());
    return w;
  }

  // Draws wrapped text from `top` (top of the first line); returns the top of what follows.
  para(page: PDFPage, text: string, x: number, top: number, width: number, o: ParaOpts = {}): number {
    const size = o.size ?? 9;
    const lh = o.lh ?? size * 1.32;
    const lines = this.wrap(text, size, o.bold ?? false, width, o.maxLines);
    lines.forEach((l, i) => page.drawText(l, { x, y: top - size * 0.8 - i * lh, size, font: this.font(o.bold), color: o.color ?? C.ink }));
    return top - lines.length * lh;
  }

  // Small uppercase section label; returns the top of what follows.
  label(page: PDFPage, text: string, x: number, top: number, color = C.accent): number {
    this.text(page, text.toUpperCase(), x, top - 6.5, { size: 7, bold: true, color, spacing: 0.7 });
    return top - 15;
  }

  rule(page: PDFPage, x0: number, x1: number, y: number, color = C.rule, thickness = 0.6) {
    page.drawLine({ start: { x: x0, y }, end: { x: x1, y }, thickness, color });
  }

  roundRect(page: PDFPage, x: number, y: number, w: number, h: number, r: number, color: RGB) {
    r = Math.min(r, w / 2, h / 2);
    const k = 0.5523 * r;
    const path =
      `M ${r} 0 H ${w - r} C ${w - r + k} 0 ${w} ${r - k} ${w} ${r} V ${h - r} C ${w} ${h - r + k} ${w - r + k} ${h} ${w - r} ${h} ` +
      `H ${r} C ${r - k} ${h} 0 ${h - r + k} 0 ${h - r} V ${r} C 0 ${r - k} ${r - k} 0 ${r} 0 Z`;
    page.drawSvgPath(path, { x, y: y + h, color });
  }

  // Coloured pill with white text (dark text on the pale "unrated" colour); returns its width.
  pill(page: PDFPage, text: string, x: number, y: number, level: Level, o: { size?: number; count?: number } = {}): number {
    const size = o.size ?? 7.5;
    const h = size + 7;
    const padX = size * 0.9;
    const label = safe(text);
    const count = o.count !== undefined ? String(o.count) : "";
    const gap = count ? size * 0.6 : 0;
    const w = this.width(label, size, true) + (count ? gap + this.width(count, size, false) : 0) + 2 * padX;
    const fg = level === "none" ? C.ink : C.white;
    this.roundRect(page, x, y, w, h, h / 2, LEVEL_RGB[level]);
    const base = y + (h - size * 0.7) / 2;
    page.drawText(label, { x: x + padX, y: base, size, font: this.bold, color: fg });
    if (count) page.drawText(count, { x: x + padX + this.width(label, size, true) + gap, y: base, size, font: this.regular, color: fg });
    return w;
  }

  // Clickable area opening `url`.
  link(page: PDFPage, url: string | null, x: number, y: number, w: number, h: number) {
    const u = webUrl(url);
    if (!u || w <= 0 || h <= 0) return;
    // URI actions are 7-bit ASCII; written as a hex string so parentheses and backslashes need no escaping.
    const ascii = u.replace(/[^\x21-\x7e]/g, (c) => encodeURIComponent(c));
    const hexBytes = Array.from(ascii, (c) => c.charCodeAt(0).toString(16).padStart(2, "0")).join("");
    const ctx = this.doc.context;
    const annot = ctx.obj({
      Type: "Annot",
      Subtype: "Link",
      Rect: [x, y, x + w, y + h],
      Border: [0, 0, 0],
      A: { Type: "Action", S: "URI", URI: PDFHexString.of(hexBytes) },
    });
    page.node.addAnnot(ctx.register(annot));
  }

  // Image fitted in a box, aspect preserved and centred; grey box when missing.
  image(page: PDFPage, img: PDFImage | null, x: number, y: number, w: number, h: number, o: { pad?: number; bg?: RGB | null; frame?: RGB; frameWidth?: number } = {}) {
    const pad = o.pad ?? 0;
    const bg = o.bg === undefined ? C.panel : o.bg;
    if (bg) page.drawRectangle({ x, y, width: w, height: h, color: bg });
    if (img) {
      const scale = Math.min((w - 2 * pad) / img.width, (h - 2 * pad) / img.height);
      const dw = img.width * scale;
      const dh = img.height * scale;
      page.drawImage(img, { x: x + (w - dw) / 2, y: y + (h - dh) / 2, width: dw, height: dh });
    } else {
      if (!bg) page.drawRectangle({ x, y, width: w, height: h, color: C.panel });
      const size = w < 50 ? 5.5 : w < 120 ? 6.5 : 8;
      const words = w < 50 ? ["sans", "image"] : ["image indisponible"];
      words.forEach((t, i) =>
        this.text(page, t, x + w / 2, y + h / 2 - size * 0.35 + ((words.length - 1) / 2 - i) * size * 1.2, { size, color: C.muted, align: "center" }),
      );
    }
    if (o.frame) {
      const fw = o.frameWidth ?? 2;
      page.drawRectangle({ x: x + fw / 2, y: y + fw / 2, width: w - fw, height: h - fw, borderColor: o.frame, borderWidth: fw });
    }
  }

  // "PI Pilot · Recherche visuelle" line, then the page title; returns the content top.
  header(page: PDFPage, title?: string, subtitle?: string): number {
    const y = H - 32;
    const x = M + this.text(page, this.ctx.appName, M, y, { size: 8, bold: true, color: C.accent });
    this.text(page, "· Recherche visuelle", x + this.width(" ", 8), y, { size: 8, color: C.muted });
    this.text(page, this.ctx.firmName, X1, y, { size: 8, color: C.muted, align: "right", maxWidth: CW / 2 });
    this.rule(page, M, X1, y - 9);
    if (!title) return y - 9;
    this.text(page, title, M, y - 40, { size: 18, bold: true, maxWidth: CW });
    if (!subtitle) return y - 56;
    this.text(page, subtitle, M, y - 57, { size: 9, color: C.muted, maxWidth: CW });
    return y - 72;
  }

  footer(page: PDFPage, index: number, total: number) {
    this.rule(page, M, X1, 34);
    const date = frenchDate(this.search.createdAt, false);
    this.text(page, `${this.ctx.appName} · ${this.ctx.clientName} · Recherche du ${date}`, M, 22, { size: 7.5, color: C.muted, maxWidth: CW - 80 });
    this.text(page, `page ${index} / ${total}`, X1, 22, { size: 7.5, color: C.muted, align: "right" });
  }

  disclaimer(page: PDFPage) {
    this.text(
      page,
      "Document de travail : recherche documentaire non exhaustive, sans valeur de constat ni d’avis juridique. Les rapprochements proposés par l’IA sont à vérifier par l’avocat.",
      M,
      42,
      { size: 7, color: C.muted, maxWidth: CW },
    );
  }
}

// ---------------------------------------------------------------------------
// PDF pages

const DISCLAIMER_TOP = 54; // content above the disclaimer line stops here

function drawCover(k: Kit, source: PDFImage | null) {
  const { search, ctx } = k;
  const a = search.analysis;
  const p = k.doc.addPage([W, H]);
  k.header(p);

  // Title block
  let top = H - 66;
  const titleLines = k.wrap(a?.titre || "Recherche visuelle", 24, true, CW, 2);
  titleLines.forEach((l, i) => p.drawText(l, { x: M, y: top - 20 - i * 28, size: 24, font: k.bold, color: C.ink }));
  top -= 20 + (titleLines.length - 1) * 28;
  const subtitle = `${MODE_LABEL[search.mode]} · ${ctx.clientName} · ${frenchDate(search.createdAt)}`;
  k.text(p, subtitle, M, top - 22, { size: 10.5, color: C.muted, maxWidth: CW });
  p.drawRectangle({ x: M, y: top - 38, width: 28, height: 2.5, color: C.accent });
  const colTop = top - 58;

  // Left half: the analysed image
  const lw = 330;
  const lt = k.label(p, "Image analysée", M, colTop);
  k.image(p, source, M, DISCLAIMER_TOP + 4, lw, lt - 4 - DISCLAIMER_TOP - 4, { pad: 10 });

  // Right half: measured first, then shortened (description, elements, note) until it fits.
  const rx = M + lw + 30;
  const rw = X1 - rx;
  const gap = 13;
  const LABEL = 15;
  const LH = 11.2;
  const floor = DISCLAIMER_TOP + 6;

  const status =
    search.status === "error" || search.error
      ? { title: "Recherche interrompue", text: search.error || "La recherche s’est arrêtée avant la fin.", color: C.error }
      : search.status === "running"
        ? { title: "Recherche en cours", text: "Résultats partiels au moment de l’export.", color: C.muted }
        : null;
  const statusLines = status ? k.wrap(status.text, 8.5, false, rw, 2) : [];
  const descAll = a?.description ? k.wrap(a.description, 9, false, rw) : [];
  const elements = a?.elements ?? [];
  const elemLines = elements.map((e) => k.wrap(e.label, 8.5, false, rw - 26, 2));
  const steps = search.steps.slice(0, 8);
  const noteAll = search.note ? k.wrap(search.note, 8.5, false, rw, 3) : [];

  const counts = levelCounts(search.results);
  const unrated = search.results.filter((r) => !r.level).length;
  const pills = LEVEL_ORDER.filter((l) => l !== "none" || unrated).map((l) => {
    const n = l === "none" ? unrated : counts[l];
    return { l, n, w: k.width(LEVEL_TEXT[l], 8, true) + k.width(String(n), 8) + 4.8 + 14.4 };
  });
  let pillRows = 1;
  pills.reduce((x, pl) => (x + pl.w > X1 ? (pillRows++, rx + pl.w + 6) : x + pl.w + 6), rx);

  const plan = { desc: descAll.length, elems: elements.length, note: noteAll.length };
  const height = () =>
    (status ? LABEL + statusLines.length * LH + gap : 0) +
    (plan.desc ? LABEL + plan.desc * 12 + gap : 0) +
    (elements.length ? LABEL + elemLines.slice(0, plan.elems).reduce((s, l) => s + l.length * LH + 2, 0) + (plan.elems < elements.length ? 13 : 0) + gap - 2 : 0) +
    (LABEL + 15 + 19 * (pillRows - 1) + 13 + gap + 4) +
    (steps.length ? LABEL + steps.length * 11.5 + gap : 0) +
    (plan.note ? LABEL + plan.note * LH : 0);
  const avail = colTop - floor;
  while (height() > avail) {
    if (plan.desc > 5) plan.desc--;
    else if (plan.elems > 4) plan.elems--;
    else if (plan.note > 1) plan.note--;
    else if (plan.desc > 3) plan.desc--;
    else if (plan.elems > 1) plan.elems--;
    else if (plan.desc > 1) plan.desc--;
    else break;
  }

  let rt = colTop;
  if (status) {
    rt = k.label(p, status.title, rx, rt, status.color);
    rt = k.para(p, status.text, rx, rt, rw, { size: 8.5, lh: LH, maxLines: 2, color: status.color }) - gap;
  }

  if (plan.desc && a?.description) {
    rt = k.label(p, "Ce que l’IA voit", rx, rt);
    rt = k.para(p, a.description, rx, rt, rw, { size: 9, lh: 12, maxLines: plan.desc }) - gap;
  }

  if (elements.length) {
    rt = k.label(p, "Éléments", rx, rt);
    elements.slice(0, plan.elems).forEach((e, j) => {
      k.text(p, e.id, rx, rt - 6.8, { size: 8.5, bold: true, color: C.accent, maxWidth: 24 });
      elemLines[j].forEach((l, i) => p.drawText(l, { x: rx + 26, y: rt - 6.8 - i * LH, size: 8.5, font: k.regular, color: C.ink }));
      rt -= elemLines[j].length * LH + 2;
    });
    if (plan.elems < elements.length) {
      k.text(p, `… et ${plural(elements.length - plan.elems, "autre élément", "autres éléments")} (voir l’Excel)`, rx + 26, rt - 6.8, { size: 8, color: C.muted });
      rt -= 13;
    }
    rt -= gap - 2;
  }

  // Results per level
  rt = k.label(p, "Résultats", rx, rt);
  let px = rx;
  let py = rt - 15;
  for (const pl of pills) {
    if (px + pl.w > X1) {
      px = rx;
      py -= 19;
    }
    px += k.pill(p, LEVEL_TEXT[pl.l], px, py, pl.l, { size: 8, count: pl.n }) + 6;
  }
  const withImage = search.results.filter((r) => r.imagePath).length;
  const total = search.results.length;
  k.text(p, total ? `${plural(total, "résultat")} au total, dont ${withImage} avec image` : "Aucun résultat trouvé.", rx, py - 13, { size: 8, color: C.muted, maxWidth: rw });
  rt = py - 13 - gap - 4;

  // Sources
  if (steps.length) {
    rt = k.label(p, "Sources consultées", rx, rt);
    const nameW = Math.min(150, Math.max(...steps.map((s) => k.width(safe(s.name), 8.5, true))) + 10);
    for (const s of steps) {
      if (rt - 11.5 < floor) break;
      const y = rt - 7.5;
      k.text(p, s.name, rx, y, { size: 8.5, bold: true, maxWidth: nameW - 8 });
      const ww = k.text(p, STEP_WORD[s.status], rx + nameW, y, { size: 8.5, color: STEP_RGB[s.status] });
      const summary = stepSummary(s);
      if (summary) k.text(p, `· ${summary}`, rx + nameW + ww + 5, y, { size: 8.5, color: C.muted, maxWidth: rw - nameW - ww - 5 });
      rt -= 11.5;
    }
    rt -= gap;
  }

  // Note
  if (plan.note && rt - LABEL - LH >= floor - 1) {
    rt = k.label(p, "Note", rx, rt);
    const room = Math.max(1, Math.floor((rt - floor + 3) / LH));
    k.para(p, search.note ?? "", rx, rt, rw, { size: 8.5, lh: LH, maxLines: Math.min(plan.note, room) });
  }

  k.disclaimer(p);
}

function drawLegend(k: Kit, p: PDFPage, rightX: number, y: number, levels: Level[]) {
  // Right-aligned row of coloured squares with labels.
  const items = levels.map((l) => ({ l, w: 10 + k.width(LEVEL_TEXT[l], 7.5) + 12 }));
  let x = rightX - items.reduce((s, i) => s + i.w, 0) + 12;
  for (const { l, w } of items) {
    p.drawRectangle({ x, y: y - 0.5, width: 7, height: 7, color: LEVEL_RGB[l] });
    k.text(p, LEVEL_TEXT[l], x + 10, y, { size: 7.5, color: C.muted });
    x += w;
  }
}

function drawMosaic(k: Kit, results: VisualResult[], images: Map<string, PDFImage | null>) {
  const cols = 7;
  const rows = 4;
  const gapX = 10;
  const cellW = (CW - gapX * (cols - 1)) / cols;
  const perPage = cols * rows;
  const pages = Math.ceil(results.length / perPage);
  const present = LEVEL_ORDER.filter((l) => results.some((r) => levelOf(r) === l));
  for (let pg = 0; pg < pages; pg++) {
    const p = k.doc.addPage([W, H]);
    const title = pages > 1 && pg > 0 ? "Mosaïque des résultats (suite)" : "Mosaïque des résultats";
    const subtitle = `${plural(results.length, "résultat")} avec image · le cadre indique le niveau proposé par l’IA, du plus proche au plus éloigné`;
    const top = k.header(p, title, subtitle);
    drawLegend(k, p, X1, H - 72, present);
    const gridTop = top - 6;
    const rowH = (gridTop - BOTTOM + 6) / rows;
    const cellH = rowH - 24;
    results.slice(pg * perPage, (pg + 1) * perPage).forEach((r, i) => {
      const c = i % cols;
      const row = Math.floor(i / cols);
      const x = M + c * (cellW + gapX);
      const yTop = gridTop - row * rowH;
      const y = yTop - cellH;
      const img = r.imagePath ? images.get(r.imagePath) ?? null : null;
      k.image(p, img, x, y, cellW, cellH, { pad: 4, frame: LEVEL_RGB[levelOf(r)], frameWidth: 2 });
      k.link(p, r.url, x, y, cellW, cellH);
      const nw = k.text(p, `n°${r.n}`, x, y - 10, { size: 7.5, bold: true });
      k.text(p, SHORT_SOURCE[r.source] ?? r.source, x + nw + 4, y - 10, { size: 7, color: C.muted, maxWidth: cellW - nw - 4 });
    });
  }
}

function drawDetail(k: Kit, r: VisualResult, source: PDFImage | null, img: PDFImage | null) {
  const p = k.doc.addPage([W, H]);
  const lvl = levelOf(r);
  const top = k.header(p, `Résultat n°${r.n}`, `${LEVEL_TEXT[lvl]} · ${r.source} · ${KIND_LABEL[r.kind] ?? r.kind}`);

  // Two images side by side
  const pw = 236;
  const gap = 14;
  const labelTop = top - 4;
  const imgTop = labelTop - 15;
  const imgH = imgTop - BOTTOM - 2;
  k.label(p, "Votre image", M, labelTop, C.muted);
  k.image(p, source, M, imgTop - imgH, pw, imgH, { pad: 8 });
  k.label(p, `Résultat n°${r.n}`, M + pw + gap, labelTop, LEVEL_RGB[lvl]);
  k.image(p, img, M + pw + gap, imgTop - imgH, pw, imgH, { pad: 8, frame: LEVEL_RGB[lvl], frameWidth: 2 });
  k.link(p, r.url, M + pw + gap, imgTop - imgH, pw, imgH);

  // Facts
  const fx = M + 2 * pw + gap + 24;
  const fw = X1 - fx;
  const lw = 74;
  const vw = fw - lw;
  let t = labelTop;
  const pillW = k.pill(p, LEVEL_TEXT[lvl], fx, t - 17, lvl, { size: 9 });
  const score = scoreValue(r.score);
  if (score !== null) {
    const sx = fx + pillW + 10;
    const sw = k.text(p, String(score), sx, t - 13.5, { size: 13, bold: true });
    k.text(p, "/100", sx + sw + 1, t - 13.5, { size: 8.5, color: C.muted });
  }
  t -= 32;

  const facts: [string, string | null, number?][] = [
    ["Source", `${r.source} · ${KIND_LABEL[r.kind] ?? r.kind}`],
    ["Titre", safeOr(r.title, "[titre en caractères non latins]") || "—", 3],
    ["Annonceur", safeOr(r.advertiser, "[nom en caractères non latins]") || "—", 2],
    ["Date", resultDate(r.date) ?? "—", 2],
    ["Pays", countryLabel(r.country) ?? "—"],
  ];
  const price = priceLabel(r.price, r.currency);
  if (price) facts.push(["Prix", price]);
  if (r.category || k.search.mode === "contrefacon") facts.push(["Catégorie", r.category ?? "—"]);
  facts.push(["Trouvé par", r.foundBy || "—", 2]);

  for (const [label, value, max] of facts) {
    const lines = k.wrap(value ?? "—", 8.5, false, vw, max ?? 1);
    k.text(p, label.toUpperCase(), fx, t - 7, { size: 6.5, bold: true, color: C.muted, spacing: 0.5 });
    lines.forEach((l, i) => p.drawText(l, { x: fx + lw, y: t - 7 - i * 11, size: 8.5, font: k.regular, color: C.ink }));
    t -= Math.max(1, lines.length) * 11 + 5;
  }

  // Link
  const url = webUrl(r.url);
  k.text(p, "LIEN", fx, t - 7, { size: 6.5, bold: true, color: C.muted, spacing: 0.5 });
  if (url) {
    const lines = k.wrap(url.replace(/^https?:\/\//i, ""), 8, false, vw, 2);
    lines.forEach((l, i) => p.drawText(l, { x: fx + lw, y: t - 7 - i * 10.5, size: 8, font: k.regular, color: C.accent }));
    k.link(p, url, fx + lw - 2, t - 7 - (lines.length - 1) * 10.5 - 3, vw + 2, (lines.length - 1) * 10.5 + 12);
    t -= lines.length * 10.5 + 5;
  } else {
    p.drawText(safe(r.url) ? k.fit(r.url, 8, false, vw) : "—", { x: fx + lw, y: t - 7, size: 8, font: k.regular, color: C.ink });
    t -= 16;
  }

  // Similarities and differences share what is left.
  t -= 8;
  k.rule(p, fx, X1, t + 2, C.hair);
  t -= 8;
  const blocks: [string, string][] = [
    ["Points communs", r.similar || "Non précisé."],
    ["Différences", r.differs || "Non précisé."],
  ];
  const lh = 11.6;
  const room = t - BOTTOM - 2 * 15 - 10;
  const wanted = blocks.map(([, txt]) => k.wrap(txt, 8.5, false, fw, 50).length);
  let maxA = Math.floor(room / lh / 2);
  let maxB = maxA;
  if (wanted[0] < maxA) maxB = Math.floor(room / lh) - wanted[0];
  else if (wanted[1] < maxB) maxA = Math.floor(room / lh) - wanted[1];
  blocks.forEach(([title, txt], i) => {
    t = k.label(p, title, fx, t, C.accent);
    t = k.para(p, txt, fx, t, fw, { size: 8.5, lh, maxLines: Math.max(1, i === 0 ? maxA : maxB) }) - 10;
  });
}

type Column = { title: string; w: number; align?: "left" | "center" };

const TABLE_COLUMNS: Column[] = [
  { title: "", w: 44 },
  { title: "N°", w: 30 },
  { title: "Niveau", w: 70 },
  { title: "Source", w: 80 },
  { title: "Titre", w: 168 },
  { title: "Annonceur", w: 90 },
  { title: "Date", w: 60 },
  { title: "Motif", w: 160 },
  { title: "Lien", w: 60 },
];

function drawTable(k: Kit, results: VisualResult[], images: Map<string, PDFImage | null>, hasCloseOverflow: boolean) {
  const rowH = 42;
  const size = 7.5;
  const lh = 9.2;
  const padX = 5;
  let p: PDFPage | null = null;
  let y = 0;
  let pageNo = 0;
  const xs: number[] = [];
  TABLE_COLUMNS.reduce((x, c) => (xs.push(x), x + c.w), M);
  const levels = hasCloseOverflow ? "Proches non détaillés, même thème, écartés et non évalués" : "Même thème, écartés et non évalués";

  const newPage = () => {
    p = k.doc.addPage([W, H]);
    pageNo++;
    const top = k.header(p, pageNo > 1 ? "Autres résultats (suite)" : "Autres résultats", `${levels} · ${plural(results.length, "résultat")}, du plus proche au plus éloigné`);
    const hy = top - 10;
    TABLE_COLUMNS.forEach((c, i) => c.title && k.text(p!, c.title.toUpperCase(), xs[i] + padX, hy - 6, { size: 6.5, bold: true, color: C.muted, spacing: 0.5 }));
    k.rule(p, M, X1, hy - 12, C.rule, 0.8);
    y = hy - 12;
  };

  const cell = (page: PDFPage, text: string, col: number, rowTop: number, o: { maxLines?: number; bold?: boolean; color?: RGB; x?: number } = {}) => {
    const c = TABLE_COLUMNS[col];
    const x = o.x ?? xs[col] + padX;
    const w = xs[col] + c.w - padX - x;
    const lines = k.wrap(text, size, o.bold ?? false, w, o.maxLines ?? 1);
    const blockH = (lines.length - 1) * lh + size * 0.7;
    const first = rowTop - (rowH - blockH) / 2 - size * 0.7;
    lines.forEach((l, i) => page.drawText(l, { x, y: first - i * lh, size, font: k.font(o.bold), color: o.color ?? C.ink }));
  };

  for (const r of results) {
    if (!p || y - rowH < BOTTOM - 8) newPage();
    const page = p!;
    const rowTop = y;
    const lvl = levelOf(r);
    // Thumbnail
    const tb = 34;
    const tx = xs[0] + (TABLE_COLUMNS[0].w - tb) / 2 - 2;
    const ty = rowTop - (rowH + tb) / 2;
    if (r.imagePath) k.image(page, images.get(r.imagePath) ?? null, tx, ty, tb, tb, { pad: 0, bg: C.panel });
    else k.image(page, null, tx, ty, tb, tb, { bg: C.panel });
    cell(page, `n°${r.n}`, 1, rowTop, { bold: true });
    // Level: dot + label
    const mid = rowTop - rowH / 2;
    page.drawCircle({ x: xs[2] + padX + 3, y: mid, size: 3, color: LEVEL_RGB[lvl] });
    cell(page, LEVEL_TEXT[lvl], 2, rowTop, { x: xs[2] + padX + 10, maxLines: 2 });
    cell(page, r.source, 3, rowTop, { maxLines: 2 });
    cell(page, safeOr(r.title, "[titre en caractères non latins]") || "—", 4, rowTop, { maxLines: 3 });
    cell(page, safeOr(r.advertiser, "[caractères non latins]") || "—", 5, rowTop, { maxLines: 3, color: r.advertiser ? C.ink : C.muted });
    cell(page, resultDate(r.date) ?? "—", 6, rowTop, { maxLines: 3, color: r.date ? C.ink : C.muted });
    const reason = lvl === "ecarte" ? r.differs || r.similar : r.similar || r.differs;
    cell(page, reason || "—", 7, rowTop, { maxLines: 3, color: reason ? C.ink : C.muted });
    const url = webUrl(r.url);
    if (url) {
      cell(page, domainOf(url), 8, rowTop, { maxLines: 2, color: C.accent });
      k.link(page, url, xs[8], rowTop - rowH + 2, TABLE_COLUMNS[8].w, rowH - 4);
      k.link(page, url, tx, ty, tb, tb);
    } else cell(page, "—", 8, rowTop, { color: C.muted });
    y = rowTop - rowH;
    k.rule(page, M, X1, y, C.hair, 0.5);
  }
}

const LIMITS = [
  "Meta Ad Library ne couvre que les publicités diffusées dans l’Union européenne, sur environ la dernière année.",
  "La recherche par image ne trouve que des images publiées en ligne : les supports imprimés, l’affichage et la télévision ne sont pas couverts.",
  "Les niveaux de proximité et les commentaires sont proposés par une IA ; ils doivent être vérifiés par l’avocat.",
  "L’absence de résultat ne prouve pas l’absence d’antériorité ou de copie.",
  "Les liens et les images reproduits appartiennent à leurs titulaires ; ils sont cités à titre documentaire.",
];

function drawMethod(k: Kit) {
  const { search, ctx } = k;
  const a = search.analysis;
  const p = k.doc.addPage([W, H]);
  const top = k.header(p, "Méthode et limites", `Recherche ${search.id} · ${MODE_LABEL[search.mode]} · origine : ${ORIGIN_WORD[search.origin]}`);
  const lw = 380;
  const rx = M + lw + 34;
  const rw = X1 - rx;
  const floor = DISCLAIMER_TOP + 6;

  // Left: how the search ran
  let t = top - 4;
  t = k.label(p, "Déroulé", M, t);
  const facts: [string, string][] = [
    ["Lancée le", frenchDate(search.createdAt)],
    ["Terminée le", search.finishedAt ? frenchDate(search.finishedAt) : STATUS_WORD[search.status]],
    ["Rapport généré le", `${frenchDate(new Date().toISOString())} par ${ctx.appName}`],
  ];
  for (const [label, value] of facts) {
    k.text(p, label, M, t - 7.5, { size: 8.5, color: C.muted });
    k.text(p, value, M + 100, t - 7.5, { size: 8.5, maxWidth: lw - 100 });
    t -= 12;
  }
  t -= 12;

  t = k.label(p, "Sources consultées", M, t);
  for (const s of search.steps) {
    if (t - 12 < floor) break;
    k.text(p, s.name, M, t - 6.8, { size: 8.5, bold: true, maxWidth: 140 });
    k.text(p, STEP_WORD[s.status], M + 150, t - 6.8, { size: 8.5, color: STEP_RGB[s.status] });
    const summary = stepSummary(s) || "—";
    const after = k.para(p, summary, M + 200, t, lw - 200, { size: 8.5, lh: 11, maxLines: 2, color: C.muted });
    t = Math.min(t - 12, after) - 4;
  }
  t -= 10;

  const keywords: [string, string][] = [];
  if (a?.motsCles?.length) keywords.push(["Mots-clés", a.motsCles.join(", ")]);
  if (a?.keywordsEn?.length) keywords.push(["En anglais", a.keywordsEn.join(", ")]);
  if (a?.marquesVisibles?.length) keywords.push(["Marques visibles", a.marquesVisibles.join(", ")]);
  if (a?.texteVisible) keywords.push(["Texte visible", a.texteVisible]);
  if (keywords.length && t - 30 > floor) {
    t = k.label(p, "Requêtes", M, t);
    for (const [label, value] of keywords) {
      if (t - 12 < floor) break;
      k.text(p, label, M, t - 6.8, { size: 8.5, color: C.muted });
      const room = Math.max(1, Math.min(3, Math.floor((t - floor) / 11)));
      t = Math.min(t - 12, k.para(p, value, M + 100, t, lw - 100, { size: 8.5, lh: 11, maxLines: room })) - 4;
    }
  }

  // Right: limits and how to read the levels
  let rt = top - 4;
  rt = k.label(p, "Limites", rx, rt);
  for (const line of LIMITS) {
    p.drawCircle({ x: rx + 2, y: rt - 5.2, size: 1.6, color: C.accent });
    rt = k.para(p, line, rx + 10, rt, rw - 10, { size: 8.5, lh: 11.4 }) - 5;
  }
  rt -= 12;
  rt = k.label(p, "Lecture des niveaux", rx, rt);
  for (const l of LEVEL_ORDER) {
    // Pill 14 pt high; its text baseline sits at rt - 9.45, the definition shares it.
    k.pill(p, LEVEL_TEXT[l], rx, rt - 14, l, { size: 7 });
    rt = Math.min(rt - 14, k.para(p, LEVEL_MEANING[l], rx + 76, rt - 2.65, rw - 76, { size: 8.5, lh: 11 })) - 6;
  }
  k.disclaimer(p);
}

export async function buildVisualPdf(search: VisualSearch, ctx: ReportContext): Promise<Uint8Array> {
  const doc = await PDFDocument.create();
  const regular = await doc.embedFont(StandardFonts.Helvetica);
  const bold = await doc.embedFont(StandardFonts.HelveticaBold);
  ensureCharset(regular);
  const title = search.analysis?.titre || "Recherche visuelle";
  doc.setTitle(`Recherche visuelle — ${title}`);
  doc.setAuthor(ctx.firmName);
  doc.setSubject(`${MODE_LABEL[search.mode]} · ${ctx.clientName}`);
  doc.setCreator(ctx.appName);
  doc.setProducer(ctx.appName);
  doc.setLanguage("fr-FR");
  doc.setCreationDate(new Date());

  // Embed every image once, a few at a time.
  const source = await embedJpeg(doc, ctx.sourceImage, true);
  const paths = Array.from(new Set(search.results.map((r) => r.imagePath).filter((x): x is string => !!x)));
  const images = new Map<string, PDFImage | null>();
  const embedded = await mapPool(paths, 8, async (path) => embedJpeg(doc, await loadImage(ctx.getImage, path), false));
  paths.forEach((path, i) => images.set(path, embedded[i]));

  const k = new Kit(doc, regular, bold, ctx, search);
  const sorted = [...search.results].sort(byRelevance);

  drawCover(k, source);

  const withImage = sorted.filter((r) => r.imagePath);
  if (withImage.length) drawMosaic(k, withImage, images);

  const close = sorted.filter(isClose);
  const detailed = close.slice(0, 15);
  for (const r of detailed) drawDetail(k, r, source, r.imagePath ? images.get(r.imagePath) ?? null : null);

  const others = [...close.slice(15), ...sorted.filter((r) => !isClose(r))];
  if (others.length) drawTable(k, others, images, close.length > 15);

  drawMethod(k);

  const pages = doc.getPages();
  pages.forEach((page, i) => k.footer(page, i + 1, pages.length));
  return doc.save();
}

// ---------------------------------------------------------------------------
// Excel

// Text exceljs can write into XML: no control characters or lone surrogates, within Excel's cell limit.
function xml(value: string | null | undefined): string {
  if (!value) return "";
  let out = "";
  for (const ch of String(value)) {
    const cp = ch.codePointAt(0)!;
    if ((cp < 0x20 && cp !== 0x09 && cp !== 0x0a && cp !== 0x0d) || cp === 0xfffe || cp === 0xffff || (cp >= 0xd800 && cp <= 0xdfff)) continue;
    out += ch;
  }
  return out.slice(0, 32000);
}

const argb = (h: string) => `FF${h.slice(1).toUpperCase()}`;
// Always name the font: without it LibreOffice (and some viewers) fall back to a serif.
const xfont = (f: Partial<ExcelJS.Font> = {}): Partial<ExcelJS.Font> => ({ name: "Calibri", size: 11, ...f });
const solid = (h: string): ExcelJS.Fill => ({ type: "pattern", pattern: "solid", fgColor: { argb: argb(h) } });
const thinBottom: Partial<ExcelJS.Borders> = { bottom: { style: "thin", color: { argb: argb("#d0d4d9") } } };

function hyperlink(text: string, url: string | null | undefined): ExcelJS.CellValue {
  const u = webUrl(url);
  // Excel refuses very long hyperlink addresses; keep those as plain text.
  if (!u || u.length > 2000) return xml(url ?? "") || null;
  return { text: xml(text), hyperlink: u, tooltip: u.slice(0, 250) };
}

function styleHyperlink(cell: ExcelJS.Cell) {
  if (cell.value && typeof cell.value === "object" && "hyperlink" in cell.value) cell.font = xfont({ color: { argb: argb("#24527a") }, underline: true });
}

function titleRows(ws: ExcelJS.Worksheet, columns: number, title: string, subtitle: string) {
  ws.mergeCells(1, 1, 1, columns);
  ws.mergeCells(2, 1, 2, columns);
  const t = ws.getCell(1, 1);
  t.value = xml(title);
  t.font = xfont({ bold: true, size: 14, color: { argb: argb("#1f2328") } });
  t.alignment = { vertical: "middle" };
  ws.getRow(1).height = 26;
  const s = ws.getCell(2, 1);
  s.value = xml(subtitle);
  s.font = xfont({ size: 10, color: { argb: argb("#6b6f76") } });
  ws.getRow(2).height = 18;
}

function headerRow(ws: ExcelJS.Worksheet, rowNumber: number, titles: string[], fills: (string | null)[] = []) {
  const row = ws.getRow(rowNumber);
  titles.forEach((title, i) => {
    const cell = row.getCell(i + 1);
    cell.value = title;
    cell.font = xfont({ bold: true, color: { argb: "FFFFFFFF" } });
    cell.fill = solid(fills[i] ?? "#24527a");
    cell.alignment = { vertical: "middle", wrapText: true };
  });
  row.height = 22;
}

// A4 landscape, one page wide, header repeated on every printed page.
function printSetup(ws: ExcelJS.Worksheet, headerRowNumber: number) {
  ws.pageSetup = {
    paperSize: 9,
    orientation: "landscape",
    fitToPage: true,
    fitToWidth: 1,
    fitToHeight: 0,
    printTitlesRow: `${headerRowNumber}:${headerRowNumber}`,
    margins: { left: 0.4, right: 0.4, top: 0.5, bottom: 0.5, header: 0.3, footer: 0.3 },
  };
}

const RESULT_COLUMNS: { header: string; width: number }[] = [
  { header: "n°", width: 6 },
  { header: "Niveau", width: 13 },
  { header: "Score", width: 7 },
  { header: "Source", width: 16 },
  { header: "Type", width: 11 },
  { header: "Titre", width: 42 },
  { header: "Annonceur / vendeur", width: 22 },
  { header: "Date", width: 14 },
  { header: "Pays", width: 12 },
  { header: "Prix", width: 10 },
  { header: "Devise", width: 8 },
  { header: "Lien", width: 38 },
  { header: "Points communs", width: 44 },
  { header: "Différences", width: 44 },
  { header: "Catégorie", width: 13 },
  { header: "Trouvé par", width: 26 },
  { header: "Image", width: 15 },
];

export async function buildVisualXlsx(search: VisualSearch, ctx: ReportContext): Promise<Buffer> {
  const wb = new ExcelJS.Workbook();
  wb.creator = ctx.appName;
  wb.created = new Date();
  const results = [...search.results].sort(byRelevance);
  const a = search.analysis;
  const titre = a?.titre || "Recherche visuelle";
  const date = frenchDate(search.createdAt);

  // Results
  const ws = wb.addWorksheet("Résultats");
  ws.columns = RESULT_COLUMNS.map((c) => ({ width: c.width }));
  titleRows(ws, RESULT_COLUMNS.length, `Recherche visuelle — ${titre} — ${date} — ${plural(results.length, "résultat")}`, `${MODE_LABEL[search.mode]} · ${ctx.clientName} · ${ctx.firmName} · généré par ${ctx.appName}`);
  const HEADER = 3;
  headerRow(ws, HEADER, RESULT_COLUMNS.map((c) => c.header));

  const thumbs = await mapPool(results, 8, async (r) => {
    if (!r.imagePath) return null;
    const buf = await loadImage(ctx.getImage, r.imagePath);
    if (!buf?.length) return null;
    const converted = await toJpeg(buf, 240, 80);
    if (converted) return { buffer: converted.data, width: converted.width, height: converted.height };
    const size = jpegSize(buf);
    return size ? { buffer: buf, ...size } : null;
  });
  const anyImage = thumbs.some(Boolean);
  const imageCol = RESULT_COLUMNS.length - 1; // 0-based
  const cellPx = { w: RESULT_COLUMNS[imageCol].width * 7 + 5, h: (60 * 96) / 72 };

  results.forEach((r, i) => {
    const lvl = levelOf(r);
    const iso = r.date && /^\d{4}-\d{2}-\d{2}$/.test(r.date.trim()) ? excelDate(`${r.date.trim()}T12:00:00Z`) : null;
    const row = ws.getRow(HEADER + 1 + i);
    row.values = [
      r.n,
      LEVEL_TEXT[lvl],
      scoreValue(r.score),
      xml(r.source),
      KIND_LABEL[r.kind] ?? r.kind,
      xml(r.title),
      xml(r.advertiser),
      iso ?? (xml(r.date) || null),
      xml(countryLabel(r.country)),
      r.price ?? null,
      xml(r.currency),
      hyperlink(r.url, r.url),
      xml(r.similar),
      xml(r.differs),
      xml(r.category),
      xml(r.foundBy),
      null,
    ];
    row.height = anyImage ? 60 : 30;
    row.eachCell({ includeEmpty: true }, (cell, col) => {
      cell.alignment = { vertical: "middle", wrapText: [6, 7, 8, 13, 14, 16].includes(col) };
      cell.border = thinBottom;
    });
    const level = row.getCell(2);
    level.fill = solid(LEVEL_HEX[lvl]);
    level.font = xfont({ bold: true, color: { argb: lvl === "none" ? argb("#1f2328") : "FFFFFFFF" } });
    level.alignment = { vertical: "middle", horizontal: "center" };
    row.getCell(1).alignment = { vertical: "middle", horizontal: "center" };
    row.getCell(3).alignment = { vertical: "middle", horizontal: "center" };
    if (iso) row.getCell(8).numFmt = "dd/mm/yyyy";
    row.getCell(10).numFmt = "#,##0.00";
    styleHyperlink(row.getCell(12));

    const thumb = thumbs[i];
    if (thumb) {
      const scale = Math.min((cellPx.w - 8) / thumb.width, (cellPx.h - 8) / thumb.height);
      const id = wb.addImage({ buffer: thumb.buffer as unknown as ExcelJS.Buffer, extension: "jpeg" });
      ws.addImage(id, {
        tl: { col: imageCol + 0.05, row: HEADER + i + 0.05 } as ExcelJS.Anchor,
        ext: { width: Math.round(thumb.width * scale), height: Math.round(thumb.height * scale) },
        editAs: "oneCell",
      });
    }
  });
  ws.views = [{ state: "frozen", ySplit: HEADER, xSplit: 1 }];
  printSetup(ws, HEADER);
  ws.autoFilter = { from: { row: HEADER, column: 1 }, to: { row: HEADER + Math.max(results.length, 1), column: RESULT_COLUMNS.length } };

  // Search metadata
  const info = wb.addWorksheet("Recherche");
  info.columns = [{ width: 24 }, { width: 100 }];
  info.pageSetup = { paperSize: 9, orientation: "portrait", fitToPage: true, fitToWidth: 1, fitToHeight: 0 };
  titleRows(info, 2, `Recherche visuelle — ${titre}`, `${ctx.clientName} · ${date}`);
  const pairs: [string, ExcelJS.CellValue][] = [
    ["Identifiant", search.id],
    ["Date", excelDate(search.createdAt)],
    ["Terminée le", excelDate(search.finishedAt)],
    ["Client", xml(ctx.clientName)],
    ["Mode", MODE_LABEL[search.mode]],
    ["Origine", ORIGIN_WORD[search.origin]],
    ["Statut", STATUS_WORD[search.status] + (search.error ? ` : ${xml(search.error)}` : "")],
    ["Note", xml(search.note)],
    ["Titre", xml(a?.titre)],
    ["Description", xml(a?.description)],
    ["Éléments", xml((a?.elements ?? []).map((e) => `${e.id} — ${e.label}`).join("\n"))],
    ["Mots-clés", xml((a?.motsCles ?? []).join(", "))],
    ["Mots-clés (anglais)", xml((a?.keywordsEn ?? []).join(", "))],
    ["Marques visibles", xml((a?.marquesVisibles ?? []).join(", "))],
    ["Texte visible", xml(a?.texteVisible)],
    ["Résultats", results.length],
    ...LEVEL_ORDER.map((l): [string, ExcelJS.CellValue] => [`  ${LEVEL_TEXT[l]}`, results.filter((r) => levelOf(r) === l).length]),
    ["Fiche", hyperlink(`${ctx.appUrl.replace(/\/+$/, "")}/recherche-visuelle?id=${encodeURIComponent(search.id)}`, `${ctx.appUrl.replace(/\/+$/, "")}/recherche-visuelle?id=${encodeURIComponent(search.id)}`)],
  ];
  let rowNo = 4;
  for (const [key, value] of pairs) {
    const row = info.getRow(rowNo++);
    row.getCell(1).value = key;
    row.getCell(1).font = xfont({ bold: true });
    row.getCell(1).alignment = { vertical: "top" };
    const cell = row.getCell(2);
    cell.value = value ?? null;
    cell.alignment = { vertical: "top", wrapText: true, horizontal: "left" };
    if (value instanceof Date) cell.numFmt = "dd/mm/yyyy hh:mm";
    styleHyperlink(cell);
  }
  rowNo++;
  headerRow(info, rowNo++, ["Étape", "Statut · résultats · détail"]);
  for (const s of search.steps) {
    const row = info.getRow(rowNo++);
    row.getCell(1).value = xml(s.name);
    row.getCell(1).font = xfont({ bold: true });
    row.getCell(2).value = xml([STEP_WORD[s.status], stepSummary(s)].filter(Boolean).join(" · "));
    row.getCell(2).alignment = { wrapText: true, vertical: "top" };
    if (s.status === "error") row.getCell(2).font = xfont({ color: { argb: argb("#c0392b") } });
  }

  return Buffer.from(await wb.xlsx.writeBuffer());
}

// ---------------------------------------------------------------------------
// Register of searches

const REGISTER_COLUMNS: { header: string; width: number; fill?: string }[] = [
  { header: "Date", width: 17 },
  { header: "Mode", width: 28 },
  { header: "Origine", width: 9 },
  { header: "Titre", width: 40 },
  { header: "Note", width: 36 },
  { header: "Statut", width: 10 },
  { header: "Résultats", width: 10 },
  { header: "Très proche", width: 11, fill: LEVEL_HEX.tres_proche },
  { header: "Proche", width: 9, fill: LEVEL_HEX.proche },
  { header: "Même thème", width: 12, fill: LEVEL_HEX.meme_theme },
  { header: "Écartés", width: 9, fill: LEVEL_HEX.ecarte },
  { header: "Fiche", width: 15 },
  { header: "Rapport PDF", width: 13 },
  { header: "Excel", width: 13 },
];

function fileUrl(appUrl: string, path: string | null): string | null {
  if (!path) return null;
  return `${appUrl}/api/files/${path.replace(/^\/+/, "").split("/").map(encodeURIComponent).join("/")}`;
}

// One line per search, for the client's file ("registre des recherches").
export async function buildSearchRegister(searches: VisualSearch[], ctx: { clientName: string; appUrl: string }): Promise<Buffer> {
  const appUrl = ctx.appUrl.replace(/\/+$/, "");
  const wb = new ExcelJS.Workbook();
  wb.creator = "PI Pilot";
  wb.created = new Date();
  const ws = wb.addWorksheet("Registre");
  ws.columns = REGISTER_COLUMNS.map((c) => ({ width: c.width }));
  const sorted = [...searches].sort((x, y) => (validDate(y.createdAt)?.getTime() ?? 0) - (validDate(x.createdAt)?.getTime() ?? 0));
  titleRows(ws, REGISTER_COLUMNS.length, `Registre des recherches visuelles — ${ctx.clientName}`, `${plural(sorted.length, "recherche")} · extrait le ${frenchDate(new Date().toISOString())}`);
  const HEADER = 3;
  headerRow(ws, HEADER, REGISTER_COLUMNS.map((c) => c.header), REGISTER_COLUMNS.map((c) => c.fill ?? null));

  sorted.forEach((s, i) => {
    const counts = levelCounts(s.results);
    const fiche = `${appUrl}/recherche-visuelle?id=${encodeURIComponent(s.id)}`;
    const row = ws.getRow(HEADER + 1 + i);
    row.values = [
      excelDate(s.createdAt),
      MODE_LABEL[s.mode],
      ORIGIN_WORD[s.origin],
      xml(s.analysis?.titre) || null,
      xml(s.note) || null,
      STATUS_WORD[s.status],
      s.results.length,
      counts.tres_proche,
      counts.proche,
      counts.meme_theme,
      counts.ecarte,
      hyperlink("Ouvrir la fiche", fiche),
      s.reportPdfPath ? hyperlink("Rapport PDF", fileUrl(appUrl, s.reportPdfPath)) : null,
      s.reportXlsxPath ? hyperlink("Fichier Excel", fileUrl(appUrl, s.reportXlsxPath)) : null,
    ];
    row.eachCell({ includeEmpty: true }, (cell, col) => {
      cell.alignment = { vertical: "top", wrapText: col === 4 || col === 5, horizontal: col >= 7 && col <= 11 ? "center" : undefined };
      cell.border = thinBottom;
    });
    row.getCell(1).numFmt = "dd/mm/yyyy hh:mm";
    if (s.status === "error") row.getCell(6).font = xfont({ color: { argb: argb("#c0392b") } });
    if (counts.tres_proche) row.getCell(8).font = xfont({ bold: true, color: { argb: argb(LEVEL_HEX.tres_proche) } });
    for (const col of [12, 13, 14]) styleHyperlink(row.getCell(col));
  });
  ws.views = [{ state: "frozen", ySplit: HEADER }];
  printSetup(ws, HEADER);
  ws.autoFilter = { from: { row: HEADER, column: 1 }, to: { row: HEADER + Math.max(sorted.length, 1), column: REGISTER_COLUMNS.length } };
  return Buffer.from(await wb.xlsx.writeBuffer());
}
