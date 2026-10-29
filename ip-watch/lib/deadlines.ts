import { REMINDER_OFFSETS_DAYS } from "./config";
import type { Right } from "./types";

const DAY = 24 * 60 * 60 * 1000;

function addYears(iso: string, years: number): Date {
  const d = new Date(iso);
  d.setUTCFullYear(d.getUTCFullYear() + years);
  return d;
}

function addMonths(d: Date, months: number): Date {
  const out = new Date(d);
  out.setUTCMonth(out.getUTCMonth() + months);
  return out;
}

export type DeadlineState = "a_venir" | "fenetre_ouverte" | "delai_de_grace" | "expire" | "inconnu";

export type Deadline = {
  rightId: string;
  expiry: Date | null;
  // True when the expiry date was computed from the filing date rather than read from the registry.
  computed: boolean;
  windowOpens: Date | null;
  graceEnds: Date | null;
  reminders: Date[];
  state: DeadlineState;
  daysLeft: number | null;
};

// Trademarks run for 10-year terms; designs for 5-year terms capped at 25 years.
// Renewal opens 6 months before expiry, with a 6-month grace period after it (FR, EU and WO).
function computedExpiry(right: Right, today: Date): Date | null {
  const start = right.office === "WO" ? right.registrationDate ?? right.filingDate : right.filingDate;
  if (!start) return null;
  const term = right.type === "marque" ? 10 : 5;
  const max = right.type === "marque" ? Infinity : 25;
  for (let years = term; years <= max; years += term) {
    const end = addYears(start, years);
    if (addMonths(end, 6) >= today) return end;
    if (years > 200) break;
  }
  return addYears(start, right.type === "marque" ? term : 25);
}

export function deadlineFor(right: Right, today = new Date()): Deadline {
  const fromRegistry = right.expiryDate ? new Date(right.expiryDate) : null;
  const expiry = fromRegistry ?? computedExpiry(right, today);
  if (!expiry) {
    return { rightId: right.id, expiry: null, computed: false, windowOpens: null, graceEnds: null, reminders: [], state: "inconnu", daysLeft: null };
  }
  const windowOpens = addMonths(expiry, -6);
  const graceEnds = addMonths(expiry, 6);
  const reminders = REMINDER_OFFSETS_DAYS.map((d) => new Date(expiry.getTime() - d * DAY));
  const daysLeft = Math.ceil((expiry.getTime() - today.getTime()) / DAY);
  let state: DeadlineState = "a_venir";
  if (today > graceEnds) state = "expire";
  else if (today > expiry) state = "delai_de_grace";
  else if (today >= windowOpens) state = "fenetre_ouverte";
  return { rightId: right.id, expiry, computed: !fromRegistry, windowOpens, graceEnds, reminders, state, daysLeft };
}

const INACTIVE_STATUS = /expir|retir|radi|rejet|refus|abandon|non renouvel|annul|caduc|withdrawn|expired|surrendered|cancel|lapsed/i;

// In force: not marked as ended by the registry and not past its grace period.
export function isActiveRight(right: Right, today = new Date()): boolean {
  if (INACTIVE_STATUS.test(right.status ?? "")) return false;
  return deadlineFor(right, today).state !== "expire";
}

export function daysLabel(d: Deadline): string {
  if (d.daysLeft === null) return "";
  return d.daysLeft >= 0 ? `J-${d.daysLeft}` : `échu depuis ${-d.daysLeft} j`;
}

export const DEADLINE_LABELS: Record<DeadlineState, string> = {
  a_venir: "À venir",
  fenetre_ouverte: "Renouvellement ouvert",
  delai_de_grace: "Délai de grâce",
  expire: "Expiré",
  inconnu: "Date inconnue",
};
