import countries from "i18n-iso-countries";
import fr from "i18n-iso-countries/langs/fr.json";
import type { Office, Right, Territory } from "./types";

countries.registerLocale(fr);

export const EU_MEMBERS = [
  "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IE",
  "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
];

export const OFFICE_LABELS: Record<Office, string> = {
  FR: "INPI (France)",
  EU: "EUIPO (Union européenne)",
  WO: "OMPI (international)",
};

export function countryName(code: string): string {
  if (code === "EU") return "Union européenne";
  return countries.getName(code, "fr") ?? code;
}

export function numericToAlpha2(id: string | number | undefined): string | undefined {
  if (id === undefined) return undefined;
  return countries.numericToAlpha2(String(id).padStart(3, "0"));
}

const LAPSED = /non renouvel|refus total|radi|retir|invalid|annul|expir|cess/i;

export function isActiveTerritory(t: Territory): boolean {
  return !LAPSED.test(t.status ?? "");
}

// Countries where a right produces effect, with the EU expanded to its member states.
export function rightCountries(right: Right): string[] {
  const out = new Set<string>();
  for (const t of right.territories.filter(isActiveTerritory)) {
    if (t.code === "EU") EU_MEMBERS.forEach((c) => out.add(c));
    else out.add(t.code);
  }
  return Array.from(out);
}

export function countByCountry(rights: Right[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const r of rights) {
    for (const c of rightCountries(r)) counts.set(c, (counts.get(c) ?? 0) + 1);
  }
  return counts;
}
