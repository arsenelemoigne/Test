import { piste } from "./piste";

const JL = "/cassation/judilibre/v1.0";

export const decisionUrl = (id: string) => `https://www.courdecassation.fr/decision/${id}`;

type Hit = Record<string, unknown> & { id: string };

function label(v: unknown): string {
  if (typeof v === "string") return v;
  if (v && typeof v === "object" && "label" in v) return String((v as { label: unknown }).label);
  return "";
}

function summarize(d: Hit) {
  const highlights = (d.highlights as { text?: string[] } | undefined)?.text ?? [];
  return {
    id: d.id,
    jurisdiction: label(d.jurisdiction),
    chamber: label(d.chamber),
    number: Array.isArray(d.numbers) ? (d.numbers as string[]).join(", ") : label(d.number),
    date: label(d.decision_date),
    solution: label(d.solution),
    summary: label(d.summary).slice(0, 600),
    highlight: highlights.join(" … ").replace(/<\/?em>/g, "").slice(0, 400),
    url: decisionUrl(d.id),
  };
}

export async function searchCaseLaw(query: string, jurisdiction: "cc" | "ca" | "tous" = "tous") {
  const params = new URLSearchParams({ query, page_size: "8", resolve_references: "true", operator: "and" });
  if (jurisdiction === "tous") {
    params.append("jurisdiction", "cc");
    params.append("jurisdiction", "ca");
  } else params.append("jurisdiction", jurisdiction);
  const data = await piste<{ results?: Hit[]; total?: number }>("GET", `${JL}/search?${params}`);
  return { total: data.total ?? null, results: (data.results ?? []).map(summarize) };
}

export async function getDecision(id: string) {
  const d = await piste<Hit & { text?: string; zones?: Record<string, { start: number; end: number }[]> }>("GET", `${JL}/decision?${new URLSearchParams({ id, resolve_references: "true" })}`);
  const text = d.text ?? "";
  // Prefer the court's reasoning; fall back to the start of the text.
  const zone = d.zones?.motivations?.[0];
  const excerpt = zone ? text.slice(zone.start, Math.min(zone.end, zone.start + 2500)) : text.slice(0, 2500);
  const dispositif = d.zones?.dispositif?.[0];
  return { ...summarize(d), excerpt: excerpt.trim(), dispositif: dispositif ? text.slice(dispositif.start, Math.min(dispositif.end, dispositif.start + 800)).trim() : null };
}
