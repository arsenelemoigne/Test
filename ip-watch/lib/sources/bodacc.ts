// BODACC legal announcements (registrations, sales, insolvency proceedings), no key required.
const BASE = "https://bodacc-datadila.opendatasoft.com/api/explore/v2.1/catalog/datasets/annonces-commerciales/records";

type Record_ = Record<string, unknown>;

export async function searchAnnouncements(q: string) {
  const isSiren = /^\d{9}$/.test(q.replace(/\s/g, ""));
  const where = isSiren ? `registre like "${q.replace(/\s/g, "")}"` : `"${q.replace(/"/g, "")}"`;
  const res = await fetch(`${BASE}?${new URLSearchParams({ where, order_by: "dateparution desc", limit: "6" })}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`BODACC ${res.status}`);
  const json = (await res.json()) as { results?: Record_[] };
  return (json.results ?? []).map((r) => ({
    date: String(r.dateparution ?? ""),
    type: String(r.familleavis_lib ?? r.typeavis_lib ?? ""),
    company: String(r.commercant ?? ""),
    registry: String(r.registre ?? ""),
    court: String(r.tribunal ?? ""),
    url: typeof r.url_complete === "string" ? r.url_complete : `https://www.bodacc.fr/pages/annonces-commerciales-detail/?q.id=id:${r.id}`,
  }));
}
