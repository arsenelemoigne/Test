import type { Company } from "../types";

// Official French company directory (no key required).
const BASE = "https://recherche-entreprises.api.gouv.fr";

type Result = {
  siren: string;
  nom_complet?: string;
  nom_raison_sociale?: string;
  sigle?: string | null;
  date_creation?: string;
  activite_principale?: string;
  categorie_entreprise?: string;
  siege?: { adresse?: string; geo_adresse?: string; libelle_commune?: string };
  dirigeants?: { nom?: string; prenoms?: string; denomination?: string; qualite?: string }[];
};

export const companyUrl = (siren: string) => `https://annuaire-entreprises.data.gouv.fr/entreprise/${siren}`;

function toCompany(r: Result): Company {
  return {
    siren: r.siren,
    name: r.nom_complet ?? r.nom_raison_sociale ?? r.siren,
    legalName: r.nom_raison_sociale ?? undefined,
    address: r.siege?.adresse ?? r.siege?.geo_adresse,
    createdOn: r.date_creation,
    activity: r.activite_principale,
    category: r.categorie_entreprise,
    officers: (r.dirigeants ?? [])
      .slice(0, 5)
      .map((d) => [d.denomination ?? [d.prenoms, d.nom].filter(Boolean).join(" "), d.qualite].filter(Boolean).join(" — ")),
    url: companyUrl(r.siren),
  };
}

export async function searchCompanies(q: string, limit = 5): Promise<Company[]> {
  const res = await fetch(`${BASE}/search?${new URLSearchParams({ q, per_page: String(limit), page: "1" })}`, { headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`Recherche d'entreprises ${res.status}`);
  const json = (await res.json()) as { results?: Result[] };
  return (json.results ?? []).map(toCompany);
}
