"use client";

import Link from "next/link";
import { useMemo, useState } from "react";
import { useData } from "@/components/EditsProvider";
import { EditableCell } from "@/components/EditableCell";
import { formatDate, toIsoDate } from "@/lib/format";
import { countryName, OFFICE_LABELS, rightCountries } from "@/lib/territories";
import { TYPE_LABELS } from "@/lib/labels";
import type { Office, RightType } from "@/lib/types";

export default function PortfolioPage() {
  const { rights, rightEdits, editRight, history, resetEdits, role } = useData();
  const [query, setQuery] = useState("");
  const [type, setType] = useState<RightType | "">("");
  const [office, setOffice] = useState<Office | "">("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return rights.filter(
      (r) =>
        (!type || r.type === type) &&
        (!office || r.office === office) &&
        (!q || `${r.name} ${r.number} ${r.owner} ${r.classes.join(" ")}`.toLowerCase().includes(q)),
    );
  }, [rights, query, type, office]);

  const dateCell = (id: string, field: "filingDate" | "registrationDate" | "expiryDate", value: string | null) => (
    <EditableCell
      display={formatDate(value)}
      initial={value ? formatDate(value) : ""}
      placeholder="jj/mm/aaaa"
      edit={rightEdits[id]?.[field]}
      className="font-mono tabular-nums"
      onSave={(raw) => {
        if (!raw.trim()) return editRight(id, field, null);
        const iso = toIsoDate(raw);
        if (!iso) return false;
        editRight(id, field, iso);
      }}
    />
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Portefeuille</h1>
          <p className="text-sm text-neutral-600">
            Chaque cellule est modifiable ({role === "avocat" ? "revue de l'avocat" : "correction du client"}). Un point noir signale une valeur modifiée ; survolez pour voir l&apos;historique.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          {role === "avocat" && (
            <Link href="/import" className="btn">
              Importer depuis l&apos;INPI
            </Link>
          )}
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Rechercher…" className="border border-black px-2 py-1.5 font-mono text-xs" />
          <select value={type} onChange={(e) => setType(e.target.value as RightType | "")} className="border border-black bg-white px-2 py-1.5 font-mono text-xs">
            <option value="">Tous types</option>
            <option value="marque">Marques</option>
            <option value="dessin">Dessins & modèles</option>
          </select>
          <select value={office} onChange={(e) => setOffice(e.target.value as Office | "")} className="border border-black bg-white px-2 py-1.5 font-mono text-xs">
            <option value="">Tous offices</option>
            <option value="FR">INPI</option>
            <option value="EU">EUIPO</option>
            <option value="WO">OMPI</option>
          </select>
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="table-base min-w-[1100px]">
          <thead>
            <tr>
              <th>Type</th>
              <th>Office</th>
              <th>Numéro</th>
              <th>Nom</th>
              <th>Titulaire</th>
              <th>Classes</th>
              <th>Dépôt</th>
              <th>Enregistrement</th>
              <th>Expiration</th>
              <th>Statut</th>
              <th>Pays</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((r) => {
              const countries = rightCountries(r);
              return (
                <tr key={r.id}>
                  <td className="whitespace-nowrap">{TYPE_LABELS[r.type]}</td>
                  <td className="whitespace-nowrap font-mono text-xs">{r.office}</td>
                  <td className="whitespace-nowrap font-mono text-xs">
                    {r.sourceUrl ? (
                      <a href={r.sourceUrl} target="_blank" rel="noreferrer" className="underline">
                        {r.number}
                      </a>
                    ) : (
                      r.number
                    )}
                  </td>
                  <td className="min-w-[180px]">
                    <EditableCell display={r.name} initial={r.name} edit={rightEdits[r.id]?.name} onSave={(v) => editRight(r.id, "name", v.trim())} />
                  </td>
                  <td className="min-w-[160px]">
                    <EditableCell display={r.owner} initial={r.owner} edit={rightEdits[r.id]?.owner} onSave={(v) => editRight(r.id, "owner", v.trim())} />
                  </td>
                  <td>
                    <EditableCell
                      display={r.classes.join(", ") || "—"}
                      initial={r.classes.join(", ")}
                      edit={rightEdits[r.id]?.classes}
                      className="font-mono"
                      onSave={(v) => {
                        const parts = v.split(/[,\s]+/).filter(Boolean).map(Number);
                        if (parts.some((n) => !Number.isInteger(n) || n < 1 || n > 45)) return false;
                        editRight(r.id, "classes", parts);
                      }}
                    />
                  </td>
                  <td className="whitespace-nowrap">{dateCell(r.id, "filingDate", r.filingDate)}</td>
                  <td className="whitespace-nowrap">{dateCell(r.id, "registrationDate", r.registrationDate)}</td>
                  <td className="whitespace-nowrap">{dateCell(r.id, "expiryDate", r.expiryDate)}</td>
                  <td>
                    <EditableCell display={r.status} initial={r.status} edit={rightEdits[r.id]?.status} onSave={(v) => editRight(r.id, "status", v.trim())} />
                  </td>
                  <td className="max-w-[260px] text-xs" title={countries.map(countryName).join(", ")}>
                    {r.territories.map((t) => (t.code === "EU" ? "UE (27)" : t.code)).join(" · ")}
                    <span className="text-neutral-500"> — {countries.length} pays</span>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      <p className="font-mono text-[11px] text-neutral-500">
        {filtered.length} titre(s) affiché(s) sur {rights.length}. Offices : {Object.values(OFFICE_LABELS).join(" · ")}.
      </p>

      <section className="pt-4">
        <div className="flex items-baseline justify-between">
          <h2 className="label">Historique des modifications</h2>
          {history.length > 0 && (
            <button type="button" className="font-mono text-xs underline" onClick={resetEdits}>
              Annuler toutes les modifications
            </button>
          )}
        </div>
        {history.length === 0 ? (
          <p className="mt-2 text-sm text-neutral-500">Aucune modification. Les modifications sont enregistrées dans ce navigateur pendant la démo.</p>
        ) : (
          <ul className="mt-2 divide-y divide-neutral-200 border-y border-black font-mono text-xs">
            {history.slice(0, 20).map((h, i) => (
              <li key={i} className="flex flex-wrap gap-x-3 py-1.5">
                <span className="text-neutral-500">{new Date(h.at).toLocaleString("fr-FR")}</span>
                <span>{h.by === "avocat" ? "Avocat" : "Client"}</span>
                <span>{h.id}</span>
                <span>
                  {h.field} : {String(h.prev ?? "—")} → {String(h.value ?? "—")}
                </span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
