"use client";

import { useMemo, useState } from "react";
import { useData } from "@/components/EditsProvider";
import { WorldMap } from "@/components/WorldMap";
import { countByCountry, countryName, rightCountries } from "@/lib/territories";
import { deadlineFor } from "@/lib/deadlines";
import { formatDate } from "@/lib/format";
import { TYPE_LABELS } from "@/lib/labels";

export default function MapPage() {
  const { rights } = useData();
  const [selected, setSelected] = useState<string | null>(null);

  const counts = useMemo(() => countByCountry(rights), [rights]);
  const ranking = useMemo(() => Array.from(counts.entries()).sort((a, b) => b[1] - a[1] || countryName(a[0]).localeCompare(countryName(b[0]))), [counts]);
  const inCountry = useMemo(() => (selected ? rights.filter((r) => rightCountries(r).includes(selected)) : []), [rights, selected]);

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Carte</h1>
        <p className="text-sm text-neutral-600">Pays où au moins un titre produit effet. Cliquez sur un pays pour voir les titres concernés.</p>
      </div>

      <WorldMap counts={counts} selected={selected} onSelect={setSelected} unit="titre(s)" />

      <div className="grid gap-6 lg:grid-cols-3">
        <section className="lg:col-span-2">
          <h2 className="label">{selected ? `${countryName(selected)} — ${inCountry.length} titre(s)` : "Sélectionnez un pays"}</h2>
          {selected && (
            <table className="table-base mt-2">
              <thead>
                <tr>
                  <th>Titre</th>
                  <th>Type</th>
                  <th>Voie</th>
                  <th>Expiration</th>
                  <th>Statut local</th>
                </tr>
              </thead>
              <tbody>
                {inCountry.map((r) => {
                  const local = r.territories.find((t) => t.code === selected)?.status;
                  return (
                    <tr key={r.id}>
                      <td>
                        {r.name}
                        <div className="font-mono text-[11px] text-neutral-500">{r.id}</div>
                      </td>
                      <td>{TYPE_LABELS[r.type]}</td>
                      <td className="font-mono text-xs">{r.office === "EU" ? "UE" : r.office === "WO" ? "Internationale" : "Nationale"}</td>
                      <td className="font-mono text-xs tabular-nums">{formatDate(deadlineFor(r).expiry)}</td>
                      <td className="text-xs">{local ?? r.status}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </section>
        <section>
          <h2 className="label">Pays par nombre de titres</h2>
          <ul className="mt-2 max-h-[420px] divide-y divide-neutral-200 overflow-y-auto border-y border-black">
            {ranking.map(([code, n]) => (
              <li key={code}>
                <button
                  type="button"
                  onClick={() => setSelected(code === selected ? null : code)}
                  className={`flex w-full justify-between px-1 py-1.5 text-left text-sm ${code === selected ? "bg-black text-white" : "hover:bg-neutral-100"}`}
                >
                  <span>{countryName(code)}</span>
                  <span className="font-mono tabular-nums">{n}</span>
                </button>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}
