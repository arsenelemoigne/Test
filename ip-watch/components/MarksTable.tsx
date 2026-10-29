"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { EditableCell } from "./EditableCell";
import { deadlineFor, DEADLINE_LABELS } from "@/lib/deadlines";
import { formatDate, toIsoDate } from "@/lib/format";
import { TYPE_LABELS } from "@/lib/labels";
import { countryName, rightCountries } from "@/lib/territories";
import type { Mark, Office, Right, RightType } from "@/lib/types";

export function MarksTable({ marks: initial }: { marks: Mark[] }) {
  const router = useRouter();
  const [marks, setMarks] = useState(initial);
  const [query, setQuery] = useState("");
  const [type, setType] = useState<RightType | "">("");
  const [office, setOffice] = useState<Office | "">("");
  const [starting, setStarting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return marks
      .filter((m) => (!type || m.type === type) && (!office || m.office === office) && (!q || `${m.name} ${m.number} ${m.owner} ${m.classes.join(" ")}`.toLowerCase().includes(q)))
      .sort((a, b) => a.name.localeCompare(b.name));
  }, [marks, query, type, office]);

  const save = async (m: Mark, field: keyof Right, value: unknown) => {
    const res = await fetch(`/api/marks/${encodeURIComponent(m.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ field, value }) });
    if (!res.ok) return false;
    setMarks((all) => all.map((x) => (x.id === m.id ? { ...x, [field]: value, edits: { ...x.edits, [field]: { value, prev: x.edits[field]?.prev ?? x[field], by: "avocat", at: new Date().toISOString() } } } : x)));
    return true;
  };

  const watch = async (m: Mark) => {
    setStarting(m.id);
    setError(null);
    const res = await fetch(`/api/marks/${encodeURIComponent(m.id)}/watch`, { method: "POST" });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      setStarting(null);
      setError(json.error ?? "Impossible de lancer la recherche.");
      return;
    }
    router.push(`/?run=${json.runId}`);
  };

  const dateCell = (m: Mark, field: "filingDate" | "expiryDate") => (
    <EditableCell
      display={formatDate(m[field])}
      initial={m[field] ? formatDate(m[field]) : ""}
      placeholder="jj/mm/aaaa"
      edit={m.edits[field]}
      className="tabular-nums"
      onSave={(raw) => {
        if (!raw.trim()) return save(m, field, null);
        const iso = toIsoDate(raw);
        return iso ? save(m, field, iso) : false;
      }}
    />
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="hand text-[28px] font-semibold leading-tight">Marques</h1>
          <p className="text-[13px] text-[color:var(--pencil)]">
            {marks.filter((m) => m.type === "marque").length} marques, {marks.filter((m) => m.type === "dessin").length} dessins & modèles. Cliquez une cellule pour la corriger ; un point signale une valeur modifiée.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="rechercher…" className="field w-48" />
          <select value={type} onChange={(e) => setType(e.target.value as RightType | "")} className="field">
            <option value="">tous types</option>
            <option value="marque">marques</option>
            <option value="dessin">dessins & modèles</option>
          </select>
          <select value={office} onChange={(e) => setOffice(e.target.value as Office | "")} className="field">
            <option value="">tous offices</option>
            <option value="FR">INPI</option>
            <option value="EU">EUIPO</option>
            <option value="WO">OMPI</option>
          </select>
        </div>
      </div>
      {error && <p className="hand text-[13px] text-red-700">{error}</p>}

      <div className="overflow-x-auto">
        <table className="table-base min-w-[1100px]">
          <thead>
            <tr>
              <th>Marque</th>
              <th>Type</th>
              <th>Classes</th>
              <th>Dépôt</th>
              <th>Expiration</th>
              <th>Statut</th>
              <th>Pays</th>
              <th>Veille</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {filtered.map((m) => {
              const d = deadlineFor(m);
              const countries = rightCountries(m);
              const log = `/api/marks/${encodeURIComponent(m.id)}/log`;
              return (
                <tr key={m.id}>
                  <td className="min-w-[200px]">
                    <EditableCell display={m.name} initial={m.name} edit={m.edits.name} className="hand text-[14px]" onSave={(v) => save(m, "name", v.trim())} />
                    <div className="font-mono text-[10.5px] text-[color:var(--pencil)]">
                      {m.sourceUrl ? (
                        <a href={m.sourceUrl} target="_blank" rel="noreferrer" className="hover:underline">
                          {m.id}
                        </a>
                      ) : (
                        m.id
                      )}
                    </div>
                  </td>
                  <td className="whitespace-nowrap text-[12.5px]">
                    {TYPE_LABELS[m.type]}
                    <div className="text-[11px] text-[color:var(--pencil)]">{m.office === "FR" ? "INPI" : m.office === "EU" ? "EUIPO" : "OMPI"}</div>
                  </td>
                  <td className="w-[90px]">
                    <EditableCell
                      display={m.classes.join(", ") || "—"}
                      initial={m.classes.join(", ")}
                      edit={m.edits.classes}
                      onSave={(v) => {
                        const parts = v.split(/[,\s]+/).filter(Boolean).map(Number);
                        if (parts.some((n) => !Number.isInteger(n) || n < 1 || n > 45)) return false;
                        return save(m, "classes", parts);
                      }}
                    />
                  </td>
                  <td className="whitespace-nowrap">{dateCell(m, "filingDate")}</td>
                  <td className="w-[150px]">
                    {dateCell(m, "expiryDate")}
                    {d.state !== "a_venir" && d.state !== "inconnu" && <span className="chip-ink mt-1 w-fit whitespace-nowrap">{DEADLINE_LABELS[d.state]}</span>}
                  </td>
                  <td className="w-[120px] text-[12.5px]">
                    <EditableCell display={m.status || "—"} initial={m.status} edit={m.edits.status} onSave={(v) => save(m, "status", v.trim())} />
                  </td>
                  <td className="text-[12.5px]" title={countries.map(countryName).join(", ")}>
                    {countries.length}
                  </td>
                  <td className="whitespace-nowrap text-[12px]">
                    {m.lastRun ? (
                      <>
                        <span className="hand">{formatDate(m.lastRun.startedAt)}</span>
                        <div className="text-[color:var(--pencil)]">
                          {m.lastRun.total} résultats, {m.lastRun.high} forts
                        </div>
                      </>
                    ) : (
                      <span className="text-[color:var(--pencil)]">—</span>
                    )}
                  </td>
                  <td className="whitespace-nowrap text-right">
                    <button type="button" className="btn" disabled={starting !== null} onClick={() => watch(m)}>
                      {starting === m.id ? "lancement…" : "rechercher"}
                    </button>
                    {m.lastRun && (
                      <div className="mt-1 flex justify-end gap-2">
                        <a href={`${log}?format=xlsx`} className="btn-quiet text-[11.5px]">
                          journal xlsx
                        </a>
                        <a href={`${log}?format=csv`} className="btn-quiet text-[11.5px]">
                          csv
                        </a>
                      </div>
                    )}
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={9} className="hand py-6 text-center text-[color:var(--pencil)]">
                  Aucun titre.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
