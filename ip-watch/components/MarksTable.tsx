"use client";

import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";
import { useApp } from "./AppProvider";
import { AddTitleForm } from "./AddTitleForm";
import { ClientActions } from "./ClientActions";
import { EditableCell } from "./EditableCell";
import { MarkDrawer } from "./MarkDrawer";
import { deadlineFor, DEADLINE_LABELS, isActiveRight } from "@/lib/deadlines";
import { formatDate, toIsoDate } from "@/lib/format";
import { OFFICE_LABELS_SHORT, TYPE_LABELS } from "@/lib/labels";
import { countryName, rightCountries } from "@/lib/territories";
import type { Mark, Office, Right, RightType } from "@/lib/types";

const WATCHABLE = new Set<RightType>(["marque", "dessin"]);

// "Marque enregistrée" → "enregistrée", "Registered" → "enregistrée"…
function shortStatus(s: string): string {
  const t = s.replace(/^marque\s+/i, "").trim();
  const map: Record<string, string> = { registered: "enregistrée", expired: "expirée", renewed: "renouvelée", "application filed": "déposée", withdrawn: "retirée", refused: "refusée" };
  return map[t.toLowerCase()] ?? t.toLowerCase();
}

function compactClasses(c: number[]): string {
  if (c.length === 0) return "—";
  return c.length <= 3 ? c.join(", ") : `${c.slice(0, 3).join(", ")} +${c.length - 3}`;
}

type Scope = "client" | "groupe" | "tous";

export function MarksTable({ marks: initial }: { marks: Mark[] }) {
  const router = useRouter();
  const { role, client } = useApp();
  const [marks, setMarks] = useState(initial);
  const [query, setQuery] = useState("");
  const [type, setType] = useState<RightType | "">("");
  const [office, setOffice] = useState<Office | "">("");
  const [scope, setScope] = useState<Scope>("client");
  const [onlyActive, setOnlyActive] = useState(true);
  const [onlyFav, setOnlyFav] = useState(false);
  const [open, setOpen] = useState<Mark | null>(null);
  const [adding, setAdding] = useState(false);
  const [starting, setStarting] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const groupCount = useMemo(() => marks.filter((m) => m.clientOwned === false).length, [marks]);
  const typesPresent = useMemo(() => Array.from(new Set(marks.map((m) => m.type))), [marks]);

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    return marks
      .filter((m) => (scope === "tous" ? true : scope === "client" ? m.clientOwned !== false : m.clientOwned === false))
      .filter((m) => (!onlyActive || isActiveRight(m)) && (!onlyFav || m.favorite))
      .filter((m) => (!type || m.type === type) && (!office || m.office === office))
      .filter((m) => !q || `${m.name} ${m.number} ${m.id} ${m.owner} ${m.classes.join(" ")}`.toLowerCase().includes(q))
      .sort((a, b) => Number(b.favorite) - Number(a.favorite) || a.name.localeCompare(b.name));
  }, [marks, query, type, office, scope, onlyActive, onlyFav]);

  const patch = (id: string, update: Partial<Mark>) => {
    setMarks((all) => all.map((x) => (x.id === id ? { ...x, ...update } : x)));
    setOpen((o) => (o && o.id === id ? { ...o, ...update } : o));
  };

  const save = async (m: Mark, field: keyof Right, value: unknown) => {
    const res = await fetch(`/api/marks/${encodeURIComponent(m.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ field, value }) });
    if (!res.ok) return false;
    patch(m.id, { [field]: value, edits: { ...m.edits, [field]: { value, prev: m.edits[field]?.prev ?? m[field], by: role, at: new Date().toISOString() } } } as Partial<Mark>);
    return true;
  };

  const favorite = async (m: Mark, value: boolean) => {
    patch(m.id, { favorite: value });
    await fetch(`/api/marks/${encodeURIComponent(m.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ field: "favorite", value }) });
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

  const chip = (on: boolean, label: string, onClick: () => void) => (
    <button type="button" onClick={onClick} className={on ? "chip-ink" : "chip"}>
      {label}
    </button>
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="hand text-[28px] font-semibold leading-tight">Marques</h1>
          <p className="text-[13px] text-[color:var(--pencil)]">{filtered.length} titre(s). Cliquez un nom pour la fiche ; une cellule pour la corriger.</p>
        </div>
        <ClientActions />
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {chip(onlyActive, "en vigueur", () => setOnlyActive(true))}
        {chip(!onlyActive, "tous, y compris expirés", () => setOnlyActive(false))}
        <span className="mx-1 text-[color:var(--line)]">|</span>
        {chip(onlyFav, "★ favoris", () => setOnlyFav((v) => !v))}
        <span className="mx-1 text-[color:var(--line)]">|</span>
        {chip(scope === "client", `déposés par ${client?.company?.legalName ?? client?.name ?? "le client"}`, () => setScope("client"))}
        {groupCount > 0 && chip(scope === "groupe", `sociétés du groupe (${groupCount})`, () => setScope("groupe"))}
        {groupCount > 0 && chip(scope === "tous", "tous titulaires", () => setScope("tous"))}
        <span className="ml-auto flex flex-wrap gap-2">
          <select value={type} onChange={(e) => setType(e.target.value as RightType | "")} className="field py-1 text-[13px]">
            <option value="">tous types</option>
            {typesPresent.map((t) => (
              <option key={t} value={t}>
                {TYPE_LABELS[t]}
              </option>
            ))}
          </select>
          <select value={office} onChange={(e) => setOffice(e.target.value as Office | "")} className="field py-1 text-[13px]">
            <option value="">tous registres</option>
            {(Object.keys(OFFICE_LABELS_SHORT) as Office[]).map((o) => (
              <option key={o} value={o}>
                {OFFICE_LABELS_SHORT[o]}
              </option>
            ))}
          </select>
          <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="rechercher…" className="field w-44 py-1 text-[13px]" />
          {role === "avocat" && (
            <button type="button" className="btn" onClick={() => setAdding(true)}>
              + ajouter un titre
            </button>
          )}
        </span>
      </div>

      {adding && (
        <AddTitleForm
          owner={client?.company?.legalName ?? client?.name ?? ""}
          onCancel={() => setAdding(false)}
          onAdded={(m) => {
            setMarks((all) => [m, ...all]);
            setAdding(false);
            setOpen(m);
          }}
        />
      )}
      {error && <p className="hand text-[13px] text-red-700">{error}</p>}

      <div className="overflow-x-auto">
        <table className="table-base min-w-[1000px]">
          <thead>
            <tr>
              <th className="w-8" />
              <th>Titre</th>
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
              return (
                <tr key={m.id} className="hover:bg-white/50">
                  <td>
                    <button type="button" onClick={() => favorite(m, !m.favorite)} className="hand text-[16px] leading-none" title={m.favorite ? "Retirer des favoris" : "Ajouter aux favoris"}>
                      {m.favorite ? "★" : "☆"}
                    </button>
                  </td>
                  <td className="min-w-[220px]">
                    <button type="button" onClick={() => setOpen(m)} className="hand text-left text-[14px] hover:underline">
                      {m.name === "(marque figurative)" ? <span className="italic text-[color:var(--pencil)]">logo (marque figurative)</span> : m.name}
                    </button>
                    <div className="flex items-center gap-1.5 font-mono text-[10.5px] text-[color:var(--pencil)]">
                      <span className="rounded border border-[color:var(--line)] px-1">{OFFICE_LABELS_SHORT[m.office]}</span>
                      {m.id}
                      {m.clientOwned === false && <span>· groupe</span>}
                      {m.origin === "manuel" && <span>· manuel</span>}
                    </div>
                  </td>
                  <td className="whitespace-nowrap text-[12.5px]">{TYPE_LABELS[m.type]}</td>
                  <td className="w-[110px] text-[12.5px]" title={m.classes.join(", ")}>
                    <EditableCell
                      display={compactClasses(m.classes)}
                      initial={m.classes.join(", ")}
                      edit={m.edits.classes}
                      onSave={(v) => {
                        const parts = v.split(/[,\s]+/).filter(Boolean).map(Number);
                        if (parts.some((n) => !Number.isInteger(n) || n < 1 || n > 45)) return false;
                        return save(m, "classes", parts);
                      }}
                    />
                  </td>
                  <td className="whitespace-nowrap text-[12.5px]">{dateCell(m, "filingDate")}</td>
                  <td className="w-[140px] text-[12.5px]">
                    {dateCell(m, "expiryDate")}
                    {d.state !== "a_venir" && d.state !== "inconnu" && <span className="chip-ink mt-1 w-fit whitespace-nowrap">{DEADLINE_LABELS[d.state]}</span>}
                  </td>
                  <td className="w-[110px] text-[12.5px]">
                    <EditableCell display={m.status ? shortStatus(m.status) : "—"} initial={m.status} edit={m.edits.status} onSave={(v) => save(m, "status", v.trim())} />
                  </td>
                  <td className="text-[12.5px]" title={countries.map(countryName).join(", ") || (m.office === "WO" ? "pays désignés non chargés" : "")}>
                    {countries.length || (m.office === "WO" ? "?" : "—")}
                  </td>
                  <td className="whitespace-nowrap text-[12px]">
                    {m.lastRun ? (
                      <>
                        <span className="hand">{formatDate(m.lastRun.startedAt)}</span>
                        <div className="text-[color:var(--pencil)]">{m.lastRun.high} fort(s)</div>
                      </>
                    ) : (
                      <span className="text-[color:var(--pencil)]">—</span>
                    )}
                  </td>
                  <td className="whitespace-nowrap text-right">
                    {WATCHABLE.has(m.type) && (
                      <button type="button" className="btn py-0.5 text-[12px]" disabled={starting !== null} onClick={() => watch(m)}>
                        {starting === m.id ? "lancement…" : "rechercher"}
                      </button>
                    )}
                  </td>
                </tr>
              );
            })}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={10} className="hand py-6 text-center text-[color:var(--pencil)]">
                  Aucun titre avec ces filtres.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {open && (
        <MarkDrawer
          mark={open}
          onClose={() => setOpen(null)}
          onFavorite={favorite}
          onDeleted={(id) => {
            setMarks((all) => all.filter((x) => x.id !== id));
            setOpen(null);
          }}
        />
      )}
    </div>
  );
}
