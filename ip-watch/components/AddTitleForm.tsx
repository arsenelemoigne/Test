"use client";

import { useMemo, useState } from "react";
import { deadlineFor } from "@/lib/deadlines";
import { formatDate } from "@/lib/format";
import { OFFICE_LABELS_SHORT, TYPE_HINTS, TYPE_LABELS } from "@/lib/labels";
import type { Mark, Office, Right, RightType } from "@/lib/types";

const TYPES = Object.keys(TYPE_LABELS) as RightType[];
const OFFICES = Object.keys(OFFICE_LABELS_SHORT) as Office[];
const DEFAULT_OFFICE: Partial<Record<RightType, Office>> = { brevet: "FR", certificat_utilite: "FR", domaine: "AUTRE", droit_auteur: "FR", cov: "AUTRE", ig: "FR" };

// Adds a title the registry import does not cover (patent, domain name, proof of creation…).
export function AddTitleForm({ owner, onAdded, onCancel }: { owner: string; onAdded: (m: Mark) => void; onCancel: () => void }) {
  const [f, setF] = useState({ type: "brevet" as RightType, name: "", number: "", office: "FR" as Office, owner, filingDate: "", expiryDate: "", countries: "", classes: "", sourceUrl: "", notes: "" });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const set = (k: keyof typeof f, v: string) => setF((x) => ({ ...x, [k]: v }));

  // Suggests the expiry from the usual term of the chosen type.
  const suggested = useMemo(() => {
    if (!f.filingDate || f.expiryDate) return null;
    const probe = { type: f.type, office: f.office, filingDate: f.filingDate, registrationDate: null, expiryDate: null, territories: [], classes: [], id: "", number: "", name: "", owner: "", status: "" } as Right;
    return deadlineFor(probe).expiry;
  }, [f.type, f.office, f.filingDate, f.expiryDate]);

  const submit = async () => {
    setBusy(true);
    setError(null);
    const res = await fetch("/api/marks", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...f, expiryDate: f.expiryDate || (suggested ? suggested.toISOString().slice(0, 10) : "") }),
    });
    const json = await res.json().catch(() => ({}));
    setBusy(false);
    if (!res.ok) return setError(json.error ?? "Ajout impossible.");
    onAdded({ ...json.mark, edits: {}, lastRun: null, favorite: false });
  };

  const field = (label: string, el: React.ReactNode) => (
    <label className="block">
      <span className="label">{label}</span>
      <div className="mt-0.5">{el}</div>
    </label>
  );

  return (
    <div className="wobble space-y-3 bg-white/40 p-4">
      <div className="flex items-baseline justify-between">
        <h2 className="hand text-[17px]">Ajouter un titre</h2>
        <button type="button" className="btn-quiet text-[12px]" onClick={onCancel}>
          annuler
        </button>
      </div>
      <div className="grid gap-3 sm:grid-cols-3">
        {field(
          "type",
          <select
            className="field w-full"
            value={f.type}
            onChange={(e) => {
              const type = e.target.value as RightType;
              setF((x) => ({ ...x, type, office: DEFAULT_OFFICE[type] ?? x.office }));
            }}
          >
            {TYPES.map((t) => (
              <option key={t} value={t}>
                {TYPE_LABELS[t]}
              </option>
            ))}
          </select>,
        )}
        <div className="sm:col-span-2">{field("nom ou titre", <input className="field w-full" value={f.name} onChange={(e) => set("name", e.target.value)} placeholder="ex. Fermoir à double articulation" />)}</div>
      </div>
      <p className="text-[12px] text-[color:var(--pencil)]">Durée habituelle : {TYPE_HINTS[f.type]}.</p>
      <div className="grid gap-3 sm:grid-cols-3">
        {field("numéro", <input className="field w-full" value={f.number} onChange={(e) => set("number", e.target.value)} placeholder="ex. FR3101234" />)}
        {field(
          "registre",
          <select className="field w-full" value={f.office} onChange={(e) => set("office", e.target.value)}>
            {OFFICES.map((o) => (
              <option key={o} value={o}>
                {OFFICE_LABELS_SHORT[o]}
              </option>
            ))}
          </select>,
        )}
        {field("titulaire", <input className="field w-full" value={f.owner} onChange={(e) => set("owner", e.target.value)} />)}
        {field("date de dépôt", <input type="date" className="field w-full" value={f.filingDate} onChange={(e) => set("filingDate", e.target.value)} />)}
        {field(
          "date d'expiration",
          <>
            <input type="date" className="field w-full" value={f.expiryDate} onChange={(e) => set("expiryDate", e.target.value)} />
            {suggested && <span className="text-[11px] text-[color:var(--pencil)]">par défaut : {formatDate(suggested)}</span>}
          </>,
        )}
        {field("pays (codes)", <input className="field w-full" value={f.countries} onChange={(e) => set("countries", e.target.value)} placeholder="FR, DE, US" />)}
        {field("classes", <input className="field w-full" value={f.classes} onChange={(e) => set("classes", e.target.value)} placeholder="14" />)}
        <div className="sm:col-span-2">{field("lien (registre, Espacenet…)", <input className="field w-full" value={f.sourceUrl} onChange={(e) => set("sourceUrl", e.target.value)} placeholder="https://" />)}</div>
      </div>
      {field("notes", <textarea className="field w-full" rows={2} value={f.notes} onChange={(e) => set("notes", e.target.value)} />)}
      {error && <p className="hand text-[13px] text-red-700">{error}</p>}
      <div className="flex justify-end">
        <button type="button" className="btn-ink" disabled={busy || !f.name.trim()} onClick={submit}>
          {busy ? "ajout…" : "ajouter"}
        </button>
      </div>
    </div>
  );
}
