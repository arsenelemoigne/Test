"use client";

import { useRouter } from "next/navigation";
import { useState } from "react";
import { deadlineFor, DEADLINE_LABELS, daysLabel } from "@/lib/deadlines";
import { formatDate } from "@/lib/format";
import { OFFICE_LABELS_SHORT, TYPE_LABELS } from "@/lib/labels";
import { countryName, rightCountries } from "@/lib/territories";
import type { Mark } from "@/lib/types";

const WATCHABLE = new Set(["marque", "dessin"]);

// Everything about one title, opened from the timeline or the marks list.
export function MarkDrawer({ mark, onClose, onFavorite, onDeleted }: { mark: Mark; onClose: () => void; onFavorite?: (m: Mark, value: boolean) => void; onDeleted?: (id: string) => void }) {
  const router = useRouter();
  const [starting, setStarting] = useState(false);
  const [photo, setPhoto] = useState(mark.imageUrl ?? "");
  const [photoDraft, setPhotoDraft] = useState(mark.imageUrl ?? "");
  const [photoError, setPhotoError] = useState<string | null>(null);

  const savePhoto = async () => {
    setPhotoError(null);
    const value = photoDraft.trim() || null;
    const res = await fetch(`/api/marks/${encodeURIComponent(mark.id)}`, { method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ field: "imageUrl", value }) });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) return setPhotoError(json.error ?? "Enregistrement impossible.");
    setPhoto(value ?? "");
  };
  const d = deadlineFor(mark);
  const countries = rightCountries(mark);
  const log = `/api/marks/${encodeURIComponent(mark.id)}/log`;

  const watch = async () => {
    setStarting(true);
    const res = await fetch(`/api/marks/${encodeURIComponent(mark.id)}/watch`, { method: "POST" });
    const json = await res.json().catch(() => ({}));
    if (res.ok) {
      onClose();
      router.push(`/?run=${json.runId}`);
    } else setStarting(false);
  };

  const row = (label: string, value: React.ReactNode) => (
    <div className="grid grid-cols-[130px_1fr] gap-2 border-b border-dashed border-[color:var(--line)] py-1.5 text-[13px]">
      <span className="label pt-0.5">{label}</span>
      <span>{value}</span>
    </div>
  );

  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/10" onClick={onClose}>
      <aside onClick={(e) => e.stopPropagation()} className="flex h-full w-full max-w-[440px] flex-col overflow-y-auto border-l-[1.5px] border-[color:var(--ink)] bg-[color:var(--paper)] p-5 shadow-xl">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="label">
              {TYPE_LABELS[mark.type]} · {OFFICE_LABELS_SHORT[mark.office]}
              {mark.clientOwned === false && " · société du groupe"}
              {mark.origin === "manuel" && " · ajouté à la main"}
            </div>
            <h2 className="hand text-[22px] leading-tight">{mark.name}</h2>
            <div className="font-mono text-[11px] text-[color:var(--pencil)]">{mark.id}</div>
          </div>
          <div className="flex shrink-0 items-center gap-3">
            {onFavorite && (
              <button type="button" onClick={() => onFavorite(mark, !mark.favorite)} className="hand text-[20px] leading-none" title={mark.favorite ? "Retirer des favoris" : "Ajouter aux favoris"}>
                {mark.favorite ? "★" : "☆"}
              </button>
            )}
            <button type="button" className="btn-quiet text-[12px]" onClick={onClose}>
              fermer
            </button>
          </div>
        </div>

        <div className="mt-4">
          {row("titulaire", mark.owner || "—")}
          {row("statut", mark.status || "—")}
          {row("dépôt", formatDate(mark.filingDate))}
          {mark.registrationDate && row("enregistrement", formatDate(mark.registrationDate))}
          {row(
            "expiration",
            <>
              {formatDate(d.expiry)} {d.expiry && <span className="text-[color:var(--pencil)]">({daysLabel(d)})</span>}
              {d.state !== "a_venir" && d.state !== "inconnu" && <span className="chip-ink ml-2">{DEADLINE_LABELS[d.state]}</span>}
              {d.computed && <div className="text-[11px] text-[color:var(--pencil)]">calculée à partir du dépôt, à confirmer</div>}
            </>,
          )}
          {(mark.type === "marque" || mark.type === "dessin") && d.windowOpens && row("renouvellement", `du ${formatDate(d.windowOpens)} au ${formatDate(d.graceEnds)} (délai de grâce inclus)`)}
          {row("classes", mark.classes.length ? mark.classes.join(", ") : "—")}
          {row("pays", countries.length ? countries.map(countryName).sort((a, b) => a.localeCompare(b)).join(", ") : mark.office === "WO" ? "pays désignés non chargés" : "—")}
          {mark.notes && row("notes", mark.notes)}
          {WATCHABLE.has(mark.type) &&
            row(
              "photo du produit",
              <div className="space-y-1">
                {photo && (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={photo} alt={`Produit ${mark.name}`} className="max-h-32 rounded border border-[color:var(--line)]" />
                )}
                <div className="flex gap-1">
                  <input value={photoDraft} onChange={(e) => setPhotoDraft(e.target.value)} placeholder="https://… (photo du produit authentique)" className="field w-full py-0.5 text-[12px]" />
                  {photoDraft.trim() !== photo && (
                    <button type="button" className="btn py-0.5 text-[12px]" onClick={savePhoto}>
                      enregistrer
                    </button>
                  )}
                </div>
                <p className="text-[11px] text-[color:var(--pencil)]">Sert de référence pour la recherche visuelle. Sans photo, elle est cherchée sur le site officiel.</p>
                {photoError && <p className="text-[11px] text-red-700">{photoError}</p>}
              </div>,
            )}
          {mark.lastRun && row("dernière veille", `${formatDate(mark.lastRun.startedAt)} · ${mark.lastRun.total} résultats, ${mark.lastRun.high} forts`)}
        </div>

        <div className="mt-5 flex flex-wrap items-center gap-3">
          {WATCHABLE.has(mark.type) && (
            <button type="button" className="btn-ink" disabled={starting} onClick={watch}>
              {starting ? "lancement…" : "rechercher des contrefaçons"}
            </button>
          )}
          {mark.sourceUrl && (
            <a href={mark.sourceUrl} target="_blank" rel="noreferrer" className="btn">
              ouvrir au registre ↗
            </a>
          )}
          {mark.lastRun && (
            <>
              <a href={`${log}?format=xlsx`} className="btn-quiet text-[12px]">
                journal xlsx
              </a>
              <a href={`${log}?format=csv`} className="btn-quiet text-[12px]">
                csv
              </a>
            </>
          )}
          {mark.origin === "manuel" && onDeleted && (
            <button
              type="button"
              className="btn-quiet text-[12px]"
              onClick={async () => {
                if (!window.confirm(`Supprimer « ${mark.name} » ?`)) return;
                const res = await fetch(`/api/marks/${encodeURIComponent(mark.id)}`, { method: "DELETE" });
                if (res.ok) onDeleted(mark.id);
              }}
            >
              supprimer ce titre
            </button>
          )}
        </div>
      </aside>
    </div>
  );
}
