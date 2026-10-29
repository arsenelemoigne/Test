"use client";

import { useState } from "react";
import { useData } from "@/components/EditsProvider";

type Step = { kind: string; query?: string; position?: number; status: number | null; hits?: number | null; error?: string };

const KIND_LABELS: Record<string, string> = {
  login: "Connexion INPI",
  marques: "Marques (FR, UE, international)",
  modeles: "Dessins & modèles (FR, international)",
  metadata: "Date de mise à jour des données",
};

export default function ImportPage() {
  const { role } = useData();
  const [holder, setHolder] = useState("BOUCHERON");
  const [busy, setBusy] = useState(false);
  const [steps, setSteps] = useState<Step[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);

  if (role !== "avocat") return <p className="text-sm text-neutral-600">Import réservé à l&apos;avocat.</p>;

  const run = async () => {
    setBusy(true);
    setError(null);
    setSteps([]);
    try {
      const res = await fetch("/api/admin/import-inpi", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ holder }),
      });
      const json = await res.json().catch(() => ({}));
      setSteps(json.steps ?? []);
      setRunId(json.runId ?? null);
      if (!res.ok) setError(json.error ?? `Erreur ${res.status}`);
    } catch {
      setError("Serveur injoignable.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="max-w-3xl space-y-6">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight">Import depuis l&apos;INPI</h1>
        <p className="text-sm text-neutral-600">
          Recherche les marques (France, UE, international) et les dessins & modèles déposés au nom du titulaire. Les réponses brutes sont conservées pour
          vérification avant d&apos;alimenter le portefeuille.
        </p>
      </div>

      <div className="flex flex-wrap items-end gap-2">
        <label className="block">
          <span className="label">Titulaire</span>
          <input value={holder} onChange={(e) => setHolder(e.target.value)} className="mt-1 block w-72 border border-black px-2 py-1.5 font-mono text-sm" />
        </label>
        <button type="button" onClick={run} disabled={busy || !holder.trim()} className="btn-solid">
          {busy ? "Import en cours…" : "Lancer l'import"}
        </button>
      </div>

      {steps.length > 0 && (
        <table className="table-base">
          <thead>
            <tr>
              <th>Étape</th>
              <th>Page</th>
              <th>Réponse</th>
              <th>Résultats</th>
            </tr>
          </thead>
          <tbody>
            {steps.map((s, i) => (
              <tr key={i}>
                <td>{KIND_LABELS[s.kind] ?? s.kind}</td>
                <td className="font-mono text-xs">{s.position !== undefined ? s.position / 100 + 1 : "—"}</td>
                <td className="font-mono text-xs">{s.status ?? "—"}</td>
                <td className="font-mono text-xs">{s.hits ?? "—"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
      {error && <p className="font-mono text-xs text-red-600">{error}</p>}
      {runId && !error && <p className="font-mono text-xs">Import enregistré (réf. {runId.slice(0, 8)}). Le portefeuille sera mis à jour après vérification.</p>}
    </div>
  );
}
