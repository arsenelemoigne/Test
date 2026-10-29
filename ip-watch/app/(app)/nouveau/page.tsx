"use client";

import { useState } from "react";
import { useApp } from "@/components/AppProvider";
import { RoughBox } from "@/components/RoughBox";

export default function NewClientPage() {
  const { role, assistantEmail } = useApp();
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (role !== "avocat") return <p className="hand text-[color:var(--pencil)]">Réservé à l&apos;avocat.</p>;

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/onboard", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.error ?? `Erreur ${res.status}`);
      window.location.assign("/");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
      setBusy(false);
    }
  };

  return (
    <RoughBox seed={5} className="mx-auto mt-4 max-w-xl">
      <div className="space-y-4 p-6">
        <div>
          <h1 className="hand text-[24px] font-semibold">Nouveau client</h1>
          <p className="text-[13px] text-[color:var(--pencil)]">
            Nom de la société, SIREN, lien vers sa fiche ou email transféré : l&apos;assistant trouve la société et importe ses marques et dessins depuis l&apos;INPI.
          </p>
        </div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={5} placeholder="Ex. : Notre nouveau client est Boucheron." className="field w-full" autoFocus />
        {error && <p className="hand text-[13px] text-red-700">{error}</p>}
        <div className="flex items-center justify-between gap-3">
          <p className="text-[12px] text-[color:var(--pencil)]">{assistantEmail ? <>ou écrivez à <span className="hand text-[color:var(--ink)]">{assistantEmail}</span></> : null}</p>
          <button type="button" onClick={submit} disabled={busy || !text.trim()} className="btn-ink">
            {busy ? (
              <>
                recherche INPI
                <span className="dots">
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </>
            ) : (
              "créer et importer"
            )}
          </button>
        </div>
      </div>
    </RoughBox>
  );
}
