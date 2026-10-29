"use client";

import { useEffect, useState } from "react";
import { useApp } from "@/components/AppProvider";
import { RoughBox } from "@/components/RoughBox";
import { TraceSteps, useTrace } from "@/components/Trace";

export default function NewClientPage() {
  const { role, assistantEmail, coulisses } = useApp();
  const [text, setText] = useState("");
  const [traceId, setTraceId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const trace = useTrace(traceId);

  // When the import ends, open the new client's dashboard.
  useEffect(() => {
    if (!trace || trace.status === "running") return;
    const result = trace.result as { slug?: string; error?: string } | null;
    if (trace.status === "error" || !result?.slug) {
      setError(result?.error ?? "Import impossible.");
      setTraceId(null);
      return;
    }
    fetch("/api/client", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ slug: result.slug }) }).finally(() => window.location.assign("/"));
  }, [trace]);

  if (role !== "avocat") return <p className="hand text-[color:var(--pencil)]">Réservé à l&apos;avocat.</p>;

  const submit = async () => {
    setError(null);
    const res = await fetch("/api/onboard", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ text }) });
    const json = await res.json().catch(() => ({}));
    if (!res.ok || !json.traceId) return setError(json.error ?? `Erreur ${res.status}`);
    setTraceId(json.traceId);
  };

  const busy = traceId !== null;
  const lastStep = trace?.steps[trace.steps.length - 1];

  return (
    <RoughBox seed={5} className="mx-auto mt-4 max-w-xl">
      <div className="space-y-4 p-6">
        <div>
          <h1 className="hand text-[24px] font-semibold">Nouveau client</h1>
          <p className="text-[13px] text-[color:var(--pencil)]">
            Nom de la société, SIREN, lien vers sa fiche ou email transféré : l&apos;assistant trouve la société et importe les marques et dessins qu&apos;elle a déposés.
          </p>
        </div>
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={5} placeholder="Ex. : Notre nouveau client est Boucheron." className="field w-full" autoFocus disabled={busy} />
        {error && <p className="hand text-[13px] text-red-700">{error}</p>}
        <div className="flex items-center justify-between gap-3">
          <p className="text-[12px] text-[color:var(--pencil)]">{assistantEmail ? <>ou écrivez à <span className="hand text-[color:var(--ink)]">{assistantEmail}</span></> : null}</p>
          <button type="button" onClick={submit} disabled={busy || !text.trim()} className="btn-ink">
            {busy ? "import en cours…" : "créer et importer"}
          </button>
        </div>
        {busy && (
          <div className="border-t border-dashed border-[color:var(--line)] pt-3">
            {coulisses && trace ? (
              <TraceSteps steps={trace.steps} live={trace.status === "running"} />
            ) : (
              <p className="hand text-[13px] text-[color:var(--pencil)]">
                {lastStep ? lastStep.name : "démarrage"}
                <span className="dots">
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </p>
            )}
          </div>
        )}
      </div>
    </RoughBox>
  );
}
