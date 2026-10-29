"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useApp } from "./AppProvider";
import { TraceDrawer, useTrace } from "./Trace";

// Lawyer-only client actions: add, re-import from INPI (followed live), delete.
export function ClientActions() {
  const { role, client, coulisses } = useApp();
  const [traceId, setTraceId] = useState<string | null>(null);
  const [drawer, setDrawer] = useState(false);
  const [busy, setBusy] = useState<"refresh" | "delete" | null>(null);
  const [error, setError] = useState<string | null>(null);
  const trace = useTrace(traceId);

  // When the import ends: reload, unless the lawyer is reading the coulisses.
  useEffect(() => {
    if (!trace || trace.status === "running") return;
    if (trace.status === "error") {
      setBusy(null);
      setError(String((trace.result as { error?: string } | null)?.error ?? "Actualisation impossible."));
      return;
    }
    if (!drawer) window.location.reload();
  }, [trace, drawer]);

  if (role !== "avocat") return null;

  const refresh = async () => {
    setBusy("refresh");
    setError(null);
    const res = await fetch("/api/clients/refresh", { method: "POST" });
    const json = await res.json().catch(() => ({}));
    if (!res.ok || !json.traceId) {
      setBusy(null);
      setError(json.error ?? "Actualisation impossible.");
      return;
    }
    setTraceId(json.traceId);
    if (coulisses) setDrawer(true);
  };

  const remove = async () => {
    if (!client || !window.confirm(`Supprimer ${client.name} avec ses titres, recherches, journaux et historique ? Cette action est définitive.`)) return;
    setBusy("delete");
    setError(null);
    const res = await fetch(`/api/clients/${encodeURIComponent(client.slug)}`, { method: "DELETE" });
    if (!res.ok) {
      setBusy(null);
      setError("Suppression impossible.");
      return;
    }
    window.location.assign("/");
  };

  const lastStep = trace?.steps[trace.steps.length - 1]?.name;

  return (
    <span className="flex flex-wrap items-center gap-3">
      <Link href="/nouveau" className="btn">
        + nouveau client
      </Link>
      {client && (
        <>
          <button type="button" className="btn-quiet text-[12px]" disabled={busy !== null} onClick={refresh}>
            {busy === "refresh" ? `actualisation INPI${lastStep ? ` · ${lastStep.toLowerCase()}` : "…"}` : "actualiser depuis l'INPI"}
          </button>
          {traceId && coulisses && !drawer && (
            <button type="button" className="btn-quiet text-[12px]" onClick={() => setDrawer(true)}>
              coulisses
            </button>
          )}
          <button type="button" className="btn-quiet text-[12px]" disabled={busy !== null} onClick={remove}>
            {busy === "delete" ? "suppression…" : "supprimer ce client"}
          </button>
        </>
      )}
      {error && <span className="hand text-[12px] text-red-700">{error}</span>}
      {drawer && traceId && (
        <TraceDrawer
          id={traceId}
          onClose={() => {
            setDrawer(false);
            if (trace && trace.status === "done") window.location.reload();
          }}
        />
      )}
    </span>
  );
}
