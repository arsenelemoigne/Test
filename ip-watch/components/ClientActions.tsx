"use client";

import Link from "next/link";
import { useState } from "react";
import { useApp } from "./AppProvider";

// Lawyer-only client actions: add, re-import from INPI, delete.
export function ClientActions() {
  const { role, client } = useApp();
  const [busy, setBusy] = useState<"refresh" | "delete" | null>(null);
  const [error, setError] = useState<string | null>(null);
  if (role !== "avocat") return null;

  const refresh = async () => {
    setBusy("refresh");
    setError(null);
    const res = await fetch("/api/clients/refresh", { method: "POST" });
    const json = await res.json().catch(() => ({}));
    if (!res.ok) {
      setBusy(null);
      setError(json.error ?? "Actualisation impossible.");
      return;
    }
    window.location.reload();
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

  return (
    <span className="flex flex-wrap items-center gap-3">
      <Link href="/nouveau" className="btn">
        + nouveau client
      </Link>
      {client && (
        <>
          <button type="button" className="btn-quiet text-[12px]" disabled={busy !== null} onClick={refresh}>
            {busy === "refresh" ? "actualisation INPI…" : "actualiser depuis l'INPI"}
          </button>
          <button type="button" className="btn-quiet text-[12px]" disabled={busy !== null} onClick={remove}>
            {busy === "delete" ? "suppression…" : "supprimer ce client"}
          </button>
        </>
      )}
      {error && <span className="hand text-[12px] text-red-700">{error}</span>}
    </span>
  );
}
