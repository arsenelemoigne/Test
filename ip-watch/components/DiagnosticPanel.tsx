"use client";

import { useState } from "react";
import type { Check } from "@/lib/diagnostic";

const mark = (ok: boolean | null) => (ok === null ? "—" : ok ? "✓" : "✗");

export function DiagnosticPanel() {
  const [checks, setChecks] = useState<Check[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    setBusy(true);
    setError(null);
    const res = await fetch("/api/diagnostic", { method: "POST" });
    const json = await res.json().catch(() => ({}));
    setBusy(false);
    if (!res.ok) return setError(json.error ?? `Erreur ${res.status}`);
    setChecks(json.checks);
  };

  const groups = checks ? Array.from(new Set(checks.map((c) => c.group))) : [];
  const failed = checks?.filter((c) => c.ok === false).length ?? 0;

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3">
        <button type="button" className="btn-ink" disabled={busy} onClick={run}>
          {busy ? "test en cours… (30 s environ)" : checks ? "relancer" : "lancer le diagnostic"}
        </button>
        {checks && <span className="hand text-[14px]">{failed === 0 ? "tout répond" : `${failed} source(s) en erreur`}</span>}
      </div>
      {error && <p className="hand text-[13px] text-red-700">{error}</p>}
      {groups.map((g) => (
        <section key={g}>
          <h2 className="hand text-[16px]">{g}</h2>
          <ul className="mt-1 divide-y divide-dashed divide-[color:var(--line)]">
            {checks!
              .filter((c) => c.group === g)
              .map((c) => (
                <li key={c.name} className="grid grid-cols-[22px_220px_1fr_56px] gap-2 py-2 text-[13px]">
                  <span className={`hand text-[15px] ${c.ok === false ? "text-red-700" : ""}`}>{mark(c.ok)}</span>
                  <span>{c.name}</span>
                  <span className="break-words font-mono text-[11px] text-[color:var(--pencil)]">{c.detail}</span>
                  <span className="text-right font-mono text-[11px] text-[color:var(--pencil)]">{(c.ms / 1000).toFixed(1)} s</span>
                </li>
              ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
