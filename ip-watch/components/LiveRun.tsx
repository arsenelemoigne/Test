"use client";

import { countryName } from "@/lib/territories";
import { formatDateTime, formatPrice } from "@/lib/format";
import { useState } from "react";
import { DerouleDrawer } from "./Deroule";
import type { Finding, RunStep, WatchRun } from "@/lib/types";

const STEP_ICON: Record<RunStep["status"], string> = { running: "…", done: "✓", skipped: "–", error: "×" };

function Confidence({ value }: { value: number | null }) {
  const v = value ?? 0;
  return (
    <span className="flex w-[46px] shrink-0 flex-col items-start" title={`Confiance ${v} %`}>
      <span className="hand text-[15px] leading-none">{v}%</span>
      <span className="mt-1 flex gap-[2px]">
        {[20, 40, 60, 80, 100].map((t) => (
          <span key={t} className={`h-1 w-1.5 rounded-sm ${v >= t - 10 ? "bg-[color:var(--ink)]" : "bg-[color:var(--line)]"}`} />
        ))}
      </span>
    </span>
  );
}

type Props = { run: WatchRun; findings: Finding[]; country: string | null; onCountry: (c: string | null) => void; onClose: () => void };

// Below this, the model judged the result official, press or unrelated: hidden unless asked.
const DISMISSED = 15;

export function LiveRun({ run, findings, country, onCountry, onClose }: Props) {
  const [showDismissed, setShowDismissed] = useState(false);
  const [deroule, setDeroule] = useState(false);
  const inCountry = findings.filter((f) => !country || f.country === country).sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0));
  const dismissed = inCountry.filter((f) => (f.confidence ?? 0) < DISMISSED).length;
  const shown = showDismissed ? inCountry : inCountry.filter((f) => (f.confidence ?? 0) >= DISMISSED);
  const steps = run.stats?.steps ?? [];
  const logHref = `/api/marks/${encodeURIComponent(run.markId)}/log`;

  return (
    <div>
      <div className="flex flex-wrap items-baseline justify-between gap-2">
        <h3 className="hand text-[16px]">
          Recherche « {run.markName} »{" "}
          <span className="text-[13px] text-[color:var(--pencil)]">
            {run.status === "running" ? (
              <>
                en cours
                <span className="dots">
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </>
            ) : run.status === "error" ? (
              `interrompue${run.error ? ` (${run.error})` : ""}`
            ) : (
              `terminée ${formatDateTime(run.finishedAt)}`
            )}
          </span>
        </h3>
        <span className="flex items-center gap-3">
          <a href={`${logHref}?format=xlsx`} className="btn-quiet text-[12px]">
            journal xlsx
          </a>
          <a href={`${logHref}?format=csv`} className="btn-quiet text-[12px]">
            csv
          </a>
          {run.stats?.traceId && (
            <button type="button" className="btn-quiet text-[12px]" onClick={() => setDeroule(true)}>
              déroulé
            </button>
          )}
          <button type="button" onClick={onClose} className="btn-quiet text-[12px]">
            fermer
          </button>
        </span>
      </div>

      {steps.length > 0 && (
        <ol className="mt-2 space-y-0.5 text-[12px]">
          {steps.map((s) => (
            <li key={s.name} className="flex gap-2">
              <span className={`w-3 shrink-0 ${s.status === "error" ? "text-red-700" : ""}`}>{STEP_ICON[s.status]}</span>
              <span className="min-w-0">
                <span className="hand text-[13px]">{s.name}</span>
                {s.detail && <span className={s.status === "error" ? "text-red-700" : "text-[color:var(--pencil)]"}> · {s.detail}</span>}
              </span>
            </li>
          ))}
        </ol>
      )}
      {deroule && run.stats?.traceId && <DerouleDrawer traceId={run.stats.traceId} title={`Recherche « ${run.markName} »`} onClose={() => setDeroule(false)} />}

      {country && (
        <p className="mt-2 text-[12px]">
          filtre : {countryName(country)}{" "}
          <button type="button" className="btn-quiet text-[12px]" onClick={() => onCountry(null)}>
            retirer
          </button>
        </p>
      )}

      <ul className="mt-3 divide-y divide-dashed divide-[color:var(--line)]">
        {shown.map((f) => (
          <li key={f.id} className="flex gap-3 py-2.5">
            <Confidence value={f.confidence} />
            <div className="min-w-0 flex-1">
              <a href={f.url} target="_blank" rel="noreferrer" className="line-clamp-1 text-[13.5px] underline decoration-[color:var(--line)] underline-offset-2 hover:decoration-[color:var(--ink)]">
                {f.title ?? f.url}
              </a>
              <p className="text-[12px] text-[color:var(--pencil)]">
                {[f.source, f.platform && f.platform !== f.source ? f.platform : null, f.country ? countryName(f.country) : null, f.price !== null ? formatPrice(f.price, f.currency) : null]
                  .filter(Boolean)
                  .join(" · ")}
                {f.screenshotPath && (
                  <>
                    {" · "}
                    <a href={`/api/evidence/${f.screenshotPath}`} target="_blank" rel="noreferrer" className="underline" title={f.sha256 ? `SHA-256 ${f.sha256}` : undefined}>
                      capture
                    </a>
                  </>
                )}
              </p>
              {(f.reason || f.company) && (
                <p className="text-[12px] text-[color:var(--pencil)]">
                  {f.reason}
                  {f.reason && f.company ? " — " : ""}
                  {f.company?.name}
                </p>
              )}
            </div>
          </li>
        ))}
        {shown.length === 0 && <li className="hand py-4 text-[13px] text-[color:var(--pencil)]">{run.status === "running" ? "les premiers résultats arrivent…" : "aucun résultat"}</li>}
      </ul>
      {dismissed > 0 && (
        <button type="button" className="btn-quiet mt-1 text-[12px]" onClick={() => setShowDismissed((v) => !v)}>
          {showDismissed ? "masquer" : "afficher"} les {dismissed} résultat(s) écarté(s) (officiel, presse, sans rapport)
        </button>
      )}
    </div>
  );
}
