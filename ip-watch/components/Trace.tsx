"use client";

import { useEffect, useState } from "react";
import { useApp } from "./AppProvider";
import type { TraceRecord, TraceStep } from "@/lib/trace";

const KIND = { ia: "IA", api: "API", etape: "étape" } as const;

function fmtMs(ms?: number) {
  if (ms === undefined) return "";
  return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(1).replace(".", ",")} s`;
}

function Value({ v }: { v: unknown }) {
  if (v === undefined || v === null || v === "") return null;
  return <pre className="mt-1 max-h-56 overflow-auto whitespace-pre-wrap break-words rounded bg-[color:var(--faint)] p-2 font-mono text-[10.5px] leading-snug">{typeof v === "string" ? v : JSON.stringify(v, null, 2)}</pre>;
}

// Compact log of an operation: one line per AI decision, API call or step, expandable for inputs and outputs.
export function TraceSteps({ steps, live = false }: { steps: TraceStep[]; live?: boolean }) {
  const [open, setOpen] = useState<number | null>(null);
  const start = steps[0] ? new Date(steps[0].at).getTime() : 0;
  return (
    <ol className="divide-y divide-dashed divide-[color:var(--line)] font-mono text-[11px]">
      {steps.map((s, i) => (
        <li key={i} className="py-1.5">
          <button type="button" onClick={() => setOpen(open === i ? null : i)} className="flex w-full items-baseline gap-2 text-left">
            <span className="w-10 shrink-0 text-[color:var(--pencil)]">+{((new Date(s.at).getTime() - start) / 1000).toFixed(1)}s</span>
            <span className={`w-9 shrink-0 ${s.kind === "ia" ? "font-semibold" : ""}`}>{KIND[s.kind]}</span>
            <span className={`min-w-0 flex-1 ${s.ok ? "" : "text-red-700"}`}>
              {s.name}
              {s.detail && <span className="text-[color:var(--pencil)]"> · {s.detail}</span>}
            </span>
            <span className="shrink-0 text-[color:var(--pencil)]">{fmtMs(s.ms)}</span>
          </button>
          {open === i && (
            <div className="pl-[5.25rem]">
              {s.input !== undefined && <div className="mt-1 text-[color:var(--pencil)]">entrée</div>}
              <Value v={s.input} />
              {s.output !== undefined && <div className="mt-1 text-[color:var(--pencil)]">sortie</div>}
              <Value v={s.output} />
            </div>
          )}
        </li>
      ))}
      {live && (
        <li className="py-1.5 text-[color:var(--pencil)]">
          en cours
          <span className="dots">
            <span>.</span>
            <span>.</span>
            <span>.</span>
          </span>
        </li>
      )}
    </ol>
  );
}

// Polls a trace while its operation runs.
export function useTrace(id: string | null): TraceRecord | null {
  const [trace, setTrace] = useState<TraceRecord | null>(null);
  useEffect(() => {
    if (!id) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const res = await fetch(`/api/traces/${id}`, { cache: "no-store" }).catch(() => null);
      if (stop) return;
      if (res?.ok) {
        const t = (await res.json()) as TraceRecord;
        setTrace(t);
        if (t.status !== "running") return;
      }
      timer = setTimeout(tick, 1200);
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [id]);
  return trace;
}

// Side panel showing a trace; used for imports, searches and history entries.
export function TraceDrawer({ id, title, onClose }: { id: string; title?: string; onClose: () => void }) {
  const trace = useTrace(id);
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/10" onClick={onClose}>
      <aside onClick={(e) => e.stopPropagation()} className="flex h-full w-full max-w-[460px] flex-col border-l-[1.5px] border-[color:var(--ink)] bg-[color:var(--paper)] p-5 shadow-xl">
        <div className="flex items-baseline justify-between gap-3">
          <div className="min-w-0">
            <div className="label">coulisses</div>
            <h2 className="hand truncate text-[17px]">{title ?? trace?.title ?? "…"}</h2>
          </div>
          <button type="button" className="btn-quiet text-[12px]" onClick={onClose}>
            fermer
          </button>
        </div>
        <p className="mt-1 text-[12px] text-[color:var(--pencil)]">
          Chaque décision de l&apos;IA et chaque appel aux sources, dans l&apos;ordre. Cliquez une ligne pour voir l&apos;entrée et la sortie.
        </p>
        <div className="mt-3 min-h-0 flex-1 overflow-y-auto">
          {trace ? <TraceSteps steps={trace.steps} live={trace.status === "running"} /> : <p className="hand text-[13px] text-[color:var(--pencil)]">chargement…</p>}
          {trace?.status === "error" && <p className="hand mt-3 text-[13px] text-red-700">Échec : {String((trace.result as { error?: string } | null)?.error ?? "")}</p>}
        </div>
      </aside>
    </div>
  );
}

// Discreet switch in the footer.
export function CoulissesToggle() {
  const { coulisses, setCoulisses } = useApp();
  return (
    <button type="button" onClick={() => setCoulisses(!coulisses)} className="btn-quiet text-[12px]" title="Afficher ce qui se passe en coulisses : décisions de l'IA, appels aux sources">
      coulisses {coulisses ? "●" : "○"}
    </button>
  );
}

// Link that opens a trace, shown only when the coulisses are on.
export function TraceLink({ id, title, className = "" }: { id: string | null | undefined; title?: string; className?: string }) {
  const { coulisses } = useApp();
  const [open, setOpen] = useState(false);
  if (!coulisses || !id) return null;
  return (
    <>
      <button type="button" className={`btn-quiet text-[11.5px] ${className}`} onClick={() => setOpen(true)}>
        coulisses
      </button>
      {open && <TraceDrawer id={id} title={title} onClose={() => setOpen(false)} />}
    </>
  );
}
