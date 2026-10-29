"use client";

import { useState } from "react";
import { TraceSteps, useTrace } from "./Trace";
import type { TraceStep } from "@/lib/trace";

const AGENT_ROUND = /^agent-\d+$/;

const host = (url: string) => {
  try {
    return new URL(url).hostname.replace(/^www\./, "");
  } catch {
    return url;
  }
};

const secs = (ms?: number) => (ms === undefined ? "" : `${(ms / 1000).toFixed(1).replace(".", ",")} s`);

function Thinking({ text }: { text: string }) {
  const [full, setFull] = useState(false);
  const long = text.length > 320;
  return (
    <div className="border-l-2 border-dotted border-[color:var(--line)] pl-2 text-[12px] italic leading-snug text-[color:var(--pencil)]">
      <span className="hand not-italic text-[color:var(--ink)]">réflexion · </span>
      <span className="whitespace-pre-wrap">{full || !long ? text : `${text.slice(0, 320)}…`}</span>
      {long && (
        <button type="button" className="btn-quiet ml-1 text-[11px] not-italic" onClick={() => setFull((f) => !f)}>
          {full ? "réduire" : "lire la suite"}
        </button>
      )}
    </div>
  );
}

function Links({ links }: { links: { url: string; title?: string }[] }) {
  return (
    <ul className="mt-1 space-y-0.5 pl-6">
      {links.map((l) => (
        <li key={l.url} className="truncate text-[11.5px]">
          ↳{" "}
          <a href={l.url} target="_blank" rel="noreferrer" className="underline decoration-[color:var(--line)] underline-offset-2 hover:decoration-[color:var(--ink)]">
            {l.title || host(l.url)}
          </a>{" "}
          <span className="font-mono text-[10px] text-[color:var(--pencil)]">{host(l.url)}</span>
        </li>
      ))}
    </ul>
  );
}

function totals(steps: TraceStep[]) {
  const ai = steps.filter((s) => s.kind === "ia" && s.ok);
  const models = Array.from(new Set(ai.map((s) => (s.detail ?? "").split(" · ")[0]).filter(Boolean)));
  const tokens = ai.reduce((n, s) => n + Number((s.detail ?? "").match(/(\d+) tokens/)?.[1] ?? 0), 0);
  const tools = steps.filter((s) => s.name.startsWith("outil · ")).length;
  const duration = steps.length > 1 ? new Date(steps[steps.length - 1].at).getTime() - new Date(steps[0].at).getTime() + (steps[steps.length - 1].ms ?? 0) : 0;
  return { models, tokens, tools, aiCalls: ai.length, duration };
}

// The assistant's work in reading order: what it thought, which source it asked, what came back.
export function DerouleView({ steps, live = false }: { steps: TraceStep[]; live?: boolean }) {
  const [technical, setTechnical] = useState(false);
  const t = totals(steps);
  let n = 0;
  return (
    <div className="space-y-2 text-[12.5px]">
      <ol className="space-y-2">
        {steps.map((s, i) => {
          if (s.kind === "ia" && AGENT_ROUND.test(s.name)) {
            if (!s.ok) return <li key={i} className="text-[12px] text-red-700">✗ modèle : {s.detail}</li>;
            const final = typeof s.output === "string" || s.output === null || s.output === undefined;
            if (!s.thinking && !s.summary && !final) return null;
            return (
              <li key={i} className="space-y-1">
                {s.thinking && <Thinking text={s.thinking} />}
                {s.summary && <p className="text-[12.5px]">« {s.summary} »</p>}
                {final && <p className="hand text-[12.5px] text-[color:var(--pencil)]">rédaction de la réponse · {secs(s.ms)}</p>}
              </li>
            );
          }
          if (s.kind === "ia") return null;
          if (s.name.startsWith("outil · ")) {
            n += 1;
            return (
              <li key={i} className="rounded-md border border-dashed border-[color:var(--line)] px-2 py-1.5">
                <div className="flex items-baseline gap-2">
                  <span className="hand w-4 shrink-0 text-[13px]">{n}</span>
                  <span className="min-w-0 flex-1">
                    <span className="hand text-[13px]">{s.detail}</span>
                    {s.what && <span> · {s.what}</span>}
                  </span>
                  <span className={`shrink-0 font-mono text-[10.5px] ${s.ok ? "text-[color:var(--pencil)]" : "text-red-700"}`}>
                    {s.ok ? "✓" : "✗"} {secs(s.ms)}
                  </span>
                </div>
                {s.summary && <p className={`mt-0.5 pl-6 text-[12px] ${s.ok ? "text-[color:var(--pencil)]" : "text-red-700"}`}>{s.summary}</p>}
                {s.links && s.links.length > 0 && <Links links={s.links} />}
              </li>
            );
          }
          if (s.kind === "etape")
            return (
              <li key={i} className="text-[12px]">
                <span className={s.ok ? "" : "text-red-700"}>{s.ok ? "✓" : "!"}</span> <span className="hand text-[13px]">{s.name}</span>
                {s.detail && <span className="text-[color:var(--pencil)]"> · {s.detail}</span>}
                {s.summary && <p className="pl-4 text-[12px]">{s.summary}</p>}
                {s.links && s.links.length > 0 && <Links links={s.links} />}
              </li>
            );
          // Calls made inside a tool or a search (registry, page, capture…), kept discreet.
          return (
            <li key={i} className="pl-4 text-[11px] text-[color:var(--pencil)]">
              <span className="font-mono">
                {s.ok ? "·" : "✗"} {s.name}
              </span>
              {s.what && <span> · {s.what}</span>}
              {s.detail && <span> · {s.detail}</span>} <span className="font-mono">{secs(s.ms)}</span>
              {!s.ok && s.summary && <p className="pl-3 text-red-700">{s.summary}</p>}
              {s.links && s.links.length > 0 && <Links links={s.links} />}
            </li>
          );
        })}
        {live && (
          <li className="hand text-[13px] text-[color:var(--pencil)]">
            en cours
            <span className="dots">
              <span>.</span>
              <span>.</span>
              <span>.</span>
            </span>
          </li>
        )}
      </ol>
      {!live && steps.length > 0 && (
        <p className="font-mono text-[10.5px] text-[color:var(--pencil)]">
          {t.models.join(", ")} · {t.aiCalls} appel(s) IA{t.tokens ? ` · ${t.tokens.toLocaleString("fr-FR")} tokens` : ""} · {secs(t.duration)}
          <button type="button" className="btn-quiet ml-2 text-[10.5px]" onClick={() => setTechnical((v) => !v)}>
            {technical ? "masquer la vue technique" : "vue technique"}
          </button>
        </p>
      )}
      {technical && (
        <div className="rounded border border-dashed border-[color:var(--line)] p-2">
          <TraceSteps steps={steps} />
        </div>
      )}
    </div>
  );
}

function Loaded({ traceId }: { traceId: string }) {
  const trace = useTrace(traceId);
  if (!trace) return <p className="hand text-[12px] text-[color:var(--pencil)]">chargement…</p>;
  return <DerouleView steps={trace.steps} live={trace.status === "running"} />;
}

// Collapsed by default under each answer; loads the trace on demand for saved conversations.
export function DerouleToggle({ steps, traceId }: { steps?: TraceStep[]; traceId?: string | null }) {
  const [open, setOpen] = useState(false);
  if (!steps?.length && !traceId) return null;
  const t = steps ? totals(steps) : null;
  return (
    <div>
      <button type="button" className="btn-quiet text-[11.5px]" onClick={() => setOpen((o) => !o)}>
        {open ? "▾" : "▸"} déroulé{t ? ` · ${t.tools} source(s) interrogée(s) · ${secs(t.duration)}` : ""}
      </button>
      {open && <div className="mt-2">{steps?.length ? <DerouleView steps={steps} /> : <Loaded traceId={traceId!} />}</div>}
    </div>
  );
}

// Side panel with the readable déroulé of a long operation (infringement search, import).
export function DerouleDrawer({ traceId, title, onClose }: { traceId: string; title: string; onClose: () => void }) {
  return (
    <div className="fixed inset-0 z-40 flex justify-end bg-black/10" onClick={onClose}>
      <aside onClick={(e) => e.stopPropagation()} className="flex h-full w-full max-w-[520px] flex-col border-l-[1.5px] border-[color:var(--ink)] bg-[color:var(--paper)] p-5 shadow-xl">
        <div className="flex items-baseline justify-between gap-3">
          <div className="min-w-0">
            <div className="label">déroulé</div>
            <h2 className="hand truncate text-[17px]">{title}</h2>
          </div>
          <button type="button" className="btn-quiet text-[12px]" onClick={onClose}>
            fermer
          </button>
        </div>
        <div className="mt-3 min-h-0 flex-1 overflow-y-auto pr-1">
          <Loaded traceId={traceId} />
        </div>
      </aside>
    </div>
  );
}

// Copies an answer as formatted text (Word, email) with a plain-text fallback.
export function CopyButton({ getHtml, text }: { getHtml: () => string; text: string }) {
  const [done, setDone] = useState(false);
  const plain = text
    .replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, "$1 ($2)")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/(^|\s)\*([^*\n]+)\*/g, "$1$2")
    .replace(/^#+\s*/gm, "");
  const copy = async () => {
    try {
      if (typeof ClipboardItem !== "undefined" && navigator.clipboard?.write) {
        await navigator.clipboard.write([new ClipboardItem({ "text/html": new Blob([getHtml()], { type: "text/html" }), "text/plain": new Blob([plain], { type: "text/plain" }) })]);
      } else await navigator.clipboard.writeText(plain);
    } catch {
      await navigator.clipboard.writeText(plain).catch(() => undefined);
    }
    setDone(true);
    setTimeout(() => setDone(false), 1500);
  };
  return (
    <button type="button" className="btn-quiet text-[11.5px]" onClick={copy} title="Copier la réponse (mise en forme conservée)">
      {done ? "copié ✓" : "copier"}
    </button>
  );
}
