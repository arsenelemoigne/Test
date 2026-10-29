"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Markdown } from "./Markdown";
import { useApp } from "./AppProvider";
import { RoughBox } from "./RoughBox";
import { CopyButton, DerouleToggle, DerouleView } from "./Deroule";
import { useTrace } from "./Trace";
import type { TraceStep } from "@/lib/trace";
import type { ConversationMessage, Mark } from "@/lib/types";

type Message = { role: "user" | "assistant"; content: string; sources?: { name: string; count: number }[]; marks?: string[]; steps?: TraceStep[]; traceId?: string | null; runId?: string | null };
type Saved = { id: string; title: string; updatedAt: string; count: number };

const storageKey = (clientId: string | undefined) => `ipw-conversation-${clientId ?? "aucun"}`;
const remember = (key: string, id: string | null) => {
  try {
    if (id) localStorage.setItem(key, id);
    else localStorage.removeItem(key);
  } catch {
    // private browsing: the conversation is still saved server-side
  }
};

function AssistantMessage({ m, onRun }: { m: Message; onRun?: (runId: string) => void }) {
  const ref = useRef<HTMLDivElement>(null);
  return (
    <div>
      <div ref={ref}>
        <Markdown text={m.content} />
      </div>
      {m.sources && m.sources.length > 0 && (
        <p className="label mt-2">sources consultées : {m.sources.map((s) => (s.count > 1 ? `${s.name} ×${s.count}` : s.name)).join(" · ")}</p>
      )}
      {m.runId && onRun && (
        <button type="button" className="btn mt-2 py-0.5 text-[12px]" onClick={() => onRun(m.runId!)}>
          suivre la recherche : étapes, résultats, captures →
        </button>
      )}
      <div className="mt-1 flex flex-wrap items-start gap-x-3">
        <CopyButton getHtml={() => ref.current?.innerHTML ?? ""} text={m.content} />
        <DerouleToggle steps={m.steps} traceId={m.traceId} />
      </div>
    </div>
  );
}

function ConversationList({ currentId, onOpen, onClose }: { currentId: string | null; onOpen: (id: string) => void; onClose: () => void }) {
  const [items, setItems] = useState<Saved[] | null>(null);
  const [query, setQuery] = useState("");
  useEffect(() => {
    fetch("/api/conversations", { cache: "no-store" })
      .then((r) => r.json())
      .then((j) => setItems(j.conversations ?? []))
      .catch(() => setItems([]));
  }, []);
  const shown = (items ?? []).filter((c) => !query.trim() || c.title.toLowerCase().includes(query.trim().toLowerCase()));
  return (
    <div className="absolute inset-0 z-20 flex flex-col bg-[color:var(--paper)] p-4">
      <div className="flex items-baseline justify-between">
        <h3 className="hand text-[16px]">Conversations enregistrées</h3>
        <button type="button" className="btn-quiet text-[12px]" onClick={onClose}>
          fermer
        </button>
      </div>
      <p className="mt-0.5 text-[11.5px] text-[color:var(--pencil)]">Chaque conversation repart de zéro : les autres ne servent jamais à répondre.</p>
      <input value={query} onChange={(e) => setQuery(e.target.value)} placeholder="filtrer…" className="field mt-2 py-1 text-[13px]" />
      <ul className="mt-2 min-h-0 flex-1 divide-y divide-dashed divide-[color:var(--line)] overflow-y-auto">
        {items === null && <li className="hand py-2 text-[13px] text-[color:var(--pencil)]">chargement…</li>}
        {items !== null && shown.length === 0 && <li className="hand py-2 text-[13px] text-[color:var(--pencil)]">Aucune conversation.</li>}
        {shown.map((c) => (
          <li key={c.id} className="flex items-baseline gap-2 py-2">
            <button type="button" onClick={() => onOpen(c.id)} className="min-w-0 flex-1 text-left hover:underline">
              <span className={`block truncate text-[13px] ${c.id === currentId ? "font-semibold" : ""}`}>{c.title}</span>
              <span className="font-mono text-[10.5px] text-[color:var(--pencil)]">
                {new Date(c.updatedAt).toLocaleString("fr-FR", { dateStyle: "short", timeStyle: "short", timeZone: "Europe/Paris" })} · {c.count} question(s)
              </span>
            </button>
            <button
              type="button"
              className="btn-quiet shrink-0 text-[11px]"
              title="Supprimer"
              onClick={async () => {
                if (!window.confirm(`Supprimer « ${c.title} » ?`)) return;
                await fetch(`/api/conversations/${c.id}`, { method: "DELETE" });
                setItems((all) => (all ?? []).filter((x) => x.id !== c.id));
              }}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
    </div>
  );
}

type Props = { marks: Mark[]; onRunStarted?: (runId: string) => void; className?: string };

const EXAMPLES = [
  { text: "Quelles marques arrivent à échéance dans les 12 prochains mois ?" },
  { text: "Lance une recherche de contrefaçons", needsMark: true },
  { text: "Quelles sont les conditions de la contrefaçon de marque ?" },
];

// Finds a "/name" token right before the caret.
function slashQuery(value: string, caret: number): { start: number; query: string } | null {
  const before = value.slice(0, caret);
  const i = before.lastIndexOf("/");
  if (i < 0 || (i > 0 && !/\s/.test(before[i - 1]))) return null;
  const query = before.slice(i + 1);
  if (query.length > 30 || /\n/.test(query)) return null;
  return { start: i, query };
}

export function AssistantPanel({ marks, onRunStarted, className = "" }: Props) {
  const { client } = useApp();
  const key = storageKey(client?.id);
  const [messages, setMessages] = useState<Message[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [listOpen, setListOpen] = useState(false);
  const [pendingTrace, setPendingTrace] = useState<string | null>(null);
  const live = useTrace(pendingTrace);
  const [input, setInput] = useState("");
  const [focus, setFocus] = useState<Mark[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [slash, setSlash] = useState<{ start: number; query: string } | null>(null);
  const [active, setActive] = useState(0);
  const inputRef = useRef<HTMLTextAreaElement>(null);
  const endRef = useRef<HTMLDivElement>(null);

  // One entry per distinct name, marks first.
  const choices = useMemo(() => {
    const byName = new Map<string, Mark>();
    const own = marks.filter((m) => m.clientOwned !== false);
    for (const m of [...own].sort((a, b) => Number(b.favorite) - Number(a.favorite) || (a.type === b.type ? a.name.localeCompare(b.name) : a.type === "marque" ? -1 : 1))) {
      if (!byName.has(m.name.toUpperCase())) byName.set(m.name.toUpperCase(), m);
    }
    return Array.from(byName.values());
  }, [marks]);

  const matches = useMemo(() => {
    if (!slash) return [];
    const q = slash.query.toUpperCase();
    return choices.filter((m) => m.name.toUpperCase().includes(q) && !focus.some((f) => f.id === m.id)).slice(0, 6);
  }, [slash, choices, focus]);

  useEffect(() => {
    endRef.current?.scrollIntoView({ block: "nearest" });
  }, [messages, busy, live?.steps.length]);

  const open = useCallback(
    async (id: string) => {
      const res = await fetch(`/api/conversations/${id}`, { cache: "no-store" }).catch(() => null);
      if (!res?.ok) {
        remember(key, null);
        return;
      }
      const c = (await res.json()) as { id: string; messages: ConversationMessage[] };
      setMessages(c.messages.map((m) => ({ role: m.role, content: m.content, marks: m.marks, sources: m.sources, traceId: m.traceId, runId: m.runId })));
      setConversationId(c.id);
      setFocus([]);
      setListOpen(false);
      remember(key, c.id);
    },
    [key],
  );

  // Picks up the last conversation after a reload.
  useEffect(() => {
    let saved: string | null = null;
    try {
      saved = localStorage.getItem(key);
    } catch {
      saved = null;
    }
    if (saved) void open(saved);
  }, [key, open]);

  const startNew = () => {
    setMessages([]);
    setConversationId(null);
    setFocus([]);
    setError(null);
    remember(key, null);
  };

  const pick = (m: Mark) => {
    if (!slash) return;
    const next = `${input.slice(0, slash.start)}${input.slice(slash.start + 1 + slash.query.length)}`.replace(/\s{2,}/g, " ");
    setInput(next.trimStart());
    setFocus((f) => [...f, m]);
    setSlash(null);
    requestAnimationFrame(() => inputRef.current?.focus());
  };

  const send = async (text: string) => {
    const content = text.trim();
    if (!content || busy) return;
    const tags = focus.map((m) => m.name);
    const next: Message[] = [...messages, { role: "user", content, marks: tags }];
    const traceId = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : null;
    setMessages(next);
    setInput("");
    setBusy(true);
    setError(null);
    setPendingTrace(traceId);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: next.map((m) => ({ role: m.role, content: m.marks?.length ? `${m.marks.map((t) => `/${t}`).join(" ")} ${m.content}` : m.content })),
          focusMarkIds: focus.map((m) => m.id),
          conversationId,
          display: { content, marks: tags },
          traceId,
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.error ?? `Erreur ${res.status}`);
      setMessages([...next, { role: "assistant", content: json.answer, sources: json.sources, steps: json.steps, traceId: json.traceId, runId: json.runId }]);
      if (json.conversationId) {
        setConversationId(json.conversationId);
        remember(key, json.conversationId);
      }
      if (json.runId) onRunStarted?.(json.runId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setBusy(false);
      setPendingTrace(null);
    }
  };

  const onChange = (value: string, caret: number) => {
    setInput(value);
    setSlash(slashQuery(value, caret));
    setActive(0);
  };

  return (
    <RoughBox seed={7} className={`overflow-hidden ${className}`}>
      <div className="relative flex h-full min-h-0 flex-col p-4">
        {listOpen && <ConversationList currentId={conversationId} onOpen={open} onClose={() => setListOpen(false)} />}
        <div className="flex items-baseline justify-between">
          <h2 className="hand text-[17px] font-semibold">Assistant</h2>
          <span className="flex gap-3">
            <button type="button" className="btn-quiet text-[12px]" onClick={() => setListOpen(true)}>
              conversations
            </button>
            {messages.length > 0 && (
              <button type="button" className="btn-quiet text-[12px]" onClick={startNew} disabled={busy}>
                + nouvelle
              </button>
            )}
          </span>
        </div>
        <p className="label">registre · Légifrance · Judilibre · INPI · annuaire · BODACC</p>

        <div className="mt-3 min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
          {messages.length === 0 && (
            <div className="space-y-2 pt-1">
              <p className="text-[13px] text-[color:var(--pencil)]">
                Tapez <span className="hand text-[color:var(--ink)]">/</span> pour viser une marque.
              </p>
              {EXAMPLES.map((ex) => (
                <button
                  key={ex.text}
                  type="button"
                  onClick={() => {
                    if (ex.needsMark) {
                      setInput(`/ ${ex.text.toLowerCase()}`);
                      setSlash({ start: 0, query: "" });
                      inputRef.current?.focus();
                    } else send(ex.text);
                  }}
                  className="block w-full rounded-md border border-dashed border-[color:var(--line)] px-3 py-2 text-left text-[13px] hover:border-[color:var(--ink)]"
                >
                  {ex.needsMark ? <span className="hand">/marque </span> : null}
                  {ex.text}
                </button>
              ))}
            </div>
          )}
          {messages.map((m, i) =>
            m.role === "user" ? (
              <div key={i} className="ml-6 rounded-lg bg-[color:var(--faint)] px-3 py-2 text-[13.5px]">
                {m.marks?.map((t) => (
                  <span key={t} className="chip-ink mr-1">
                    /{t}
                  </span>
                ))}
                {m.content}
              </div>
            ) : (
              <AssistantMessage key={i} m={m} onRun={onRunStarted} />
            ),
          )}
          {busy &&
            (live && live.id === pendingTrace && live.steps.length > 0 ? (
              <div className="rounded-md border border-dashed border-[color:var(--line)] p-2">
                <p className="label mb-1">déroulé en direct</p>
                <DerouleView steps={live.steps} live />
              </div>
            ) : (
              <p className="hand text-[13px] text-[color:var(--pencil)]">
                je cherche dans les sources
                <span className="dots">
                  <span>.</span>
                  <span>.</span>
                  <span>.</span>
                </span>
              </p>
            ))}
          {error && <p className="hand text-[13px] text-red-700">{error}</p>}
          <div ref={endRef} />
        </div>

        <div className="relative mt-3">
          {slash && matches.length > 0 && (
            <ul className="wobble absolute bottom-full left-0 z-10 mb-2 w-full bg-[color:var(--paper)] py-1 shadow-sm">
              {matches.map((m, i) => (
                <li key={m.id}>
                  <button
                    type="button"
                    onMouseDown={(e) => {
                      e.preventDefault();
                      pick(m);
                    }}
                    className={`flex w-full items-baseline justify-between px-3 py-1.5 text-left text-[13px] ${i === active ? "bg-[color:var(--faint)]" : ""}`}
                  >
                    <span className="hand">{m.name}</span>
                    <span className="font-mono text-[10px] text-[color:var(--pencil)]">{m.type === "marque" ? m.office : "dessin"}</span>
                  </button>
                </li>
              ))}
            </ul>
          )}
          {focus.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1">
              {focus.map((m) => (
                <span key={m.id} className="chip-ink">
                  /{m.name}
                  <button type="button" aria-label={`Retirer ${m.name}`} onClick={() => setFocus((f) => f.filter((x) => x.id !== m.id))} className="ml-0.5 opacity-70 hover:opacity-100">
                    ×
                  </button>
                </span>
              ))}
            </div>
          )}
          <form
            className="flex items-end gap-2"
            onSubmit={(e) => {
              e.preventDefault();
              send(input);
            }}
          >
            <textarea
              ref={inputRef}
              rows={2}
              value={input}
              onChange={(e) => onChange(e.target.value, e.target.selectionStart ?? e.target.value.length)}
              onKeyDown={(e) => {
                if (slash && matches.length) {
                  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
                    e.preventDefault();
                    const step = e.key === "ArrowDown" ? 1 : matches.length - 1;
                    setActive((a) => (a + step) % matches.length);
                    return;
                  }
                  if (e.key === "Enter" || e.key === "Tab") {
                    e.preventDefault();
                    pick(matches[active]);
                    return;
                  }
                  if (e.key === "Escape") {
                    setSlash(null);
                    return;
                  }
                }
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              placeholder="Une question, ou /marque…"
              className="field min-h-[44px] flex-1 resize-none"
            />
            <button className="btn-ink" disabled={busy || !input.trim()}>
              envoyer
            </button>
          </form>
        </div>
      </div>
    </RoughBox>
  );
}
