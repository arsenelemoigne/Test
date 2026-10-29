"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { Markdown } from "./Markdown";
import { RoughBox } from "./RoughBox";
import type { Mark } from "@/lib/types";

type Message = { role: "user" | "assistant"; content: string; sources?: { name: string; count: number }[]; marks?: string[] };

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
  const [messages, setMessages] = useState<Message[]>([]);
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
    for (const m of [...marks].sort((a, b) => (a.type === b.type ? a.name.localeCompare(b.name) : a.type === "marque" ? -1 : 1))) {
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
  }, [messages, busy]);

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
    setMessages(next);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          messages: next.map((m) => ({ role: m.role, content: m.marks?.length ? `${m.marks.map((t) => `/${t}`).join(" ")} ${m.content}` : m.content })),
          focusMarkIds: focus.map((m) => m.id),
        }),
      });
      const json = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(json.error ?? `Erreur ${res.status}`);
      setMessages([...next, { role: "assistant", content: json.answer, sources: json.sources }]);
      if (json.runId) onRunStarted?.(json.runId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setBusy(false);
    }
  };

  const onChange = (value: string, caret: number) => {
    setInput(value);
    setSlash(slashQuery(value, caret));
    setActive(0);
  };

  return (
    <RoughBox seed={7} className={`flex flex-col ${className}`}>
      <div className="flex h-full min-h-0 flex-col p-4">
        <div className="flex items-baseline justify-between">
          <h2 className="hand text-[17px] font-semibold">Assistant</h2>
          {messages.length > 0 && (
            <button type="button" className="btn-quiet text-[12px]" onClick={() => setMessages([])}>
              effacer
            </button>
          )}
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
              <div key={i}>
                <Markdown text={m.content} />
                {m.sources && m.sources.length > 0 && (
                  <p className="label mt-2">sources consultées : {m.sources.map((s) => (s.count > 1 ? `${s.name} ×${s.count}` : s.name)).join(" · ")}</p>
                )}
              </div>
            ),
          )}
          {busy && (
            <p className="hand text-[13px] text-[color:var(--pencil)]">
              je cherche dans les sources
              <span className="dots">
                <span>.</span>
                <span>.</span>
                <span>.</span>
              </span>
            </p>
          )}
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
