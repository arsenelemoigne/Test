"use client";

import { useState } from "react";
import { useData } from "@/components/EditsProvider";
import { deadlineFor } from "@/lib/deadlines";
import { rightCountries } from "@/lib/territories";

type Message = { role: "user" | "assistant"; content: string };

const SUGGESTIONS = [
  "Quelles marques expirent dans les 12 prochains mois ?",
  "Dans quels pays la bague est-elle protégée par un dessin ?",
  "Quels signalements justifient un constat ?",
  "Résume l'état du portefeuille pour la direction.",
];

export default function AssistantPage() {
  const { client, rights, findings, settings } = useData();
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const ask = async (question: string) => {
    const q = question.trim();
    if (!q || busy) return;
    const next: Message[] = [...messages, { role: "user", content: q }];
    setMessages(next);
    setInput("");
    setBusy(true);
    setError(null);
    try {
      // Send the reviewed data (with the lawyer's edits) and pre-computed deadlines.
      const data = {
        rights: rights.map((r) => {
          const d = deadlineFor(r);
          return { ...r, countries: rightCountries(r), expiry: d.expiry?.toISOString().slice(0, 10) ?? null, renewalOpens: d.windowOpens?.toISOString().slice(0, 10) ?? null, deadlineState: d.state };
        }),
        findings,
      };
      const res = await fetch("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ messages: next, data }) });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error ?? "Erreur");
      setMessages([...next, { role: "assistant", content: json.answer }]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Erreur");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="grid gap-6 lg:grid-cols-3">
      <div className="lg:col-span-2">
        <h1 className="text-2xl font-semibold tracking-tight">Assistant</h1>
        <p className="text-sm text-neutral-600">Répond uniquement à partir des titres et signalements de {client.name}, en citant les identifiants.</p>

        <div className="mt-4 min-h-[320px] space-y-4 border border-black p-4">
          {messages.length === 0 && (
            <div className="space-y-2">
              <div className="label">Exemples</div>
              {SUGGESTIONS.map((s) => (
                <button key={s} type="button" onClick={() => ask(s)} disabled={!settings.assistantEnabled} className="block w-full border border-neutral-300 px-3 py-2 text-left text-sm hover:border-black disabled:opacity-40">
                  {s}
                </button>
              ))}
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={m.role === "user" ? "ml-auto max-w-[85%] bg-black px-3 py-2 text-sm text-white" : "max-w-[90%] whitespace-pre-wrap border-l-2 border-black pl-3 text-sm"}>
              {m.content}
            </div>
          ))}
          {busy && <div className="font-mono text-xs text-neutral-500">Recherche dans le portefeuille…</div>}
          {error && <div className="font-mono text-xs text-red-600">{error}</div>}
        </div>

        <form
          className="mt-2 flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            ask(input);
          }}
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={!settings.assistantEnabled}
            placeholder={settings.assistantEnabled ? "Posez une question sur le portefeuille…" : "Assistant non configuré (clé OpenRouter manquante)"}
            className="flex-1 border border-black px-3 py-2 text-sm"
          />
          <button className="btn-solid" disabled={busy || !settings.assistantEnabled}>
            Envoyer
          </button>
        </form>
      </div>

      <aside className="space-y-4 text-sm">
        <div className="border border-black p-4">
          <div className="label">Par email</div>
          <p className="mt-2">
            Écrivez ou mettez en copie{" "}
            <span className="font-mono">{settings.assistantEmail ?? "l'adresse de l'assistant"}</span> : la réponse arrive dans le même fil, à partir des mêmes données.
          </p>
          <p className="mt-2 text-xs text-neutral-500">Seules les adresses autorisées par le cabinet reçoivent une réponse. Mise en service à l&apos;étape 3.</p>
        </div>
        <div className="border border-black p-4">
          <div className="label">Limites</div>
          <ul className="mt-2 list-disc space-y-1 pl-4 text-xs text-neutral-600">
            <li>Pas de connaissance extérieure : si l&apos;information n&apos;est pas dans le registre, l&apos;assistant le dit.</li>
            <li>Les signalements restent des pistes ; l&apos;avocat qualifie.</li>
            <li>Aucun conseil juridique automatisé : l&apos;assistant renvoie vers l&apos;avocat.</li>
          </ul>
        </div>
      </aside>
    </div>
  );
}
