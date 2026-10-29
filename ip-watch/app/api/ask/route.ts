import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";

export const maxDuration = 60;

const DEFAULT_MODEL = "google/gemini-2.5-flash";
const MAX_CONTEXT_CHARS = 400_000;

type Message = { role: "user" | "assistant"; content: string };

function systemPrompt(clientName: string, today: string) {
  return [
    `Tu es l'assistant d'un cabinet d'avocats qui suit le portefeuille de propriété industrielle de ${clientName}.`,
    `Nous sommes le ${today}.`,
    "Réponds uniquement à partir des données JSON fournies (titres, échéances, signalements de veille). N'utilise aucune connaissance extérieure sur la société ou ses titres.",
    "Cite l'identifiant de chaque titre ou signalement sur lequel tu t'appuies, entre crochets, par exemple [FR4012345].",
    "Si l'information demandée ne figure pas dans les données, dis-le clairement et propose de lancer une recherche ou de consulter l'avocat.",
    "Ne qualifie jamais un signalement de contrefaçon avérée : ce sont des pistes à vérifier par l'avocat.",
    "Réponds en français, de façon brève et structurée.",
  ].join("\n");
}

export async function POST(request: Request) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });

  const key = process.env.OPENROUTER_API_KEY;
  if (!key) return NextResponse.json({ error: "Assistant non configuré : ajoutez OPENROUTER_API_KEY dans Vercel." }, { status: 503 });

  const body = (await request.json().catch(() => null)) as { messages?: Message[]; data?: unknown } | null;
  const messages = (body?.messages ?? []).filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string").slice(-10);
  if (messages.length === 0 || messages[messages.length - 1].role !== "user") {
    return NextResponse.json({ error: "Question manquante." }, { status: 400 });
  }

  const client = currentClient();
  const context = JSON.stringify(body?.data ?? { rights: client.rights, findings: client.findings });
  if (context.length > MAX_CONTEXT_CHARS) return NextResponse.json({ error: "Portefeuille trop volumineux pour la démo." }, { status: 413 });

  const today = new Date().toLocaleDateString("fr-FR", { dateStyle: "long", timeZone: "Europe/Paris" });
  const res = await fetch("https://openrouter.ai/api/v1/chat/completions", {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Title": "Registre PI (demo)" },
    body: JSON.stringify({
      model: process.env.OPENROUTER_MODEL || DEFAULT_MODEL,
      temperature: 0.1,
      messages: [
        { role: "system", content: systemPrompt(client.name, today) },
        { role: "system", content: `Données du client (JSON) :\n${context}` },
        ...messages,
      ],
    }),
  });

  if (!res.ok) {
    const detail = await res.text().catch(() => "");
    return NextResponse.json({ error: `Erreur du fournisseur IA (${res.status}).`, detail: detail.slice(0, 300) }, { status: 502 });
  }
  const json = (await res.json()) as { choices?: { message?: { content?: string } }[] };
  const answer = json.choices?.[0]?.message?.content?.trim();
  return NextResponse.json({ answer: answer || "Je n'ai pas pu formuler de réponse." });
}
