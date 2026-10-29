import { NextResponse } from "next/server";
import { runAgent } from "@/lib/agent";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { llmConfigured } from "@/lib/llm";
import * as store from "@/lib/store";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

type Body = { messages?: { role: "user" | "assistant"; content: string }[]; focusMarkIds?: string[] };

export async function POST(request: Request) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  if (!llmConfigured()) return NextResponse.json({ error: "Assistant non configuré (OPENROUTER_API_KEY)." }, { status: 503 });
  const body = (await request.json().catch(() => null)) as Body | null;
  const messages = (body?.messages ?? []).filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string" && m.content.trim());
  if (messages.length === 0 || messages[messages.length - 1].role !== "user") return NextResponse.json({ error: "Question manquante." }, { status: 400 });

  const client = await currentClient();
  const marks = client ? await store.listMarks(client.id) : [];
  try {
    const out = await runAgent({ client, marks, messages, focusMarkIds: body?.focusMarkIds ?? [], trigger: "app" });
    return NextResponse.json(out);
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message.slice(0, 200) : "Erreur" }, { status: 502 });
  }
}
