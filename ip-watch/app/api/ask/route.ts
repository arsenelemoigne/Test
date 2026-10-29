import { NextResponse } from "next/server";
import { runAgent } from "@/lib/agent";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { llmConfigured } from "@/lib/llm";
import * as store from "@/lib/store";
import { runInTrace, startTrace } from "@/lib/trace";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

type Body = {
  messages?: { role: "user" | "assistant"; content: string }[];
  focusMarkIds?: string[];
  // Saving: the conversation to append to, and the question as shown in the panel.
  conversationId?: string | null;
  display?: { content: string; marks?: string[] };
  traceId?: string;
};

export async function POST(request: Request) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  if (!llmConfigured()) return NextResponse.json({ error: "Assistant non configuré (OPENROUTER_API_KEY)." }, { status: 503 });
  const body = (await request.json().catch(() => null)) as Body | null;
  const messages = (body?.messages ?? []).filter((m) => (m.role === "user" || m.role === "assistant") && typeof m.content === "string" && m.content.trim());
  if (messages.length === 0 || messages[messages.length - 1].role !== "user") return NextResponse.json({ error: "Question manquante." }, { status: 400 });

  const client = await currentClient();
  const marks = client ? await store.listMarks(client.id) : [];
  const question = messages[messages.length - 1].content;
  const trace = await startTrace("assistant", question.slice(0, 120), client?.id ?? null, body?.traceId);
  try {
    const out = await runInTrace(trace, () => runAgent({ client, marks, messages, focusMarkIds: body?.focusMarkIds ?? [], trigger: "app" }));
    await trace.finish("done", { sources: out.sources });
    const shown = body?.display?.content?.trim() || question;
    const at = new Date().toISOString();
    const conversationId = await store
      .appendConversation(client?.id ?? null, body?.conversationId ?? null, shown, [
        { role: "user", content: shown, marks: body?.display?.marks ?? [], at },
        { role: "assistant", content: out.answer, sources: out.sources, traceId: trace.id, runId: out.runId, at: new Date().toISOString() },
      ])
      .catch(() => body?.conversationId ?? null);
    return NextResponse.json({ ...out, traceId: trace.id, steps: trace.steps, conversationId });
  } catch (e) {
    const error = e instanceof Error ? e.message.slice(0, 200) : "Erreur";
    await trace.finish("error", { error });
    return NextResponse.json({ error, traceId: trace.id, steps: trace.steps }, { status: 502 });
  }
}
