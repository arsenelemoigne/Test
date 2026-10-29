import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { onboard } from "@/lib/onboard";
import { runInTrace, startTrace } from "@/lib/trace";

export const dynamic = "force-dynamic";

// Same pipeline as an email to the assistant, run in the background; the page follows the trace.
export async function POST(request: Request) {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const body = (await request.json().catch(() => null)) as { text?: string } | null;
  const text = body?.text?.trim();
  if (!text) return NextResponse.json({ error: "Indiquez au moins le nom de la société." }, { status: 400 });
  const trace = await startTrace("import", "Nouveau client", null);
  void runInTrace(trace, async () => {
    try {
      const r = await onboard(text, "app");
      await trace.finish("done", { summary: r.summary, slug: r.client.slug, name: r.client.name });
    } catch (e) {
      await trace.finish("error", { error: e instanceof Error ? e.message : String(e) });
    }
  });
  return NextResponse.json({ traceId: trace.id });
}
