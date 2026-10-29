import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { onboard } from "@/lib/onboard";
import { runInTrace, startTrace } from "@/lib/trace";

export const dynamic = "force-dynamic";

// Re-imports the current client from INPI in the background; the page follows the trace.
export async function POST() {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const client = await currentClient();
  if (!client) return NextResponse.json({ error: "Aucun client." }, { status: 404 });
  const trace = await startTrace("import", `Actualisation INPI — ${client.name}`, client.id);
  void runInTrace(trace, async () => {
    try {
      const r = await onboard(client.company?.siren ? `${client.name} — SIREN ${client.company.siren}` : client.name, "app");
      await trace.finish("done", { summary: r.summary, slug: r.client.slug });
    } catch (e) {
      await trace.finish("error", { error: e instanceof Error ? e.message : String(e) });
    }
  });
  return NextResponse.json({ traceId: trace.id });
}
