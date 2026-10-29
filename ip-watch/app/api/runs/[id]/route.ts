import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import * as store from "@/lib/store";

export const dynamic = "force-dynamic";

const STALE_MS = 20 * 60 * 1000;

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const r = await store.getRun(params.id);
  if (!r) return NextResponse.json({ error: "Recherche introuvable." }, { status: 404 });
  // A run left "running" by a restarted server is closed so the dashboard stops waiting.
  if (r.run.status === "running" && Date.now() - new Date(r.run.startedAt).getTime() > STALE_MS) {
    await store.updateRun(r.run.id, { status: "error", error: "Interrompue", finished_at: new Date().toISOString() });
    r.run.status = "error";
    r.run.error = "Interrompue";
  }
  return NextResponse.json(r);
}
