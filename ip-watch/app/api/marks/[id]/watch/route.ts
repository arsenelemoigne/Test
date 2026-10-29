import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import * as store from "@/lib/store";
import { startWatch } from "@/lib/watch";

export const dynamic = "force-dynamic";

export async function POST(_request: Request, { params }: { params: { id: string } }) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const client = await currentClient();
  if (!client) return NextResponse.json({ error: "Aucun client." }, { status: 404 });
  const id = decodeURIComponent(params.id);
  const mark = (await store.listMarks(client.id)).find((m) => m.id === id);
  if (!mark) return NextResponse.json({ error: "Titre introuvable." }, { status: 404 });
  const run = await startWatch(client, mark, "bouton");
  return NextResponse.json({ runId: run.id });
}
