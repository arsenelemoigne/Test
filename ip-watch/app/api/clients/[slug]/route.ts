import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { CLIENT_COOKIE } from "@/lib/data";
import * as store from "@/lib/store";

export const dynamic = "force-dynamic";

export async function DELETE(_request: Request, { params }: { params: { slug: string } }) {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const client = await store.getClientBySlug(params.slug);
  if (!client) return NextResponse.json({ error: "Client introuvable." }, { status: 404 });
  await store.deleteClient(client.id);
  await store.addEvent(null, "note", `Client supprimé — ${client.name}`, "Titres, recherches, journaux et historique du client supprimés.", { slug: client.slug });
  const response = NextResponse.json({ ok: true });
  response.cookies.delete(CLIENT_COOKIE);
  return response;
}
