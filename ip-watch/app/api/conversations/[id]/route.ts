import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import * as store from "@/lib/store";

export const dynamic = "force-dynamic";

const UUID = /^[0-9a-f-]{36}$/i;

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const conversation = UUID.test(params.id) ? await store.getConversation(params.id) : null;
  if (!conversation) return NextResponse.json({ error: "Conversation introuvable." }, { status: 404 });
  return NextResponse.json(conversation);
}

export async function DELETE(_request: Request, { params }: { params: { id: string } }) {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  if (UUID.test(params.id)) await store.deleteConversation(params.id);
  return NextResponse.json({ ok: true });
}
