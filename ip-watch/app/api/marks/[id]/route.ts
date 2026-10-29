import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { FIELD_LABELS } from "@/lib/labels";
import * as store from "@/lib/store";
import type { Right } from "@/lib/types";

const EDITABLE: (keyof Right)[] = ["name", "owner", "classes", "filingDate", "registrationDate", "expiryDate", "status", "notes"];

export async function PATCH(request: Request, { params }: { params: { id: string } }) {
  const role = currentRole();
  if (!role) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const client = await currentClient();
  if (!client) return NextResponse.json({ error: "Aucun client." }, { status: 404 });
  const body = (await request.json().catch(() => null)) as { field?: keyof Right | "favorite"; value?: unknown } | null;
  const id = decodeURIComponent(params.id);
  if (body?.field === "favorite") {
    await store.setFavorite(client.id, id, Boolean(body.value));
    return NextResponse.json({ ok: true });
  }
  if (!body?.field || !EDITABLE.includes(body.field)) return NextResponse.json({ error: "Champ non modifiable." }, { status: 400 });

  const r = await store.updateMarkField(client.id, id, body.field, body.value ?? null, role);
  if (!r) return NextResponse.json({ error: "Titre introuvable." }, { status: 404 });
  const show = (v: unknown) => (Array.isArray(v) ? v.join(", ") : v === null || v === undefined || v === "" ? "—" : String(v));
  await store.addEvent(client.id, "modification", `${r.mark.name} — ${FIELD_LABELS[body.field] ?? body.field}`, `${show(r.prev)} → ${show(body.value)} (${role === "avocat" ? "avocat" : "client"})`, { markId: id, field: body.field });
  return NextResponse.json({ ok: true });
}

// Only titles entered by hand can be deleted; imported titles follow the registry.
export async function DELETE(_request: Request, { params }: { params: { id: string } }) {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const client = await currentClient();
  if (!client) return NextResponse.json({ error: "Aucun client." }, { status: 404 });
  const id = decodeURIComponent(params.id);
  if (!id.startsWith("MAN-")) return NextResponse.json({ error: "Seuls les titres ajoutés à la main peuvent être supprimés." }, { status: 400 });
  await store.deleteMark(client.id, id);
  await store.addEvent(client.id, "modification", "Titre ajouté à la main supprimé", id, { markId: id });
  return NextResponse.json({ ok: true });
}
