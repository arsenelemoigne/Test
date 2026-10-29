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
  const body = (await request.json().catch(() => null)) as { field?: keyof Right; value?: unknown } | null;
  if (!body?.field || !EDITABLE.includes(body.field)) return NextResponse.json({ error: "Champ non modifiable." }, { status: 400 });

  const id = decodeURIComponent(params.id);
  const r = await store.updateMarkField(client.id, id, body.field, body.value ?? null, role);
  if (!r) return NextResponse.json({ error: "Titre introuvable." }, { status: 404 });
  const show = (v: unknown) => (Array.isArray(v) ? v.join(", ") : v === null || v === undefined || v === "" ? "—" : String(v));
  await store.addEvent(client.id, "modification", `${r.mark.name} — ${FIELD_LABELS[body.field] ?? body.field}`, `${show(r.prev)} → ${show(body.value)} (${role === "avocat" ? "avocat" : "client"})`, { markId: id, field: body.field });
  return NextResponse.json({ ok: true });
}
