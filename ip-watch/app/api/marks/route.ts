import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { TYPE_LABELS } from "@/lib/labels";
import * as store from "@/lib/store";
import type { Office, Right, RightType } from "@/lib/types";

const TYPES = Object.keys(TYPE_LABELS) as RightType[];
const OFFICES: Office[] = ["FR", "EU", "WO", "EP", "AUTRE"];
const isoDate = (v: unknown) => (typeof v === "string" && /^\d{4}-\d{2}-\d{2}$/.test(v) ? v : null);

// Adds a title by hand (patent, domain name, proof of creation, …).
export async function POST(request: Request) {
  const role = currentRole();
  if (role !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const client = await currentClient();
  if (!client) return NextResponse.json({ error: "Aucun client." }, { status: 404 });
  const b = (await request.json().catch(() => null)) as Record<string, unknown> | null;
  const name = String(b?.name ?? "").trim();
  const type = TYPES.includes(b?.type as RightType) ? (b!.type as RightType) : null;
  if (!name || !type) return NextResponse.json({ error: "Nom et type obligatoires." }, { status: 400 });
  const office = OFFICES.includes(b?.office as Office) ? (b!.office as Office) : "AUTRE";
  const countries = String(b?.countries ?? "").toUpperCase().match(/\b[A-Z]{2}\b/g) ?? [];
  const right: Omit<Right, "id" | "origin"> = {
    type,
    office,
    number: String(b?.number ?? "").trim(),
    name,
    owner: String(b?.owner ?? client.company?.legalName ?? client.name).trim(),
    classes: (String(b?.classes ?? "").match(/\d{1,2}/g) ?? []).map(Number).filter((n) => n >= 1 && n <= 45),
    filingDate: isoDate(b?.filingDate),
    registrationDate: null,
    expiryDate: isoDate(b?.expiryDate),
    status: String(b?.status ?? "").trim() || "en vigueur",
    territories: office === "EU" ? [{ code: "EU" }] : countries.map((code) => ({ code })),
    sourceUrl: typeof b?.sourceUrl === "string" && /^https?:\/\//.test(b.sourceUrl) ? b.sourceUrl : null,
    notes: String(b?.notes ?? "").trim() || undefined,
  };
  const created = await store.addManualMark(client.id, right);
  await store.addEvent(client.id, "modification", `Titre ajouté à la main — ${created.name}`, `${TYPE_LABELS[type]}${created.number ? ` n° ${created.number}` : ""}.`, { markId: created.id });
  return NextResponse.json({ mark: created });
}
