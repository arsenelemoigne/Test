import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { onboard } from "@/lib/onboard";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

// Re-imports the current client from INPI; the lawyer's edits are kept.
export async function POST() {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const client = await currentClient();
  if (!client) return NextResponse.json({ error: "Aucun client." }, { status: 404 });
  try {
    const r = await onboard(client.company?.siren ? `${client.name} — SIREN ${client.company.siren}` : client.name, "app");
    return NextResponse.json({ summary: r.summary });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message.slice(0, 300) : "Erreur" }, { status: 502 });
  }
}
