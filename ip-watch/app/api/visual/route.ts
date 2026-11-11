import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import * as store from "@/lib/store";
import { startVisualSearch } from "@/lib/visual";
import type { VisualMode } from "@/lib/visual-types";

export const dynamic = "force-dynamic";

const MAX_BYTES = 12_000_000;

// Starts a visual search from an uploaded image; the page then follows it by id.
export async function POST(request: Request) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const form = await request.formData().catch(() => null);
  const file = form?.get("image");
  if (!form || !(file instanceof Blob) || file.size === 0) return NextResponse.json({ error: "Aucune image reçue." }, { status: 400 });
  if (file.size > MAX_BYTES) return NextResponse.json({ error: "Image trop lourde (12 Mo maximum)." }, { status: 400 });
  if (file.type && !file.type.startsWith("image/")) return NextResponse.json({ error: "Le fichier n'est pas une image." }, { status: 400 });
  const mode: VisualMode = form.get("mode") === "contrefacon" ? "contrefacon" : "anteriorite";
  const note = String(form.get("note") ?? "").trim().slice(0, 1000) || null;
  const client = await currentClient();
  try {
    const search = await startVisualSearch({ client, mode, note, image: Buffer.from(await file.arrayBuffer()), origin: "depot" });
    return NextResponse.json({ id: search.id });
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message : String(e) }, { status: 500 });
  }
}

export async function GET() {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const client = await currentClient();
  const searches = await store.listVisualSearches(client?.id ?? null, 50);
  return NextResponse.json({ searches });
}
