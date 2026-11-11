import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import * as store from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const search = await store.getVisualSearch(params.id).catch(() => null);
  if (!search) return NextResponse.json({ error: "Recherche introuvable." }, { status: 404 });
  return NextResponse.json({ search });
}
