import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { CLIENT_COOKIE } from "@/lib/data";
import { onboard } from "@/lib/onboard";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

// Same pipeline as an email to the assistant: free text in, client file filled from public registries.
export async function POST(request: Request) {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const body = (await request.json().catch(() => null)) as { text?: string } | null;
  const text = body?.text?.trim();
  if (!text) return NextResponse.json({ error: "Indiquez au moins le nom de la société." }, { status: 400 });
  try {
    const r = await onboard(text, "app");
    const response = NextResponse.json({ slug: r.client.slug, name: r.client.name, summary: r.summary });
    response.cookies.set(CLIENT_COOKIE, r.client.slug, { path: "/", sameSite: "lax", maxAge: 60 * 60 * 24 * 30 });
    return response;
  } catch (e) {
    return NextResponse.json({ error: e instanceof Error ? e.message.slice(0, 300) : "Erreur" }, { status: 502 });
  }
}
