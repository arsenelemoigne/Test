import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { CLIENT_COOKIE, listClients } from "@/lib/data";

export async function POST(request: Request) {
  if (!currentRole()) return NextResponse.json({ ok: false }, { status: 401 });
  const body = (await request.json().catch(() => null)) as { slug?: string } | null;
  const slug = String(body?.slug ?? "");
  if (!listClients().some((c) => c.slug === slug)) return NextResponse.json({ ok: false }, { status: 400 });
  const response = NextResponse.json({ ok: true });
  response.cookies.set(CLIENT_COOKIE, slug, { path: "/", sameSite: "lax", maxAge: 60 * 60 * 24 * 30 });
  return response;
}
