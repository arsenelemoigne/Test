import { NextResponse } from "next/server";
import { checkPassword, SESSION_COOKIE, signSession } from "@/lib/auth";

// Returns JSON and lets the page navigate, so nothing depends on how the hosting proxy rewrites redirects.
export async function POST(request: Request) {
  const body = (await request.json().catch(() => null)) as { password?: string; role?: string } | null;
  const role = body?.role === "client" ? "client" : "avocat";

  if (!checkPassword(String(body?.password ?? ""))) {
    return NextResponse.json({ ok: false, error: "Mot de passe incorrect." }, { status: 401 });
  }

  const response = NextResponse.json({ ok: true });
  response.cookies.set(SESSION_COOKIE, signSession(role), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return response;
}
