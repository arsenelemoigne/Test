import { NextResponse } from "next/server";
import { checkPassword, SESSION_COOKIE, signSession } from "@/lib/auth";

export async function POST(request: Request) {
  const form = await request.formData();
  const password = String(form.get("password") ?? "");
  const role = form.get("role") === "client" ? "client" : "avocat";

  if (!checkPassword(password)) {
    return NextResponse.redirect(new URL("/login?e=1", request.url), 303);
  }

  const response = NextResponse.redirect(new URL("/", request.url), 303);
  response.cookies.set(SESSION_COOKIE, signSession(role), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return response;
}
