import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { CLIENT_COOKIE, listClients } from "@/lib/data";

export async function POST(request: Request) {
  if (!currentRole()) return NextResponse.redirect(new URL("/login", request.url), 303);
  const form = await request.formData();
  const slug = String(form.get("slug") ?? "");
  const back = new URL(String(form.get("back") ?? "/"), request.url);
  const response = NextResponse.redirect(back, 303);
  if (listClients().some((c) => c.slug === slug)) {
    response.cookies.set(CLIENT_COOKIE, slug, { path: "/", sameSite: "lax", maxAge: 60 * 60 * 24 * 30 });
  }
  return response;
}
