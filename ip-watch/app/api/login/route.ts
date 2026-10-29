import { checkPassword, SESSION_COOKIE, signSession } from "@/lib/auth";
import { seeOther } from "@/lib/http";

export async function POST(request: Request) {
  const form = await request.formData();
  const password = String(form.get("password") ?? "");
  const role = form.get("role") === "client" ? "client" : "avocat";

  if (!checkPassword(password)) {
    return seeOther("/login?e=1");
  }

  const response = seeOther("/");
  response.cookies.set(SESSION_COOKIE, signSession(role), {
    httpOnly: true,
    sameSite: "lax",
    secure: process.env.NODE_ENV === "production",
    path: "/",
    maxAge: 60 * 60 * 24 * 30,
  });
  return response;
}
