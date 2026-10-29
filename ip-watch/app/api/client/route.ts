import { currentRole } from "@/lib/auth";
import { CLIENT_COOKIE, listClients } from "@/lib/data";
import { seeOther } from "@/lib/http";

export async function POST(request: Request) {
  if (!currentRole()) return seeOther("/login");
  const form = await request.formData();
  const slug = String(form.get("slug") ?? "");
  const response = seeOther(String(form.get("back") ?? "/"));
  if (listClients().some((c) => c.slug === slug)) {
    response.cookies.set(CLIENT_COOKIE, slug, { path: "/", sameSite: "lax", maxAge: 60 * 60 * 24 * 30 });
  }
  return response;
}
