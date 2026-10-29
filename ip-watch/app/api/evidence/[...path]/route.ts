import { currentRole } from "@/lib/auth";
import { downloadObject } from "@/lib/supabase";

export const dynamic = "force-dynamic";

// Screenshots stay in a private bucket and are only served to logged-in users.
export async function GET(_request: Request, { params }: { params: { path: string[] } }) {
  if (!currentRole()) return new Response("Non connecté.", { status: 401 });
  const path = params.path.map(encodeURIComponent).join("/");
  const res = await downloadObject("evidence", path);
  if (!res.ok) return new Response("Capture introuvable.", { status: 404 });
  return new Response(res.body, { headers: { "Content-Type": "image/png", "Cache-Control": "private, max-age=3600" } });
}
