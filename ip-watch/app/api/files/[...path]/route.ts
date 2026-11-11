import { currentRole } from "@/lib/auth";
import { contentTypeOf, getFile } from "@/lib/files";

export const dynamic = "force-dynamic";

// Files of the case (images, reports) stay private and are only served to logged-in users.
export async function GET(_request: Request, { params }: { params: { path: string[] } }) {
  if (!currentRole()) return new Response("Non connecté.", { status: 401 });
  const path = params.path.map((p) => decodeURIComponent(p)).join("/");
  if (path.includes("..")) return new Response("Chemin invalide.", { status: 400 });
  const file = await getFile(path).catch(() => null);
  if (!file) return new Response("Fichier introuvable.", { status: 404 });
  const type = contentTypeOf(path);
  const name = path.split("/").slice(-3).join("-");
  const disposition = type === "application/pdf" || type.startsWith("image/") ? "inline" : "attachment";
  return new Response(new Uint8Array(file.bytes), {
    headers: { "Content-Type": type, "Content-Disposition": `${disposition}; filename="${name}"`, "Cache-Control": "private, max-age=3600" },
  });
}
