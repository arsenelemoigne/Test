import * as sb from "./supabase";

// Case files (uploaded images, thumbnails, PDF and Excel reports): the private Supabase bucket in production,
// memory for local development. Paths start with the client id so deleting a client deletes its files.
const BUCKET = "evidence";
const g = globalThis as unknown as { __ipwFiles?: Map<string, { bytes: Buffer; type: string }> };
const memory = (g.__ipwFiles ??= new Map());

export async function putFile(path: string, bytes: Buffer | Uint8Array | ArrayBuffer, type: string): Promise<void> {
  const buf = Buffer.isBuffer(bytes) ? bytes : Buffer.from(bytes instanceof ArrayBuffer ? new Uint8Array(bytes) : bytes);
  if (!sb.supabaseConfigured()) {
    memory.set(path, { bytes: buf, type });
    return;
  }
  const ab = buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
  await sb.uploadObject(BUCKET, path, ab, type);
}

export async function getFile(path: string): Promise<{ bytes: Buffer; type: string } | null> {
  if (!sb.supabaseConfigured()) return memory.get(path) ?? null;
  const res = await sb.downloadObject(BUCKET, path.split("/").map(encodeURIComponent).join("/"));
  if (!res.ok) return null;
  return { bytes: Buffer.from(await res.arrayBuffer()), type: res.headers.get("content-type") ?? "application/octet-stream" };
}

// A link that outside services can open for a few minutes; null when files only live in memory.
export async function temporaryUrl(path: string, seconds = 900): Promise<string | null> {
  if (!sb.supabaseConfigured()) return null;
  return sb.signedObjectUrl(BUCKET, path, seconds);
}

export function contentTypeOf(path: string): string {
  if (path.endsWith(".pdf")) return "application/pdf";
  if (path.endsWith(".xlsx")) return "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet";
  if (path.endsWith(".png")) return "image/png";
  if (path.endsWith(".webp")) return "image/webp";
  return "image/jpeg";
}
