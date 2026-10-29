import { NextResponse } from "next/server";

// Relative redirect: behind a proxy (Railway), request.url carries the internal host, so absolute URLs built from it break.
export function seeOther(path: string): NextResponse {
  const safe = path.startsWith("/") && !path.startsWith("//") ? path : "/";
  return new NextResponse(null, { status: 303, headers: { Location: safe } });
}
