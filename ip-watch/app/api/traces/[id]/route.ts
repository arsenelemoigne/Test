import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { getTrace } from "@/lib/trace";

export const dynamic = "force-dynamic";

export async function GET(_request: Request, { params }: { params: { id: string } }) {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const trace = await getTrace(params.id);
  if (!trace) return NextResponse.json({ error: "Trace introuvable." }, { status: 404 });
  return NextResponse.json(trace);
}
