import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { runDiagnostic } from "@/lib/diagnostic";

export const dynamic = "force-dynamic";
export const maxDuration = 120;

export async function POST() {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  return NextResponse.json(await runDiagnostic());
}
