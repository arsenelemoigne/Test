import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import * as store from "@/lib/store";

export const dynamic = "force-dynamic";

export async function GET() {
  if (!currentRole()) return NextResponse.json({ error: "Non connecté." }, { status: 401 });
  const client = await currentClient();
  const all = await store.listConversations(client?.id ?? null);
  return NextResponse.json({
    conversations: all.map((c) => ({ id: c.id, title: c.title, updatedAt: c.updatedAt, count: c.messages.filter((m) => m.role === "user").length })),
  });
}
