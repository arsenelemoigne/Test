import { NextResponse } from "next/server";
import { handleInboundEmail, verifyWebhook } from "@/lib/email";
import { logCall } from "@/lib/log";

export const dynamic = "force-dynamic";

// Resend webhook for received emails. Answers at once and processes the email in the background.
export async function POST(request: Request) {
  const payload = await request.text();
  let event: { type?: string; data?: { email_id?: string } };
  try {
    event = verifyWebhook(payload, request.headers) as typeof event;
  } catch (e) {
    void logCall("email", "webhook-rejected", null, e instanceof Error ? e.message : String(e), 401, 0);
    return NextResponse.json({ error: "Signature invalide." }, { status: 401 });
  }
  const emailId = event.data?.email_id;
  if (event.type === "email.received" && emailId) {
    void handleInboundEmail(emailId).catch((e) => logCall("email", "inbound-failed", { emailId }, e instanceof Error ? e.message : String(e), 500, 0));
  }
  return NextResponse.json({ ok: true });
}
