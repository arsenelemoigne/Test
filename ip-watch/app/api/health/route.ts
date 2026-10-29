import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

// Reports which integrations are configured, never their values.
export function GET() {
  const names = [
    "DEMO_PASSWORD", "OPENROUTER_API_KEY", "LAWYER_EMAIL", "INPI_USERNAME", "INPI_PASSWORD", "SUPABASE_URL", "SUPABASE_SERVICE_ROLE_KEY",
    "SEARCHAPI_API_KEY", "URLSCAN_API_KEY", "RESEND_API_KEY", "RESEND_WEBHOOK_SECRET", "ASSISTANT_EMAIL", "PISTE_CLIENT_ID", "PISTE_CLIENT_SECRET",
  ];
  return NextResponse.json({
    ok: true,
    // Railway sets this to the deployed commit; lets us check which version is live.
    commit: process.env.RAILWAY_GIT_COMMIT_SHA?.slice(0, 7) ?? null,
    configured: Object.fromEntries(names.map((n) => [n, Boolean(process.env[n])])),
  });
}
