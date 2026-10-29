import { NextResponse } from "next/server";

export const dynamic = "force-dynamic";

// Reports which integrations are configured, never their values.
export function GET() {
  const set = (name: string) => Boolean(process.env[name]);
  return NextResponse.json({
    ok: true,
    configured: {
      DEMO_PASSWORD: set("DEMO_PASSWORD"),
      OPENROUTER_API_KEY: set("OPENROUTER_API_KEY"),
      LAWYER_EMAIL: set("LAWYER_EMAIL"),
      INPI_USERNAME: set("INPI_USERNAME"),
      INPI_PASSWORD: set("INPI_PASSWORD"),
      SEARCHAPI_API_KEY: set("SEARCHAPI_API_KEY"),
      URLSCAN_API_KEY: set("URLSCAN_API_KEY"),
      RESEND_API_KEY: set("RESEND_API_KEY"),
    },
  });
}
