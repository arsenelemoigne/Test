import { randomUUID } from "crypto";
import { NextResponse } from "next/server";
import { currentRole } from "@/lib/auth";
import { countHits, InpiSession } from "@/lib/inpi";
import { insertRows, supabaseConfigured } from "@/lib/supabase";

export const dynamic = "force-dynamic";
export const maxDuration = 300;

const PAGE_SIZE = 100;
const MAX_PAGES = 5;

type Step = { kind: string; query?: string; position?: number; status: number | null; hits?: number | null; error?: string };

// Captures raw INPI answers for a holder into Supabase, for review before they are mapped into the portfolio.
export async function POST(request: Request) {
  if (currentRole() !== "avocat") return NextResponse.json({ error: "Réservé à l'avocat." }, { status: 403 });
  const username = process.env.INPI_USERNAME;
  const password = process.env.INPI_PASSWORD;
  if (!username || !password) return NextResponse.json({ error: "INPI_USERNAME / INPI_PASSWORD manquants." }, { status: 503 });
  if (!supabaseConfigured()) return NextResponse.json({ error: "SUPABASE_URL / SUPABASE_SERVICE_ROLE_KEY manquants." }, { status: 503 });

  const body = (await request.json().catch(() => null)) as { holder?: string } | null;
  const holder = (body?.holder ?? "").trim().toUpperCase().replace(/[\[\]]/g, "");
  if (!holder) return NextResponse.json({ error: "Nom du titulaire manquant." }, { status: 400 });

  const runId = randomUUID();
  const steps: Step[] = [];
  const save = async (step: Step, payload: unknown) => {
    steps.push(step);
    await insertRows("raw_imports", [
      { run_id: runId, source: "inpi", kind: step.kind, query: step.query ?? null, position: step.position ?? null, http_status: step.status, payload: payload ?? null, error: step.error ?? null },
    ]);
  };

  try {
    const inpi = new InpiSession();
    const loginStatus = await inpi.login(username, password);
    await save({ kind: "login", status: loginStatus }, null);
    if (loginStatus >= 400) {
      return NextResponse.json({ runId, steps, error: `Connexion INPI refusée (${loginStatus}).` }, { status: 502 });
    }

    const searches = [
      { kind: "marques", path: "/services/apidiffusion/api/marques/search", collections: ["FR", "EU", "WO"] },
      { kind: "modeles", path: "/services/apidiffusion/api/modeles/search", collections: ["FR", "WO"] },
    ];
    for (const s of searches) {
      const query = `[DEPOSANT=${holder}]`;
      for (let page = 0; page < MAX_PAGES; page++) {
        const position = page * PAGE_SIZE;
        const res = await inpi.request("POST", s.path, { collections: s.collections, query, fields: ["*"], position, size: PAGE_SIZE });
        const hits = countHits(res.data);
        await save({ kind: s.kind, query, position, status: res.status, hits }, res.data);
        if (res.status >= 400 || hits === null || hits < PAGE_SIZE) break;
      }
    }

    const meta = await inpi.request("GET", "/services/apidiffusion/api/marques/metadata");
    await save({ kind: "metadata", status: meta.status }, meta.data);

    return NextResponse.json({ runId, steps });
  } catch (e) {
    const error = e instanceof Error ? e.message : String(e);
    return NextResponse.json({ runId, steps, error }, { status: 500 });
  }
}
