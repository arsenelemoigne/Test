import { currentRole } from "@/lib/auth";
import { currentClient } from "@/lib/data";
import { appUrl } from "@/lib/email";
import * as store from "@/lib/store";
import { buildSearchRegister } from "@/lib/visual-report";

export const dynamic = "force-dynamic";

// The client's register: one line per visual search, with links to each report.
export async function GET() {
  if (!currentRole()) return new Response("Non connecté.", { status: 401 });
  const client = await currentClient();
  const searches = await store.listVisualSearches(client?.id ?? null, 1000);
  const xlsx = await buildSearchRegister(searches, { clientName: client?.name ?? "Sans client", appUrl: appUrl() });
  const name = `registre-recherches-visuelles-${client?.slug ?? "sans-client"}.xlsx`;
  return new Response(new Uint8Array(xlsx), {
    headers: { "Content-Type": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "Content-Disposition": `attachment; filename="${name}"` },
  });
}
