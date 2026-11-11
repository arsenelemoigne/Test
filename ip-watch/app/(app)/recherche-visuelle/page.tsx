import { assistantEmail } from "@/lib/config";
import { currentClient } from "@/lib/data";
import * as store from "@/lib/store";
import { VisualSearchView, type VisualListItem } from "@/components/VisualSearch";
import { levelCounts } from "@/lib/visual-types";

export const dynamic = "force-dynamic";

export default async function VisualSearchPage({ searchParams }: { searchParams: { id?: string } }) {
  const client = await currentClient();
  const [searches, selected] = await Promise.all([
    store.listVisualSearches(client?.id ?? null, 60).catch(() => []),
    searchParams.id ? store.getVisualSearch(searchParams.id).catch(() => null) : Promise.resolve(null),
  ]);
  const items: VisualListItem[] = searches.map((s) => ({
    id: s.id,
    mode: s.mode,
    origin: s.origin,
    status: s.status,
    createdAt: s.createdAt,
    title: s.analysis?.titre ?? (s.status === "running" ? "Analyse en cours…" : "Image déposée"),
    imagePath: s.imagePath,
    total: s.results.length,
    counts: levelCounts(s.results),
    reportPdfPath: s.reportPdfPath,
    reportXlsxPath: s.reportXlsxPath,
  }));
  return <VisualSearchView key={selected?.id ?? "new"} clientName={client?.name ?? null} searches={items} initial={selected} assistantEmail={assistantEmail()} />;
}
