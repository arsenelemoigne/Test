import { currentClient } from "@/lib/data";
import { listEvents } from "@/lib/store";
import { formatDate } from "@/lib/format";
import { TraceLink } from "@/components/Trace";

export const dynamic = "force-dynamic";

const KIND_LABELS = { email: "email", import: "import", recherche: "recherche", modification: "modif.", note: "note", question: "question" } as const;

export default async function HistoryPage() {
  const client = await currentClient();
  const events = await listEvents(client?.id ?? null, 300);
  const byDay = new Map<string, typeof events>();
  for (const e of events) {
    const day = formatDate(e.createdAt);
    byDay.set(day, [...(byDay.get(day) ?? []), e]);
  }

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div>
        <h1 className="hand text-[28px] font-semibold leading-tight">Historique</h1>
        <p className="text-[13px] text-[color:var(--pencil)]">Emails reçus, imports, recherches et modifications{client ? ` — ${client.name}` : ""}.</p>
      </div>
      {events.length === 0 && <p className="hand text-[color:var(--pencil)]">Rien pour l&apos;instant.</p>}
      {Array.from(byDay.entries()).map(([day, items]) => (
        <section key={day}>
          <h2 className="hand text-[15px]">{day}</h2>
          <ul className="mt-1 divide-y divide-dashed divide-[color:var(--line)]">
            {items.map((e) => (
              <li key={e.id} className="flex gap-3 py-2.5">
                <span className="w-12 shrink-0 font-mono text-[11px] text-[color:var(--pencil)]">
                  {new Date(e.createdAt).toLocaleTimeString("fr-FR", { hour: "2-digit", minute: "2-digit", timeZone: "Europe/Paris" })}
                </span>
                <span className="chip h-fit shrink-0">{KIND_LABELS[e.kind] ?? e.kind}</span>
                <div className="min-w-0">
                  <div className="text-[13.5px]">{e.title}</div>
                  {e.summary && <p className="text-[12.5px] text-[color:var(--pencil)]">{e.summary}</p>}
                  {typeof e.details?.from === "string" && <p className="font-mono text-[10.5px] text-[color:var(--pencil)]">de {e.details.from}</p>}
                  <TraceLink id={typeof e.details?.traceId === "string" ? e.details.traceId : null} title={e.title} />
                </div>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </div>
  );
}
