"use client";

import Link from "next/link";
import { useMemo } from "react";
import { useData } from "@/components/EditsProvider";
import { WorldMap } from "@/components/WorldMap";
import { daysLabel, deadlineFor, DEADLINE_LABELS } from "@/lib/deadlines";
import { countByCountry } from "@/lib/territories";
import { formatDate, formatDateTime } from "@/lib/format";
import { OPEN_STATUSES } from "@/lib/labels";

function Stat({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="border border-black p-4">
      <div className="label">{label}</div>
      <div className="mt-2 font-mono text-2xl tabular-nums sm:text-3xl">{value}</div>
      {sub && <div className="mt-1 text-xs text-neutral-500">{sub}</div>}
    </div>
  );
}

export default function OverviewPage() {
  const { client, rights, findings } = useData();

  const deadlines = useMemo(
    () =>
      rights
        .map((r) => ({ right: r, d: deadlineFor(r) }))
        .filter((x) => x.d.expiry && x.d.state !== "expire")
        .sort((a, b) => a.d.expiry!.getTime() - b.d.expiry!.getTime()),
    [rights],
  );
  const counts = useMemo(() => countByCountry(rights), [rights]);
  const open = findings.filter((f) => OPEN_STATUSES.includes(f.status));
  const next = deadlines[0];
  const urgent = deadlines.filter((x) => x.d.state !== "a_venir").length;

  return (
    <div className="space-y-8">
      <div>
        <div className="label">Client</div>
        <h1 className="text-2xl font-semibold tracking-tight">{client.name}</h1>
      </div>

      <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Titres suivis" value={String(rights.length)} sub={`${rights.filter((r) => r.type === "marque").length} marques · ${rights.filter((r) => r.type === "dessin").length} dessins`} />
        <Stat label="Pays couverts" value={String(counts.size)} sub="UE comptée pays par pays" />
        <Stat label="Prochaine échéance" value={next ? formatDate(next.d.expiry) : "—"} sub={next ? `${next.right.name} · ${daysLabel(next.d)}` : undefined} />
        <Stat label="Signalements à traiter" value={String(open.length)} sub={`${urgent} renouvellement(s) ouvert(s)`} />
      </section>

      <section className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <div className="mb-2 flex items-baseline justify-between">
            <h2 className="label">Où les droits produisent effet</h2>
            <Link href="/carte" className="font-mono text-xs underline">Carte détaillée</Link>
          </div>
          <WorldMap counts={counts} unit="titre(s)" />
        </div>
        <div className="lg:col-span-2">
          <div className="mb-2 flex items-baseline justify-between">
            <h2 className="label">Prochaines échéances</h2>
            <Link href="/echeances" className="font-mono text-xs underline">Tout voir</Link>
          </div>
          <ul className="divide-y divide-neutral-200 border-y border-black">
            {deadlines.slice(0, 7).map(({ right, d }) => (
              <li key={right.id} className="flex items-center justify-between gap-3 py-2">
                <div className="min-w-0">
                  <div className="truncate text-sm">{right.name}</div>
                  <div className="font-mono text-[11px] text-neutral-500">{right.id}</div>
                </div>
                <div className="text-right">
                  <div className="font-mono text-sm tabular-nums">{formatDate(d.expiry)}</div>
                  <div className={d.state === "a_venir" ? "font-mono text-[11px] text-neutral-500" : "chip-solid"}>{DEADLINE_LABELS[d.state]}</div>
                </div>
              </li>
            ))}
            {deadlines.length === 0 && <li className="py-4 text-sm text-neutral-500">Aucune échéance connue.</li>}
          </ul>
        </div>
      </section>

      <section>
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="label">Veille — dernier passage {client.lastWatchAt ? formatDateTime(client.lastWatchAt) : "à venir"}</h2>
          <Link href="/veille" className="font-mono text-xs underline">Journal de veille</Link>
        </div>
        {open.length === 0 ? (
          <p className="border-y border-black py-4 text-sm text-neutral-500">Aucun signalement à traiter.</p>
        ) : (
          <ul className="divide-y divide-neutral-200 border-y border-black">
            {open.slice(0, 5).map((f) => (
              <li key={f.id} className="flex items-center justify-between gap-3 py-2 text-sm">
                <span className="truncate">{f.title}</span>
                <span className="font-mono text-xs">{"●".repeat(f.relevance)}{"○".repeat(5 - f.relevance)}</span>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
