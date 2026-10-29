"use client";

import { useMemo, useState } from "react";
import { useData } from "@/components/EditsProvider";
import { daysLabel, deadlineFor, DEADLINE_LABELS, type DeadlineState } from "@/lib/deadlines";
import { formatDate } from "@/lib/format";
import { REMINDER_OFFSETS_DAYS } from "@/lib/config";
import { TYPE_LABELS } from "@/lib/labels";

const HORIZONS = [
  { value: 365, label: "12 mois" },
  { value: 730, label: "24 mois" },
  { value: 3650, label: "Tout" },
];

export default function DeadlinesPage() {
  const { rights, settings } = useData();
  const [horizon, setHorizon] = useState(730);

  const rows = useMemo(
    () =>
      rights
        .map((r) => ({ right: r, d: deadlineFor(r) }))
        .filter((x) => x.d.expiry && x.d.state !== "expire" && (x.d.daysLeft ?? 0) <= horizon)
        .sort((a, b) => a.d.expiry!.getTime() - b.d.expiry!.getTime()),
    [rights, horizon],
  );

  const byYear = useMemo(() => {
    const groups = new Map<number, typeof rows>();
    for (const row of rows) {
      const y = row.d.expiry!.getUTCFullYear();
      groups.set(y, [...(groups.get(y) ?? []), row]);
    }
    return Array.from(groups.entries());
  }, [rows]);

  const emphasis = (s: DeadlineState) => (s === "a_venir" ? "chip" : "chip-solid");

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Échéances</h1>
          <p className="text-sm text-neutral-600">
            Renouvellement possible 6 mois avant l&apos;expiration, puis délai de grâce de 6 mois avec surtaxe. Rappels envoyés au cabinet à J-
            {REMINDER_OFFSETS_DAYS.join(", J-")}
            {settings.lawyerEmail ? ` (${settings.lawyerEmail})` : ""}.
          </p>
        </div>
        <div className="flex gap-1">
          {HORIZONS.map((h) => (
            <button key={h.value} type="button" onClick={() => setHorizon(h.value)} className={horizon === h.value ? "btn-solid" : "btn"}>
              {h.label}
            </button>
          ))}
        </div>
      </div>

      {byYear.length === 0 && <p className="border-y border-black py-4 text-sm text-neutral-500">Aucune échéance sur cette période.</p>}

      {byYear.map(([year, items]) => (
        <section key={year}>
          <h2 className="font-mono text-sm">{year}</h2>
          <div className="mt-2 overflow-x-auto">
            <table className="table-base min-w-[900px]">
              <thead>
                <tr>
                  <th>Expiration</th>
                  <th>Titre</th>
                  <th>Type</th>
                  <th>Office</th>
                  <th>Ouverture</th>
                  <th>Fin du délai de grâce</th>
                  <th>Rappels au cabinet</th>
                  <th>État</th>
                </tr>
              </thead>
              <tbody>
                {items.map(({ right, d }) => (
                  <tr key={right.id}>
                    <td className="whitespace-nowrap font-mono tabular-nums">
                      {formatDate(d.expiry)}
                      {d.computed && <span title="Date calculée à partir de la date de dépôt" className="ml-1 text-neutral-500">*</span>}
                      <div className="text-[11px] text-neutral-500">{daysLabel(d)}</div>
                    </td>
                    <td>
                      <div>{right.name}</div>
                      <div className="font-mono text-[11px] text-neutral-500">{right.id}</div>
                    </td>
                    <td className="whitespace-nowrap">{TYPE_LABELS[right.type]}</td>
                    <td className="font-mono text-xs">{right.office}</td>
                    <td className="whitespace-nowrap font-mono text-xs tabular-nums">{formatDate(d.windowOpens)}</td>
                    <td className="whitespace-nowrap font-mono text-xs tabular-nums">{formatDate(d.graceEnds)}</td>
                    <td className="whitespace-nowrap font-mono text-[11px] tabular-nums">
                      {d.reminders.map((r, i) => (
                        <div key={i} className={r < new Date() ? "text-neutral-400 line-through" : undefined}>
                          {formatDate(r)}
                        </div>
                      ))}
                    </td>
                    <td>
                      <span className={emphasis(d.state)}>{DEADLINE_LABELS[d.state]}</span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}
      <p className="font-mono text-[11px] text-neutral-500">* date calculée à partir de la date de dépôt, à confirmer sur le registre.</p>
    </div>
  );
}
