"use client";

import { useMemo, useState } from "react";
import { deadlineFor, DEADLINE_LABELS, daysLabel, type Deadline } from "@/lib/deadlines";
import { formatDate } from "@/lib/format";
import { OFFICE_LABELS_SHORT } from "@/lib/labels";
import type { Mark } from "@/lib/types";

const HORIZONS = [
  { months: 12, label: "12 mois" },
  { months: 24, label: "24 mois" },
  { months: 0, label: "tout" },
];

const monthLabel = (key: string) => {
  const [y, m] = key.split("-").map(Number);
  return new Date(Date.UTC(y, m - 1, 1)).toLocaleDateString("fr-FR", { month: "long", year: "numeric", timeZone: "UTC" });
};

// Upcoming expiries on a vertical line, month by month; each title opens its card.
export function DeadlineTimeline({ marks, onOpen }: { marks: Mark[]; onOpen: (m: Mark) => void }) {
  const [horizon, setHorizon] = useState(12);

  const groups = useMemo(() => {
    const limit = horizon ? new Date(Date.now() + horizon * 30.5 * 86400000) : null;
    const rows = marks
      .map((mark) => ({ mark, d: deadlineFor(mark) }))
      .filter((x): x is { mark: Mark; d: Deadline & { expiry: Date } } => Boolean(x.d.expiry) && x.d.state !== "expire" && (!limit || x.d.expiry! <= limit))
      .sort((a, b) => a.d.expiry.getTime() - b.d.expiry.getTime());
    const byMonth = new Map<string, typeof rows>();
    for (const r of rows) {
      const key = r.d.expiry.toISOString().slice(0, 7);
      byMonth.set(key, [...(byMonth.get(key) ?? []), r]);
    }
    return Array.from(byMonth.entries());
  }, [marks, horizon]);

  const count = groups.reduce((n, [, items]) => n + items.length, 0);

  return (
    <section>
      <div className="mb-2 flex items-baseline justify-between gap-2">
        <h2 className="hand text-[16px]">
          Prochaines échéances <span className="text-[13px] text-[color:var(--pencil)]">({count})</span>
        </h2>
        <span className="flex gap-1">
          {HORIZONS.map((h) => (
            <button key={h.label} type="button" onClick={() => setHorizon(h.months)} className={horizon === h.months ? "chip-ink" : "chip"}>
              {h.label}
            </button>
          ))}
        </span>
      </div>
      {groups.length === 0 ? (
        <p className="py-3 text-[13px] text-[color:var(--pencil)]">Aucune échéance sur cette période.</p>
      ) : (
        <div className="max-h-[520px] overflow-y-auto pl-3 pr-1">
        <ol className="relative border-l-[1.5px] border-[color:var(--ink)] pl-5">
          {groups.map(([key, items]) => (
            <li key={key} className="relative pb-4">
              <span className="absolute -left-[27px] top-1 h-3 w-3 rounded-full border-[1.5px] border-[color:var(--ink)] bg-[color:var(--paper)]" aria-hidden />
              <div className="hand text-[14px] capitalize">{monthLabel(key)}</div>
              <ul className="mt-1 space-y-1">
                {items.map(({ mark, d }) => (
                  <li key={mark.id}>
                    <button type="button" onClick={() => onOpen(mark)} className="flex w-full items-baseline justify-between gap-3 rounded px-1.5 py-1 text-left hover:bg-[color:var(--faint)]">
                      <span className="min-w-0">
                        <span className="block truncate text-[13.5px]">
                          {mark.favorite && "★ "}
                          {mark.name}
                        </span>
                        <span className="block whitespace-nowrap font-mono text-[10.5px] text-[color:var(--pencil)]">
                          {OFFICE_LABELS_SHORT[mark.office]} · {mark.id}
                        </span>
                      </span>
                      <span className="shrink-0 text-right">
                        <span className="hand block text-[13px] tabular-nums">{formatDate(d.expiry)}</span>
                        {d.state === "a_venir" ? (
                          <span className="text-[11px] text-[color:var(--pencil)]">{daysLabel(d)}</span>
                        ) : (
                          <span className="chip-ink">{DEADLINE_LABELS[d.state]}</span>
                        )}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </li>
          ))}
        </ol>
        </div>
      )}
    </section>
  );
}
