"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useState } from "react";
import { AssistantPanel } from "./AssistantPanel";
import { ClientActions } from "./ClientActions";
import { LiveRun } from "./LiveRun";
import { RoughBox } from "./RoughBox";
import { WorldMap } from "./WorldMap";
import { daysLabel, deadlineFor, DEADLINE_LABELS, isActiveRight } from "@/lib/deadlines";
import { countByCountry } from "@/lib/territories";
import { formatDate, formatDateTime } from "@/lib/format";
import type { Client, Finding, Mark, WatchRun } from "@/lib/types";

type Props = { client: Client; marks: Mark[]; runs: WatchRun[]; initialRunId: string | null };

function Stat({ label, value, sub, seed }: { label: string; value: string; sub?: string; seed: number }) {
  return (
    <RoughBox seed={seed}>
      <div className="px-4 py-3">
        <div className="label truncate">{label}</div>
        <div className="hand mt-1 text-[21px] leading-tight tabular-nums sm:text-[26px]">{value}</div>
        {sub && <div className="mt-0.5 truncate text-[12px] text-[color:var(--pencil)]">{sub}</div>}
      </div>
    </RoughBox>
  );
}

export function Dashboard({ client, marks, runs, initialRunId }: Props) {
  const router = useRouter();
  const [runId, setRunId] = useState<string | null>(initialRunId);
  const [live, setLive] = useState<{ run: WatchRun; findings: Finding[] } | null>(null);
  const [view, setView] = useState<"coverage" | "run">(initialRunId ? "run" : "coverage");
  const [country, setCountry] = useState<string | null>(null);
  const [allDeadlines, setAllDeadlines] = useState(false);

  // Follows a search while it runs; stops polling once it is finished.
  useEffect(() => {
    if (!runId) return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const res = await fetch(`/api/runs/${runId}`, { cache: "no-store" }).catch(() => null);
      if (stop) return;
      if (res?.ok) {
        const json = (await res.json()) as { run: WatchRun; findings: Finding[] };
        setLive(json);
        if (json.run.status !== "running") {
          router.refresh();
          return;
        }
      }
      timer = setTimeout(tick, 2000);
    };
    tick();
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [runId, router]);

  // Keeps the dashboard current while a client is still being imported.
  useEffect(() => {
    if (client.status !== "importing" && client.status !== "pending") return;
    const t = setInterval(() => router.refresh(), 5000);
    return () => clearInterval(t);
  }, [client.status, router]);

  const openRun = useCallback((id: string) => {
    setRunId(id);
    setView("run");
    setCountry(null);
  }, []);

  const deadlines = useMemo(
    () =>
      marks
        .map((m) => ({ mark: m, d: deadlineFor(m) }))
        .filter((x) => x.d.expiry && x.d.state !== "expire")
        .sort((a, b) => a.d.expiry!.getTime() - b.d.expiry!.getTime()),
    [marks],
  );
  const active = useMemo(() => marks.filter((m) => isActiveRight(m)), [marks]);
  const coverage = useMemo(() => countByCountry(active), [active]);
  const hits = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of live?.findings ?? []) if (f.country) m.set(f.country, (m.get(f.country) ?? 0) + 1);
    return m;
  }, [live]);

  const next = deadlines[0];
  const marksCount = active.filter((m) => m.type === "marque").length;
  const designsCount = active.length - marksCount;
  const lastRun = runs[0];
  const showingRun = view === "run" && live;

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_400px]">
      <div className="min-w-0 space-y-6">
        <div className="flex flex-wrap items-baseline gap-x-3">
          <h1 className="hand text-[28px] font-semibold leading-tight">{client.name}</h1>
          {client.company && (
            <a href={client.company.url} target="_blank" rel="noreferrer" className="btn-quiet text-[12px]">
              SIREN {client.company.siren}
            </a>
          )}
          {client.status === "importing" && <span className="chip">import INPI en cours…</span>}
          {client.status === "error" && <span className="chip">import échoué : {client.error}</span>}
          <span className="ml-auto">
            <ClientActions />
          </span>
        </div>

        <section className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Stat seed={11} label="marques en vigueur" value={String(marksCount)} sub={`${designsCount} dessins & modèles · ${marks.length - active.length} titres expirés`} />
          <Stat seed={12} label="pays couverts" value={String(coverage.size)} sub="UE comptée pays par pays" />
          <Stat seed={13} label="prochaine échéance" value={next ? formatDate(next.d.expiry) : "—"} sub={next ? `${next.mark.name} · ${daysLabel(next.d)}` : undefined} />
          <Stat
            seed={14}
            label="signaux forts (dernière veille)"
            value={lastRun ? String(lastRun.stats?.high ?? 0) : "—"}
            sub={lastRun ? `${lastRun.markName} · ${formatDate(lastRun.startedAt)}` : "depuis Marques ou l'assistant"}
          />
        </section>

        <RoughBox seed={21}>
          <div className="p-4">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <button type="button" onClick={() => setView("coverage")} className={view === "coverage" || !live ? "chip-ink" : "chip"}>
                couverture
              </button>
              {live && (
                <button type="button" onClick={() => setView("run")} className={view === "run" ? "chip-ink" : "chip"}>
                  recherche « {live.run.markName} »
                </button>
              )}
              <span className="label ml-auto">{showingRun ? "pays des résultats — cliquez pour filtrer" : "où les titres produisent effet"}</span>
            </div>
            <WorldMap
              mode={showingRun ? "hits" : "coverage"}
              counts={showingRun ? hits : coverage}
              unit={showingRun ? "résultat(s)" : "titre(s)"}
              selected={showingRun ? country : null}
              onSelect={showingRun ? setCountry : undefined}
            />
            {showingRun && (
              <div className="mt-4 border-t border-dashed border-[color:var(--line)] pt-4">
                <LiveRun run={live.run} findings={live.findings} country={country} onCountry={setCountry} onClose={() => setView("coverage")} />
              </div>
            )}
          </div>
        </RoughBox>

        <div className="grid gap-6 lg:grid-cols-2">
          <section>
            <div className="mb-1 flex items-baseline justify-between">
              <h2 className="hand text-[16px]">Prochaines échéances</h2>
              {deadlines.length > 6 && (
                <button type="button" className="btn-quiet text-[12px]" onClick={() => setAllDeadlines((v) => !v)}>
                  {allDeadlines ? "réduire" : `tout voir (${deadlines.length})`}
                </button>
              )}
            </div>
            <ul className="divide-y divide-dashed divide-[color:var(--line)]">
              {deadlines.slice(0, allDeadlines ? undefined : 6).map(({ mark, d }) => (
                <li key={mark.id} className="flex items-center justify-between gap-3 py-2">
                  <div className="min-w-0">
                    <div className="truncate text-[13.5px]">{mark.name}</div>
                    <div className="font-mono text-[10.5px] text-[color:var(--pencil)]">{mark.id}</div>
                  </div>
                  <div className="text-right">
                    <div className="hand text-[14px] tabular-nums">{formatDate(d.expiry)}</div>
                    <div className={d.state === "a_venir" ? "text-[11px] text-[color:var(--pencil)]" : "chip-ink"}>{d.state === "a_venir" ? daysLabel(d) : DEADLINE_LABELS[d.state]}</div>
                  </div>
                </li>
              ))}
              {deadlines.length === 0 && <li className="py-3 text-[13px] text-[color:var(--pencil)]">Aucune échéance connue.</li>}
            </ul>
          </section>

          <section>
            <div className="mb-1 flex items-baseline justify-between">
              <h2 className="hand text-[16px]">Recherches de contrefaçons</h2>
              <Link href="/marques" className="btn-quiet text-[12px]">
                lancer depuis Marques
              </Link>
            </div>
            <ul className="divide-y divide-dashed divide-[color:var(--line)]">
              {runs.slice(0, 6).map((r) => (
                <li key={r.id}>
                  <button type="button" onClick={() => openRun(r.id)} className={`flex w-full items-center justify-between gap-3 py-2 text-left ${r.id === runId && view === "run" ? "font-semibold" : ""}`}>
                    <span className="min-w-0">
                      <span className="block truncate text-[13.5px]">{r.markName}</span>
                      <span className="text-[11px] text-[color:var(--pencil)]">
                        {formatDateTime(r.startedAt)} · {r.trigger}
                      </span>
                    </span>
                    <span className="hand text-[14px] tabular-nums">
                      {r.status === "running" ? "en cours…" : `${r.stats?.total ?? 0} résultats · ${r.stats?.high ?? 0} forts`}
                    </span>
                  </button>
                </li>
              ))}
              {runs.length === 0 && <li className="py-3 text-[13px] text-[color:var(--pencil)]">Aucune recherche pour l&apos;instant.</li>}
            </ul>
          </section>
        </div>
      </div>

      <AssistantPanel marks={marks} onRunStarted={openRun} className="xl:sticky xl:top-4 xl:h-[calc(100vh-2rem)] min-h-[520px]" />
    </div>
  );
}
