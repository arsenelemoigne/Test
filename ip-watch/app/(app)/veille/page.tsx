"use client";

import { Fragment, useMemo, useState } from "react";
import { useData } from "@/components/EditsProvider";
import { EditableCell } from "@/components/EditableCell";
import { WorldMap } from "@/components/WorldMap";
import { countryName } from "@/lib/territories";
import { formatDateTime, formatPrice } from "@/lib/format";
import { KIND_LABELS, OPEN_STATUSES, STATUS_LABELS } from "@/lib/labels";
import type { FindingStatus } from "@/lib/types";

const STATUS_OPTIONS = Object.entries(STATUS_LABELS).map(([value, label]) => ({ value, label }));
const RELEVANCE_OPTIONS = [1, 2, 3, 4, 5].map((n) => ({ value: String(n), label: `${n} — ${"●".repeat(n)}${"○".repeat(5 - n)}` }));
const YES_NO = [
  { value: "oui", label: "Oui" },
  { value: "non", label: "Non" },
];

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const s = [...values].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

export default function WatchPage() {
  const { client, findings, findingEdits, editFinding, rights } = useData();
  const [status, setStatus] = useState<"ouverts" | "tous" | FindingStatus>("ouverts");
  const [country, setCountry] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);

  const rightName = (id: string) => rights.find((r) => r.id === id)?.name ?? id;

  const filtered = useMemo(
    () =>
      findings
        .filter((f) => (status === "tous" ? true : status === "ouverts" ? OPEN_STATUSES.includes(f.status) : f.status === status))
        .filter((f) => !country || f.country === country)
        .sort((a, b) => b.relevance - a.relevance || b.foundAt.localeCompare(a.foundAt)),
    [findings, status, country],
  );

  const byCountry = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of findings) if (f.country && f.status !== "ecarte") m.set(f.country, (m.get(f.country) ?? 0) + 1);
    return m;
  }, [findings]);

  const eurPrices = findings.filter((f) => f.price !== null && f.currency === "EUR" && f.status !== "ecarte").map((f) => f.price as number);
  const byPlatform = useMemo(() => {
    const m = new Map<string, number>();
    for (const f of findings) if (f.status !== "ecarte") m.set(f.platform, (m.get(f.platform) ?? 0) + 1);
    return Array.from(m.entries()).sort((a, b) => b[1] - a[1]);
  }, [findings]);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Veille</h1>
          <p className="text-sm text-neutral-600">
            Recherche mensuelle des usages suspects : places de marché, sites, noms de domaine, nouveaux dépôts. Dernier passage :{" "}
            {client.lastWatchAt ? formatDateTime(client.lastWatchAt) : "à venir"}.
          </p>
        </div>
        <div className="flex flex-wrap gap-1">
          {(["ouverts", "tous", "constat_demande", "ecarte"] as const).map((s) => (
            <button key={s} type="button" onClick={() => setStatus(s)} className={status === s ? "btn-solid" : "btn"}>
              {s === "ouverts" ? "À traiter" : s === "tous" ? "Tous" : STATUS_LABELS[s]}
            </button>
          ))}
        </div>
      </div>

      <section className="grid gap-6 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <h2 className="label mb-2">Pays des vendeurs {country && `— filtre : ${countryName(country)}`}</h2>
          <WorldMap counts={byCountry} selected={country} onSelect={setCountry} unit="signalement(s)" />
        </div>
        <div className="space-y-4 lg:col-span-2">
          <div className="grid grid-cols-3 gap-2">
            <div className="border border-black p-3">
              <div className="label">Prix min.</div>
              <div className="mt-1 font-mono text-lg">{eurPrices.length ? formatPrice(Math.min(...eurPrices), "EUR") : "—"}</div>
            </div>
            <div className="border border-black p-3">
              <div className="label">Médiane</div>
              <div className="mt-1 font-mono text-lg">{formatPrice(median(eurPrices), "EUR")}</div>
            </div>
            <div className="border border-black p-3">
              <div className="label">Prix max.</div>
              <div className="mt-1 font-mono text-lg">{eurPrices.length ? formatPrice(Math.max(...eurPrices), "EUR") : "—"}</div>
            </div>
          </div>
          <div>
            <h2 className="label">Par plateforme</h2>
            <ul className="mt-1 divide-y divide-neutral-200 border-y border-black text-sm">
              {byPlatform.map(([p, n]) => (
                <li key={p} className="flex justify-between py-1.5">
                  <span>{p}</span>
                  <span className="font-mono tabular-nums">{n}</span>
                </li>
              ))}
              {byPlatform.length === 0 && <li className="py-2 text-neutral-500">—</li>}
            </ul>
          </div>
          <p className="text-[11px] leading-relaxed text-neutral-500">
            Prix affichés par le vendeur, hors signalements écartés ; ils ne mesurent pas les ventes réelles. Statistiques de prix calculées sur les annonces en euros.
          </p>
        </div>
      </section>

      <div className="overflow-x-auto">
        <table className="table-base min-w-[1100px]">
          <thead>
            <tr>
              <th>Détecté</th>
              <th>Type</th>
              <th>Annonce</th>
              <th>Prix</th>
              <th>Pays</th>
              <th>Vendeur / société</th>
              <th>Pertinence</th>
              <th>Statut</th>
              <th>Constat</th>
              <th>Commentaire</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map((f) => (
              <Fragment key={f.id}>
                <tr>
                  <td className="whitespace-nowrap font-mono text-xs tabular-nums">{formatDateTime(f.foundAt)}</td>
                  <td className="whitespace-nowrap text-xs">
                    {KIND_LABELS[f.kind]}
                    <div className="text-neutral-500">{f.platform}</div>
                  </td>
                  <td className="min-w-[220px]">
                    <a href={f.url} target="_blank" rel="noreferrer" className="underline">
                      {f.title}
                    </a>
                    <div className="mt-0.5 text-[11px] text-neutral-500">Droits : {f.matchedRightIds.map(rightName).join(", ")}</div>
                    <button type="button" onClick={() => setOpen(open === f.id ? null : f.id)} className="mt-1 font-mono text-[11px] underline">
                      {open === f.id ? "Masquer la preuve" : "Dossier de preuve"}
                    </button>
                  </td>
                  <td className="whitespace-nowrap font-mono tabular-nums">{formatPrice(f.price, f.currency)}</td>
                  <td className="whitespace-nowrap text-xs">{f.country ? countryName(f.country) : "—"}</td>
                  <td className="min-w-[160px] text-xs">
                    {f.seller ?? "—"}
                    {f.company && (
                      <div className="text-neutral-500">
                        {f.company.name}
                        {f.company.id && ` · ${f.company.id}`}
                        {f.company.source && ` (${f.company.source})`}
                      </div>
                    )}
                  </td>
                  <td className="w-[120px]">
                    <EditableCell
                      display={`${"●".repeat(f.relevance)}${"○".repeat(5 - f.relevance)}`}
                      initial={String(f.relevance)}
                      options={RELEVANCE_OPTIONS}
                      edit={findingEdits[f.id]?.relevance}
                      className="font-mono"
                      onSave={(v) => editFinding(f.id, "relevance", Number(v))}
                    />
                  </td>
                  <td className="w-[150px]">
                    <EditableCell
                      display={STATUS_LABELS[f.status]}
                      initial={f.status}
                      options={STATUS_OPTIONS}
                      edit={findingEdits[f.id]?.status}
                      onSave={(v) => editFinding(f.id, "status", v)}
                    />
                  </td>
                  <td className="w-[90px]">
                    <EditableCell
                      display={f.constatRecommended ? "Recommandé" : "Non"}
                      initial={f.constatRecommended ? "oui" : "non"}
                      options={YES_NO}
                      edit={findingEdits[f.id]?.constatRecommended}
                      onSave={(v) => editFinding(f.id, "constatRecommended", v === "oui")}
                    />
                  </td>
                  <td className="min-w-[200px] text-xs">
                    <EditableCell display={f.comment || "—"} initial={f.comment} edit={findingEdits[f.id]?.comment} onSave={(v) => editFinding(f.id, "comment", v.trim())} />
                  </td>
                </tr>
                {open === f.id && (
                  <tr>
                    <td colSpan={10} className="bg-neutral-50">
                      <div className="grid gap-4 p-2 md:grid-cols-3">
                        <div className="md:col-span-1">
                          {f.evidence?.screenshotUrl ? (
                            // eslint-disable-next-line @next/next/no-img-element
                            <img src={f.evidence.screenshotUrl} alt={`Capture de ${f.url}`} className="w-full border border-black" />
                          ) : (
                            <div className="grid aspect-video place-items-center border border-dashed border-black font-mono text-[11px] text-neutral-500">
                              Capture non disponible
                            </div>
                          )}
                        </div>
                        <dl className="grid grid-cols-[140px_1fr] gap-x-3 gap-y-1 font-mono text-[11px] md:col-span-2">
                          <dt className="text-neutral-500">URL</dt>
                          <dd className="break-all">{f.url}</dd>
                          <dt className="text-neutral-500">Capturé le</dt>
                          <dd>{f.evidence ? formatDateTime(f.evidence.capturedAt) : "—"} (heure de Paris)</dd>
                          <dt className="text-neutral-500">Empreinte SHA-256</dt>
                          <dd className="break-all">{f.evidence?.sha256 ?? "—"}</dd>
                          <dt className="text-neutral-500">Analyse technique</dt>
                          <dd className="break-all">{f.evidence?.scanUrl ?? "—"}</dd>
                          <dt className="text-neutral-500">Valeur probatoire</dt>
                          <dd className="font-sans text-xs leading-relaxed">
                            Ce dossier documente l&apos;atteinte présumée et permet de vérifier que la capture n&apos;a pas été modifiée. Il ne remplace pas un procès-verbal
                            de constat sur Internet dressé par un commissaire de justice (norme NF Z67-147), à commander pour les cas qui le justifient.
                          </dd>
                        </dl>
                      </div>
                    </td>
                  </tr>
                )}
              </Fragment>
            ))}
            {filtered.length === 0 && (
              <tr>
                <td colSpan={10} className="py-6 text-center text-sm text-neutral-500">
                  Aucun signalement dans ce filtre.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <p className="text-[11px] text-neutral-500">
        Les signalements sont des pistes générées automatiquement puis triées par l&apos;avocat : aucun ne vaut constatation d&apos;une contrefaçon.
      </p>
    </div>
  );
}
