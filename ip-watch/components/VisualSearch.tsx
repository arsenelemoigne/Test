"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { DerouleDrawer } from "./Deroule";
import { RoughBox } from "./RoughBox";
import { formatDateTime, formatPrice } from "@/lib/format";
import { VISUAL_STEPS } from "@/lib/visual-steps";
import { LEVEL_LABEL, levelCounts, MODE_LABEL, type VisualLevel, type VisualMode, type VisualResult, type VisualSearch, type VisualStep } from "@/lib/visual-types";

export type VisualListItem = {
  id: string;
  mode: VisualMode;
  origin: "depot" | "email";
  status: VisualSearch["status"];
  createdAt: string;
  title: string;
  imagePath: string | null;
  total: number;
  counts: Record<VisualLevel, number>;
  reportPdfPath: string | null;
  reportXlsxPath: string | null;
};

const LEVEL_COLOR: Record<VisualLevel, string> = { tres_proche: "#c0392b", proche: "#e67e22", meme_theme: "#2f6f9f", ecarte: "#9aa0a6" };
const LEVEL_ORDER: VisualLevel[] = ["tres_proche", "proche", "meme_theme", "ecarte"];
const STEP_ICON: Record<VisualStep["status"], string> = { running: "…", done: "✓", skipped: "–", error: "×" };
const fileUrl = (path: string) => `/api/files/${path.split("/").map(encodeURIComponent).join("/")}`;

type Props = { clientName: string | null; searches: VisualListItem[]; initial: VisualSearch | null; assistantEmail: string | null };

export function VisualSearchView({ clientName, searches, initial, assistantEmail }: Props) {
  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="hand text-[28px] font-semibold leading-tight">Recherche visuelle</h1>
          <p className="text-[13px] text-[color:var(--pencil)]">
            Une publicité, un visuel de campagne ou une photo de produit : PI Pilot cherche ce qui existe déjà en ligne{clientName ? ` — dossier ${clientName}` : ""}.
          </p>
        </div>
        {initial && (
          <Link href="/recherche-visuelle" className="btn">
            + nouvelle recherche
          </Link>
        )}
      </div>
      {initial ? <RunView initial={initial} /> : <NewSearch searches={searches} assistantEmail={assistantEmail} />}
    </div>
  );
}

// ---------- New search: drop an image, pick the question, launch.

function NewSearch({ searches, assistantEmail }: { searches: VisualListItem[]; assistantEmail: string | null }) {
  const router = useRouter();
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [mode, setMode] = useState<VisualMode>("anteriorite");
  const [note, setNote] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const pick = useCallback((f: File | null | undefined) => {
    if (!f) return;
    if (!f.type.startsWith("image/")) {
      setError("Choisissez une image (JPEG, PNG, WebP…).");
      return;
    }
    setError(null);
    setFile(f);
    setPreview((old) => {
      if (old) URL.revokeObjectURL(old);
      return URL.createObjectURL(f);
    });
  }, []);

  // An image copied to the clipboard can be pasted anywhere on the page.
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const item = Array.from(e.clipboardData?.items ?? []).find((i) => i.type.startsWith("image/"));
      if (item) pick(item.getAsFile());
    };
    window.addEventListener("paste", onPaste);
    return () => window.removeEventListener("paste", onPaste);
  }, [pick]);

  const launch = async () => {
    if (!file) return;
    setBusy(true);
    setError(null);
    const form = new FormData();
    form.append("image", file);
    form.append("mode", mode);
    form.append("note", note);
    const res = await fetch("/api/visual", { method: "POST", body: form }).catch(() => null);
    const json = (await res?.json().catch(() => null)) as { id?: string; error?: string } | null;
    if (!res?.ok || !json?.id) {
      setBusy(false);
      setError(json?.error ?? "La recherche n'a pas pu démarrer.");
      return;
    }
    router.push(`/recherche-visuelle?id=${json.id}`);
  };

  return (
    <div className="grid gap-6 xl:grid-cols-[minmax(0,1fr)_420px]">
      <RoughBox seed={41}>
        <div className="space-y-5 p-5">
          <div
            onClick={() => input.current?.click()}
            onDragOver={(e) => {
              e.preventDefault();
              setOver(true);
            }}
            onDragLeave={() => setOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setOver(false);
              pick(e.dataTransfer.files?.[0]);
            }}
            className={`wobble flex min-h-[260px] cursor-pointer flex-col items-center justify-center gap-3 border-dashed p-4 text-center transition-colors ${over ? "bg-white" : "bg-[color:var(--faint)]"}`}
          >
            {preview ? (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={preview} alt="Image à analyser" className="max-h-[320px] max-w-full rounded-sm object-contain shadow-sm" />
            ) : (
              <>
                <span className="hand text-[20px]">Déposez une image ici</span>
                <span className="text-[13px] text-[color:var(--pencil)]">ou cliquez pour la choisir — vous pouvez aussi la coller (Ctrl / Cmd + V)</span>
              </>
            )}
            {file && <span className="text-[12px] text-[color:var(--pencil)]">{file.name} · changer d&apos;image</span>}
            <input ref={input} type="file" accept="image/*" className="hidden" onChange={(e) => pick(e.target.files?.[0])} />
          </div>

          <div>
            <div className="label mb-2">ce que l&apos;on cherche</div>
            <div className="grid gap-3 sm:grid-cols-2">
              <ModeCard active={mode === "anteriorite"} onClick={() => setMode("anteriorite")} title="Campagnes antérieures" text="Avant de lancer une publicité : une campagne semblable a-t-elle déjà été diffusée ?" />
              <ModeCard active={mode === "contrefacon"} onClick={() => setMode("contrefacon")} title="Contrefaçons" text="Des produits en vente copient-ils celui du client ?" />
            </div>
          </div>

          <label className="block">
            <span className="label">précisions (facultatif)</span>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              rows={2}
              placeholder="Produit, slogan, pays de lancement, marque qui inquiète…"
              className="field mt-1 w-full resize-y"
            />
          </label>

          <div className="flex flex-wrap items-center gap-3">
            <button type="button" className="btn-ink px-5 py-1.5 text-[15px]" disabled={!file || busy} onClick={launch}>
              {busy ? "lancement…" : "Lancer la recherche"}
            </button>
            {assistantEmail && <span className="text-[12.5px] text-[color:var(--pencil)]">ou envoyez l&apos;image par email à <span className="hand text-[color:var(--ink)]">{assistantEmail}</span> : le rapport revient par email.</span>}
          </div>
          {error && <p className="text-[13px] text-red-700">{error}</p>}
          <p className="text-[11.5px] text-[color:var(--pencil)]">
            Campagnes : bibliothèque publicitaire de Meta, presse et sites de campagnes, Google Images, Lens et Vision. Contrefaçons : Google Lens, Vision, Shopping et Images. L&apos;IA compare chaque visuel à votre image. Repérage exploratoire, à vérifier par l&apos;avocat.
          </p>
        </div>
      </RoughBox>

      <section>
        <div className="flex items-baseline justify-between gap-2">
          <h2 className="hand text-[17px]">Recherches du dossier</h2>
          {searches.length > 0 && (
            <a href="/api/visual/registre" className="btn-quiet text-[12px]">
              registre Excel
            </a>
          )}
        </div>
        {searches.length === 0 && <p className="hand mt-3 text-[13px] text-[color:var(--pencil)]">Aucune recherche pour l&apos;instant.</p>}
        <ul className="mt-2 divide-y divide-dashed divide-[color:var(--line)]">
          {searches.map((s) => (
            <li key={s.id} className="flex gap-3 py-2.5">
              <Link href={`/recherche-visuelle?id=${s.id}`} className="shrink-0">
                {s.imagePath ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img src={fileUrl(s.imagePath)} alt="" className="h-14 w-14 rounded-sm border border-[color:var(--line)] object-cover" />
                ) : (
                  <span className="block h-14 w-14 rounded-sm bg-[color:var(--faint)]" />
                )}
              </Link>
              <div className="min-w-0 flex-1">
                <Link href={`/recherche-visuelle?id=${s.id}`} className="line-clamp-1 text-[13.5px] underline decoration-[color:var(--line)] underline-offset-2 hover:decoration-[color:var(--ink)]">
                  {s.title}
                </Link>
                <p className="text-[11.5px] text-[color:var(--pencil)]">
                  {formatDateTime(s.createdAt)} · {MODE_LABEL[s.mode]}
                  {s.origin === "email" ? " · par email" : ""}
                  {s.status === "running" ? " · en cours" : s.status === "error" ? " · interrompue" : ""}
                </p>
                <p className="mt-0.5 flex flex-wrap items-center gap-x-2.5 gap-y-0.5 text-[11.5px]">
                  {LEVEL_ORDER.slice(0, 3).map((l) => (
                    <span key={l} className="inline-flex items-center gap-1">
                      <span className="inline-block h-2 w-2 rounded-full" style={{ background: LEVEL_COLOR[l] }} />
                      {s.counts[l]} {LEVEL_LABEL[l].toLowerCase()}
                    </span>
                  ))}
                  {s.reportPdfPath && (
                    <a href={fileUrl(s.reportPdfPath)} target="_blank" rel="noreferrer" className="btn-quiet text-[11.5px]">
                      PDF
                    </a>
                  )}
                  {s.reportXlsxPath && (
                    <a href={fileUrl(s.reportXlsxPath)} className="btn-quiet text-[11.5px]">
                      Excel
                    </a>
                  )}
                </p>
              </div>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}

function ModeCard({ active, onClick, title, text }: { active: boolean; onClick: () => void; title: string; text: string }) {
  return (
    <button type="button" onClick={onClick} className={`wobble px-3.5 py-2.5 text-left transition-colors ${active ? "bg-[color:var(--ink)] text-[color:var(--paper)]" : "hover:bg-white"}`}>
      <span className="hand block text-[15px]">
        {active ? "● " : "○ "}
        {title}
      </span>
      <span className={`block text-[12.5px] leading-snug ${active ? "text-[color:var(--paper)] opacity-85" : "text-[color:var(--pencil)]"}`}>{text}</span>
    </button>
  );
}

// ---------- A search: what the AI sees, where it looked, what it found.

function RunView({ initial }: { initial: VisualSearch }) {
  const router = useRouter();
  const [search, setSearch] = useState(initial);
  const [deroule, setDeroule] = useState(false);
  const [showDismissed, setShowDismissed] = useState(false);
  const [only, setOnly] = useState<VisualLevel | null>(null);

  useEffect(() => {
    if (search.status !== "running") return;
    let stop = false;
    let timer: ReturnType<typeof setTimeout>;
    const tick = async () => {
      const res = await fetch(`/api/visual/${search.id}`, { cache: "no-store" }).catch(() => null);
      if (stop) return;
      if (res?.ok) {
        const json = (await res.json()) as { search: VisualSearch };
        setSearch(json.search);
        if (json.search.status !== "running") {
          router.refresh();
          return;
        }
      }
      timer = setTimeout(tick, 1800);
    };
    timer = setTimeout(tick, 1200);
    return () => {
      stop = true;
      clearTimeout(timer);
    };
  }, [search.id, search.status, router]);

  const counts = useMemo(() => levelCounts(search.results), [search.results]);
  const sorted = useMemo(
    () => [...search.results].sort((a, b) => (a.level ? LEVEL_ORDER.indexOf(a.level) : 4) - (b.level ? LEVEL_ORDER.indexOf(b.level) : 4) || (b.score ?? -1) - (a.score ?? -1) || a.n - b.n),
    [search.results],
  );
  const shown = sorted.filter((r) => (only ? r.level === only : showDismissed || r.level !== "ecarte"));
  const steps = mergeSteps(search.steps, search.status);
  const running = search.status === "running";
  const title = search.analysis?.titre ?? (running ? "analyse en cours…" : "Image déposée");

  return (
    <div className="space-y-5">
      <div className="grid items-start gap-5 lg:grid-cols-[300px_minmax(0,1fr)]">
        <RoughBox seed={51} className="lg:sticky lg:top-4">
          <div className="space-y-3 p-4">
            {search.imagePath && (
              // eslint-disable-next-line @next/next/no-img-element
              <img src={fileUrl(search.imagePath)} alt="Image analysée" className="w-full rounded-sm border border-[color:var(--line)] object-contain" />
            )}
            <div>
              <div className="label">image analysée</div>
              <h2 className="hand text-[19px] leading-snug">{title}</h2>
              <p className="mt-1 flex flex-wrap gap-1.5">
                <span className="chip-ink">{MODE_LABEL[search.mode]}</span>
                {search.origin === "email" && <span className="chip">reçue par email</span>}
              </p>
              <p className="mt-1 text-[11.5px] text-[color:var(--pencil)]">{formatDateTime(search.createdAt)}</p>
            </div>
            {search.analysis?.description && (
              <div>
                <div className="label">ce que l&apos;IA voit</div>
                <p className="text-[13px] leading-snug">{search.analysis.description}</p>
              </div>
            )}
            {search.analysis && search.analysis.elements.length > 0 && (
              <ul className="space-y-1 text-[12.5px]">
                {search.analysis.elements.map((e) => (
                  <li key={e.id} className="flex gap-2">
                    <span className="hand w-6 shrink-0 text-[color:var(--pencil)]">{e.id}</span>
                    <span className="min-w-0">{e.label}</span>
                  </li>
                ))}
              </ul>
            )}
            {search.note && <p className="text-[12px] text-[color:var(--pencil)]">Précisions : {search.note}</p>}
          </div>
        </RoughBox>

        <div className="space-y-4">
          <RoughBox seed={52}>
            <div className="p-4">
              <div className="flex flex-wrap items-baseline justify-between gap-2">
                <h3 className="hand text-[16px]">
                  {running ? (
                    <>
                      PI Pilot cherche
                      <span className="dots">
                        <span>.</span>
                        <span>.</span>
                        <span>.</span>
                      </span>
                    </>
                  ) : search.status === "error" ? (
                    `Recherche interrompue${search.error ? ` : ${search.error}` : ""}`
                  ) : (
                    `Recherche terminée · ${search.results.length} résultat(s) examiné(s)`
                  )}
                </h3>
                <span className="flex items-center gap-3">
                  {search.reportPdfPath && (
                    <a href={fileUrl(search.reportPdfPath)} target="_blank" rel="noreferrer" className="btn text-[12.5px]">
                      rapport PDF
                    </a>
                  )}
                  {search.reportXlsxPath && (
                    <a href={fileUrl(search.reportXlsxPath)} className="btn text-[12.5px]">
                      tableur Excel
                    </a>
                  )}
                  {search.traceId && (
                    <button type="button" className="btn-quiet text-[12px]" onClick={() => setDeroule(true)}>
                      déroulé
                    </button>
                  )}
                </span>
              </div>
              <ol className="mt-2 grid gap-x-6 gap-y-1 text-[12.5px] md:grid-cols-2">
                {steps.map((s) => (
                  <li key={s.name} className={`flex gap-2 ${s.status === "pending" ? "opacity-40" : ""}`}>
                    <span className={`w-3 shrink-0 ${s.status === "error" ? "text-red-700" : ""}`}>{s.status === "pending" ? "·" : STEP_ICON[s.status]}</span>
                    <span className="min-w-0">
                      <span className="hand text-[13px]">{s.name}</span>
                      {s.detail && <span className={s.status === "error" ? "text-red-700" : "text-[color:var(--pencil)]"}> · {s.detail}</span>}
                    </span>
                  </li>
                ))}
              </ol>
            </div>
          </RoughBox>

          <div className="flex flex-wrap items-center gap-2">
            {LEVEL_ORDER.map((l) => (
              <button
                key={l}
                type="button"
                onClick={() => setOnly((o) => (o === l ? null : l))}
                className={`hand inline-flex items-center gap-1.5 rounded-full border px-3 py-0.5 text-[13px] ${only === l ? "text-white" : ""}`}
                style={{ borderColor: LEVEL_COLOR[l], background: only === l ? LEVEL_COLOR[l] : "transparent" }}
              >
                <span className="inline-block h-2.5 w-2.5 rounded-full" style={{ background: only === l ? "#fff" : LEVEL_COLOR[l] }} />
                {LEVEL_LABEL[l]} <span className="tabular-nums">{counts[l]}</span>
              </button>
            ))}
            {!only && counts.ecarte > 0 && (
              <button type="button" className="btn-quiet text-[12px]" onClick={() => setShowDismissed((v) => !v)}>
                {showDismissed ? "masquer" : "afficher"} les écartés
              </button>
            )}
          </div>

          <ul className="grid grid-cols-2 gap-3 md:grid-cols-3 2xl:grid-cols-4">
            {shown.map((r) => (
              <ResultCard key={`${r.n}-${r.url}`} r={r} />
            ))}
          </ul>
          {shown.length === 0 && <p className="hand text-[13px] text-[color:var(--pencil)]">{running ? "les premiers résultats arrivent…" : "aucun résultat à afficher"}</p>}
          <p className="text-[11.5px] text-[color:var(--pencil)]">Repérage exploratoire : rapprochements proposés par l&apos;IA, à vérifier par l&apos;avocat ; ni recherche exhaustive ni constat de contrefaçon.</p>
        </div>
      </div>
      {deroule && search.traceId && <DerouleDrawer traceId={search.traceId} title={`Recherche visuelle — ${title}`} onClose={() => setDeroule(false)} />}
    </div>
  );
}

function ResultCard({ r }: { r: VisualResult }) {
  const color = r.level ? LEVEL_COLOR[r.level] : "#c4c8cc";
  return (
    <li className="flex flex-col overflow-hidden rounded-[6px] border bg-white" style={{ borderColor: color, borderWidth: r.level === "tres_proche" || r.level === "proche" ? 2 : 1 }}>
      <a href={r.url} target="_blank" rel="noreferrer" className="relative block aspect-[4/3] bg-[color:var(--faint)]">
        {r.imagePath ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img src={fileUrl(r.imagePath)} alt="" className="h-full w-full object-cover" loading="lazy" />
        ) : (
          <span className="hand flex h-full items-center justify-center px-3 text-center text-[12px] text-[color:var(--pencil)]">{r.kind === "page" ? "page web" : "image indisponible"}</span>
        )}
        <span className="hand absolute left-1.5 top-1.5 rounded-full px-2 py-px text-[11.5px] text-white shadow-sm" style={{ background: color }}>
          {r.level ? LEVEL_LABEL[r.level] : "évaluation…"}
          {r.score !== null ? ` · ${r.score}` : ""}
        </span>
        <span className="absolute right-1.5 top-1.5 rounded bg-white/90 px-1.5 text-[10.5px] text-[color:var(--pencil)]">n°{r.n}</span>
      </a>
      <div className="flex flex-1 flex-col gap-1 p-2.5">
        <p className="truncate text-[11px] text-[color:var(--pencil)]">
          {[r.source, r.advertiser, r.date, r.price !== null ? formatPrice(r.price, r.currency) : null].filter(Boolean).join(" · ")}
        </p>
        <a href={r.url} target="_blank" rel="noreferrer" className="line-clamp-2 text-[12.5px] leading-snug underline decoration-[color:var(--line)] underline-offset-2 hover:decoration-[color:var(--ink)]">
          {r.title}
        </a>
        {r.category && <span className="chip w-fit text-[11px]">{r.category}</span>}
        {r.similar && <p className="text-[11.5px] leading-snug"><span className="text-[color:var(--pencil)]">commun : </span>{r.similar}</p>}
        {r.differs && <p className="text-[11.5px] leading-snug"><span className="text-[color:var(--pencil)]">diffère : </span>{r.differs}</p>}
      </div>
    </li>
  );
}

type ShownStep = { name: string; status: VisualStep["status"] | "pending"; detail?: string };

// Planned steps appear greyed until they start; extra steps (e.g. the web fallback) are kept.
function mergeSteps(steps: VisualStep[], status: VisualSearch["status"]): ShownStep[] {
  const byName = new Map(steps.map((s) => [s.name, s]));
  const out: ShownStep[] = VISUAL_STEPS.map((name) => byName.get(name) ?? { name, status: status === "running" ? "pending" : "skipped" });
  for (const s of steps) if (!(VISUAL_STEPS as readonly string[]).includes(s.name)) out.splice(out.length - 3, 0, s);
  return out;
}
