"use client";

import rough from "roughjs";

type PathInfo = { d: string; stroke: string; strokeWidth: number; fill?: string };

const gen = rough.generator();
const INK = "#1c1c1a";
const PENCIL = "#6e6a63";

function box(x: number, y: number, w: number, h: number, seed: number, opts: { dashed?: boolean; fill?: string } = {}): PathInfo[] {
  return gen.toPaths(
    gen.rectangle(x, y, w, h, { seed, roughness: 1, bowing: 0.7, stroke: INK, strokeWidth: opts.dashed ? 1.2 : 1.3, strokeLineDash: opts.dashed ? [7, 6] : undefined, fill: opts.fill, fillStyle: "solid" }),
  ) as PathInfo[];
}

function arrow(x1: number, y1: number, x2: number, y2: number, seed: number): PathInfo[] {
  const angle = Math.atan2(y2 - y1, x2 - x1);
  const head = (a: number) => gen.line(x2, y2, x2 - 9 * Math.cos(angle + a), y2 - 9 * Math.sin(angle + a), { seed, roughness: 0.6, stroke: INK, strokeWidth: 1.3 });
  return [gen.line(x1, y1, x2, y2, { seed, roughness: 0.8, bowing: 1.2, stroke: INK, strokeWidth: 1.3 }), head(0.45), head(-0.45)].flatMap((d) => gen.toPaths(d) as PathInfo[]);
}

type Label = { x: number; y: number; text: string; size?: number; weight?: number; color?: string; anchor?: "start" | "middle" };

const T = (x: number, y: number, text: string, extra: Partial<Label> = {}): Label => ({ x, y, text, ...extra });

const ZONE_FILL = "#f1ede4";

const SHAPES: PathInfo[] = [
  // People and email entry
  ...box(20, 60, 150, 84, 1),
  ...box(20, 210, 150, 74, 2),
  ...box(20, 330, 150, 74, 3),
  // Application zone (Scaleway)
  ...box(205, 18, 470, 584, 4, { dashed: true }),
  ...box(235, 72, 410, 64, 5, { fill: "#fff" }),
  ...box(235, 176, 410, 84, 6, { fill: "#fff" }),
  ...box(235, 300, 410, 140, 7, { fill: "#fff" }),
  ...box(235, 482, 195, 92, 8, { fill: "#fff" }),
  ...box(450, 482, 195, 92, 9, { fill: "#fff" }),
  // Inference zone (Mistral)
  ...box(715, 18, 325, 150, 10, { dashed: true, fill: ZONE_FILL }),
  // Public sources and watch services
  ...box(715, 200, 325, 206, 11),
  ...box(715, 426, 325, 176, 12),
  // Arrows
  ...arrow(170, 102, 233, 102, 21),
  ...arrow(95, 284, 95, 328, 22),
  ...arrow(170, 367, 233, 240, 23),
  ...arrow(440, 136, 440, 174, 24),
  ...arrow(440, 260, 440, 298, 25),
  ...arrow(332, 440, 332, 480, 26),
  ...arrow(548, 440, 548, 480, 27),
  ...arrow(645, 205, 713, 110, 28),
  ...arrow(645, 340, 713, 315, 29),
  ...arrow(645, 410, 713, 500, 30),
];

const LABELS: Label[] = [
  T(95, 92, "Avocat · client", { anchor: "middle", size: 15, weight: 600 }),
  T(95, 114, "navigateur", { anchor: "middle", color: PENCIL }),
  T(95, 240, "Email", { anchor: "middle", size: 15, weight: 600 }),
  T(95, 262, "écrit ou transféré", { anchor: "middle", color: PENCIL }),
  T(95, 360, "Brevo", { anchor: "middle", size: 15, weight: 600 }),
  T(95, 382, "emails · France", { anchor: "middle", color: PENCIL }),

  T(222, 46, "APPLICATION — Scaleway · Paris (Île-de-France)", { size: 14, weight: 600 }),
  T(440, 100, "Interface web", { anchor: "middle", size: 15, weight: 600 }),
  T(440, 121, "tableau de bord · marques · conversations · historique", { anchor: "middle", color: PENCIL }),
  T(440, 206, "API & tâches de fond", { anchor: "middle", size: 15, weight: 600 }),
  T(440, 228, "import INPI · recherche de contrefaçons · tri des emails", { anchor: "middle", color: PENCIL }),
  T(440, 246, "assistant · journal par marque · déroulé", { anchor: "middle", color: PENCIL }),
  T(440, 328, "Outils de l'assistant (compétences)", { anchor: "middle", size: 15, weight: 600 }),
  T(255, 356, "legifrance_article · legifrance_recherche", { color: PENCIL }),
  T(255, 376, "judilibre_recherche · judilibre_decision", { color: PENCIL }),
  T(255, 396, "inpi_marques · entreprise · bodacc · droit_ue", { color: PENCIL }),
  T(255, 416, "recherche_contrefacon · resultats · web (repli)", { color: PENCIL }),
  T(332, 514, "PostgreSQL", { anchor: "middle", size: 15, weight: 600 }),
  T(332, 536, "registre · journaux", { anchor: "middle", color: PENCIL }),
  T(332, 554, "historique", { anchor: "middle", color: PENCIL }),
  T(548, 514, "Stockage objet", { anchor: "middle", size: 15, weight: 600 }),
  T(548, 536, "captures d'écran", { anchor: "middle", color: PENCIL }),
  T(548, 554, "empreintes SHA-256", { anchor: "middle", color: PENCIL }),

  T(732, 46, "INFÉRENCE — Mistral AI (UE)", { size: 14, weight: 600 }),
  T(877, 84, "modèle Mistral (texte et image)", { anchor: "middle", size: 15, weight: 600 }),
  T(877, 108, "réponses juridiques · tri des emails", { anchor: "middle", color: PENCIL }),
  T(877, 128, "plan de recherche · note de confiance", { anchor: "middle", color: PENCIL }),
  T(877, 148, "comparaison visuelle avec le produit", { anchor: "middle", color: PENCIL }),

  T(732, 226, "SOURCES PUBLIQUES", { size: 14, weight: 600 }),
  T(732, 246, "requêtes, pas de sous-traitance", { color: PENCIL }),
  T(732, 272, "Légifrance · Judilibre (PISTE)"),
  T(732, 294, "INPI — marques et dessins"),
  T(732, 316, "Annuaire des entreprises"),
  T(732, 338, "BODACC"),
  T(732, 360, "EUR-Lex · CURIA"),
  T(732, 382, "RDAP (AFNIC, Verisign)", { color: PENCIL }),

  T(732, 452, "VEILLE EN LIGNE", { size: 14, weight: 600 }),
  T(732, 476, "Recherche web — requêtes ciblées"),
  T(732, 498, "Google Lens · Shopping (SearchApi, US)"),
  T(732, 520, "Amazon · eBay (SearchApi, US)"),
  T(732, 542, "Google Cloud Vision — par image (UE)"),
  T(732, 564, "urlscan (DE) — captures, domaines"),
  T(732, 590, "envoyé : marque, produit, photo publique", { color: PENCIL }),
];

export function ArchitectureDiagram() {
  return (
    <div className="overflow-x-auto">
      <svg viewBox="0 0 1060 620" className="h-auto w-full min-w-[800px]" role="img" aria-label="Architecture cible">
        <rect x="205" y="18" width="470" height="584" fill={ZONE_FILL} opacity="0.6" />
        {SHAPES.map((p, i) => (
          <path key={i} d={p.d} stroke={p.stroke} strokeWidth={p.strokeWidth} fill={p.fill ?? "none"} strokeLinecap="round" />
        ))}
        {LABELS.map((l, i) => (
          <text
            key={i}
            x={l.x}
            y={l.y}
            textAnchor={l.anchor ?? "start"}
            fontFamily="'Shantell Sans', system-ui, sans-serif"
            fontSize={l.size ?? 13}
            fontWeight={l.weight ?? 400}
            fill={l.color ?? INK}
          >
            {l.text}
          </text>
        ))}
      </svg>
    </div>
  );
}
