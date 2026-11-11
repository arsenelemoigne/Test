// Visual search: an image (an ad, a campaign visual or a product) is compared with what already exists online.

// "anteriorite": before a launch, has something like this already been done?
// "contrefacon": is someone copying the client's product or visual?
export type VisualMode = "anteriorite" | "contrefacon";

export type VisualLevel = "tres_proche" | "proche" | "meme_theme" | "ecarte";

export const LEVEL_LABEL: Record<VisualLevel, string> = {
  tres_proche: "Très proche",
  proche: "Proche",
  meme_theme: "Même thème",
  ecarte: "Écarté",
};

export const MODE_LABEL: Record<VisualMode, string> = {
  anteriorite: "Antériorités avant lancement",
  contrefacon: "Copies du client",
};

export type VisualResultKind = "publicite" | "produit" | "image" | "page";

export type VisualResult = {
  // Stable number of the result within its search (1, 2, 3…).
  n: number;
  // Platform examined: "Google Vision", "Google Lens", "Meta Ad Library", "Google Shopping", "Google Images", "Web".
  source: string;
  kind: VisualResultKind;
  title: string;
  url: string;
  // Remote image as returned by the source; may be signed and expire.
  imageUrl: string | null;
  // Stored JPEG copy (files store path), used by the interface, the PDF and the Excel file.
  imagePath: string | null;
  advertiser: string | null;
  // ISO date (YYYY-MM-DD) when known, or a short text such as "diffusée depuis le 2025-11-02".
  date: string | null;
  country: string | null;
  price: number | null;
  currency: string | null;
  // The query or input that found the result.
  foundBy: string;
  level: VisualLevel | null;
  score: number | null;
  similar: string | null;
  differs: string | null;
  // Counterfeit mode only: copie, usage du signe, officiel, revente, sans rapport.
  category: string | null;
};

export type VisualAnalysis = {
  titre: string;
  type: "publicite" | "produit" | "autre";
  description: string;
  // E0 is the overall concept; E1… the visible elements (subject, setting, season, framing, text).
  elements: { id: string; label: string }[];
  motsCles: string[];
  keywordsEn: string[];
  marquesVisibles: string[];
  texteVisible: string | null;
};

export type VisualStep = { name: string; status: "running" | "done" | "skipped" | "error"; detail?: string; count?: number };

export type VisualSearch = {
  id: string;
  clientId: string | null;
  mode: VisualMode;
  origin: "depot" | "email";
  note: string | null;
  imagePath: string | null;
  status: "running" | "done" | "error";
  analysis: VisualAnalysis | null;
  steps: VisualStep[];
  results: VisualResult[];
  reportPdfPath: string | null;
  reportXlsxPath: string | null;
  traceId: string | null;
  error: string | null;
  createdAt: string;
  finishedAt: string | null;
};

export function levelCounts(results: VisualResult[]): Record<VisualLevel, number> {
  const out: Record<VisualLevel, number> = { tres_proche: 0, proche: 0, meme_theme: 0, ecarte: 0 };
  for (const r of results) if (r.level) out[r.level]++;
  return out;
}
