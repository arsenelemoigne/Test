// Google Cloud Vision, web detection: pages showing the same or a similar image, found from a product photo.
const ENDPOINT = "https://vision.googleapis.com/v1/images:annotate";

export function visionConfigured(): boolean {
  return Boolean(process.env.GOOGLE_VISION_API_KEY);
}

export type VisualMatch = { url: string; title: string; image: string | null; match: "identique" | "partielle" | "similaire" };

type WebDetection = {
  pagesWithMatchingImages?: { url?: string; pageTitle?: string; fullMatchingImages?: { url?: string }[]; partialMatchingImages?: { url?: string }[] }[];
  visuallySimilarImages?: { url?: string }[];
  bestGuessLabels?: { label?: string }[];
};

const stripTags = (s: string) => s.replace(/<[^>]+>/g, "").trim();

// The image is given by URL, or as bytes when it comes from a capture.
export async function webDetection(image: { url: string } | { bytes: ArrayBuffer }): Promise<{ pages: VisualMatch[]; similarImages: string[]; guess: string | null }> {
  const key = process.env.GOOGLE_VISION_API_KEY;
  if (!key) throw new Error("GOOGLE_VISION_API_KEY manquant");
  const source = "url" in image ? { source: { imageUri: image.url } } : { content: Buffer.from(image.bytes).toString("base64") };
  const res = await fetch(`${ENDPOINT}?key=${encodeURIComponent(key)}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ requests: [{ image: source, features: [{ type: "WEB_DETECTION", maxResults: 40 }] }] }),
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`Google Vision ${res.status}: ${text.slice(0, 200)}`);
  const json = JSON.parse(text) as { responses?: { webDetection?: WebDetection; error?: { message?: string } }[] };
  const r = json.responses?.[0];
  if (r?.error?.message) throw new Error(`Google Vision : ${r.error.message}`);
  const web = r?.webDetection ?? {};
  const pages: VisualMatch[] = (web.pagesWithMatchingImages ?? [])
    .filter((p) => p.url)
    .map((p) => ({
      url: p.url!,
      title: stripTags(p.pageTitle ?? p.url!),
      image: p.fullMatchingImages?.[0]?.url ?? p.partialMatchingImages?.[0]?.url ?? null,
      match: p.fullMatchingImages?.length ? "identique" : "partielle",
    }));
  return { pages, similarImages: (web.visuallySimilarImages ?? []).map((i) => i.url!).filter(Boolean).slice(0, 20), guess: web.bestGuessLabels?.[0]?.label ?? null };
}
