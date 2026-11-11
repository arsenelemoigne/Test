import data from "@/data/ompi-la-haye.json";

// Official WIPO texts on the Hague system (international designs), quoted verbatim from the WIPO Guide,
// the Common Regulations and the Schedule of Fees, each with its official link. Refreshed by hand.
type Chunk = { titre: string; source: string; texte: string };
const CHUNKS = (data as { maj: string; chunks: Chunk[] }).chunks;

const norm = (s: string) => s.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "");

export function haguePassages(question: string, limit = 5): { maj: string; passages: Chunk[] } {
  const words = norm(question).split(/[^a-z0-9]+/).filter((w) => w.length > 3);
  const scored = CHUNKS.map((c) => {
    const hay = norm(`${c.titre} ${c.texte}`);
    return { c, score: words.reduce((n, w) => n + (hay.includes(w) ? 1 : 0), 0) + (/renouvel|grace|surtaxe|duree|expir/.test(norm(question)) && /renouvel|duree|regle 24|bareme/.test(norm(c.titre)) ? 2 : 0) };
  });
  const passages = scored.sort((a, b) => b.score - a.score).slice(0, limit).map((x) => x.c);
  return { maj: (data as { maj: string }).maj, passages };
}
