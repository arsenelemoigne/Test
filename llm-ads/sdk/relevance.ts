import type { CatalogOffer } from '../shared/types.ts';

/**
 * Scores how well each offer fits a piece of text. Runs inside the publisher's
 * infrastructure: whatever implementation you plug in must not send the text to the marketplace.
 */
export interface RelevanceScorer {
  /** Called whenever a new catalog arrives. */
  index(offers: CatalogOffer[]): Promise<void>;
  /** Relevance in [0, 1] per offer id. */
  score(text: string): Promise<Map<string, number>>;
}

const STOPWORDS = new Set(
  (
    'a an and are as at be but by can do for from get have how i im in is it me my need of on or ' +
    'please should so some that the this to want was what where which who why will with would you your ' +
    'le la les un une des du de et ou pour avec dans sur est je tu il elle nous vous mon ma mes'
  ).split(' '),
);

export function tokenize(text: string): string[] {
  return text
    .normalize('NFD')
    .replace(/\p{Diacritic}/gu, '')
    .toLowerCase()
    .split(/[^\p{L}\p{N}]+/u)
    .filter((t) => t.length > 1 && !STOPWORDS.has(t))
    .map((t) => (t.length > 3 && t.endsWith('s') && !t.endsWith('ss') ? t.slice(0, -1) : t));
}

/** Text describing an offer, keywords counted twice because advertisers chose them for targeting. */
export function offerText(offer: CatalogOffer): string {
  return [
    offer.keywords.join(' '),
    offer.keywords.join(' '),
    offer.title,
    offer.brand,
    offer.claims.join(' '),
    offer.categories.join(' ').replaceAll('.', ' '),
  ].join(' ');
}

/** TF-IDF cosine similarity. Zero dependencies, English/French stopwords, no synonyms. */
export class LexicalScorer implements RelevanceScorer {
  private idf = new Map<string, number>();
  private unknownIdf = 1;
  private vectors = new Map<string, Map<string, number>>();

  async index(offers: CatalogOffer[]): Promise<void> {
    const docs = offers.map((o) => [o.id, tokenize(offerText(o))] as const);
    const df = new Map<string, number>();
    for (const [, tokens] of docs) for (const t of new Set(tokens)) df.set(t, (df.get(t) ?? 0) + 1);
    const n = Math.max(docs.length, 1);
    this.idf = new Map([...df].map(([t, d]) => [t, Math.log(1 + n / d)]));
    // Words no offer uses still count, so an off-topic sentence dilutes a single matching word.
    this.unknownIdf = Math.log(1 + n);
    this.vectors = new Map(docs.map(([id, tokens]) => [id, this.vectorize(tokens)]));
  }

  async score(text: string): Promise<Map<string, number>> {
    const query = this.vectorize(tokenize(text));
    const scores = new Map<string, number>();
    for (const [id, vector] of this.vectors) scores.set(id, sparseCosine(query, vector));
    return scores;
  }

  private vectorize(tokens: string[]): Map<string, number> {
    const v = new Map<string, number>();
    for (const t of tokens) v.set(t, (v.get(t) ?? 0) + (this.idf.get(t) ?? this.unknownIdf));
    return v;
  }
}

/**
 * Semantic matching with any embedding function (multilingual, handles synonyms).
 * Keep `embed` local or on your own inference server to preserve the privacy promise.
 */
export class EmbeddingScorer implements RelevanceScorer {
  private readonly embed: (texts: string[]) => Promise<number[][]>;
  private vectors = new Map<string, number[]>();

  constructor(embed: (texts: string[]) => Promise<number[][]>) {
    this.embed = embed;
  }

  async index(offers: CatalogOffer[]): Promise<void> {
    const vectors = offers.length ? await this.embed(offers.map(offerText)) : [];
    this.vectors = new Map(offers.map((o, i) => [o.id, vectors[i]]));
  }

  async score(text: string): Promise<Map<string, number>> {
    const [query] = await this.embed([text]);
    const scores = new Map<string, number>();
    for (const [id, vector] of this.vectors) scores.set(id, Math.max(0, denseCosine(query, vector)));
    return scores;
  }
}

function sparseCosine(a: Map<string, number>, b: Map<string, number>): number {
  let dot = 0;
  for (const [t, w] of a) dot += w * (b.get(t) ?? 0);
  const norm = (v: Map<string, number>) => Math.sqrt([...v.values()].reduce((s, w) => s + w * w, 0));
  const denominator = norm(a) * norm(b);
  return denominator === 0 ? 0 : dot / denominator;
}

function denseCosine(a: number[], b: number[]): number {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < Math.min(a.length, b.length); i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  return na === 0 || nb === 0 ? 0 : dot / Math.sqrt(na * nb);
}
