import { chat, webSearch, type ChatMessage, type ToolSpec } from "./llm";
import { logCall } from "./log";
import { traceStep } from "./trace";
import { deadlineFor } from "./deadlines";
import * as legifrance from "./sources/legifrance";
import * as judilibre from "./sources/judilibre";
import * as inpi from "./sources/inpi";
import { searchCompanies } from "./sources/entreprises";
import { searchAnnouncements } from "./sources/bodacc";
import { pisteConfigured } from "./sources/piste";
import { rightCountries } from "./territories";
import { startWatch } from "./watch";
import * as store from "./store";
import type { Client, Mark } from "./types";

const MAX_ROUNDS = 6;
const MAX_TOOL_CHARS = 6000;
// Thinking budget per round (0 turns it off); the reasoning appears in the déroulé.
const REASONING_TOKENS = Number(process.env.OPENROUTER_REASONING_TOKENS ?? 1024);

const SYSTEM = `Tu es l'assistant juridique d'un cabinet d'avocats en propriété intellectuelle (droit français et de l'Union européenne). Tu réponds à des avocats : va droit au but.

Sources
- Utilise d'abord les outils : registre du client (ci-dessous), Légifrance, Judilibre, INPI, annuaire des entreprises, BODACC, droit de l'UE.
- La recherche web est un dernier recours : marque toute source web « (source web) ».
- N'invente aucun article, décision, date ou lien. Ne cite que des liens renvoyés par les outils.
- Avant de citer un article, récupère son texte avec legifrance_article. Si le résultat porte un champ « via », reprends cette mention entre parenthèses.

Déroulé
- Avant chaque appel d'outil, écris une phrase courte : ce que tu cherches et pourquoi. L'avocat la lit dans le déroulé.

Forme
- Français, phrases courtes. Pas de formule de politesse, pas de récapitulatif final.
- Liens intégrés aux mots, jamais d'URL nue : [art. L. 713-2 CPI](url), [Cass. com., 10 mai 2023, n° 21-12.345](url).
- Citations textuelles courtes entre « » (une phrase utile au plus), tirées de la source.
- Écris « Interprétation : » quand tu raisonnes au-delà des textes. Dis simplement « Je ne sais pas » ou « Aucune source trouvée » le cas échéant.
- Question juridique, dans cet ordre :
  **Réponse** — une ou deux phrases.
  **Règle** — texte applicable, lien, courte citation.
  **Application** — majeure, mineure, conclusion.
  **Limites** — ce qui reste incertain ou à vérifier.
  **Jurisprudence** — seulement si une décision pertinente a été trouvée (une ou deux).
- Question sur le portefeuille : réponse directe, identifiants entre crochets, par exemple [FR4012345].
- Recherche de contrefaçons : lance l'outil uniquement si on te le demande, puis confirme en une phrase (« Recherche lancée sur QUATRE. »), sans identifiant ni autre mention.
- Pas de conseil définitif : l'avocat décide.`;

const fn = (name: string, description: string, properties: Record<string, unknown>, required: string[]): ToolSpec => ({
  type: "function",
  function: { name, description, parameters: { type: "object", properties, required } },
});

const TOOLS: ToolSpec[] = [
  fn("legifrance_article", "Texte en vigueur d'un article de code (Légifrance).", { code: { type: "string", description: "Ex. « Code de la propriété intellectuelle »" }, numero: { type: "string", description: "Ex. « L713-2 »" } }, ["code", "numero"]),
  fn("legifrance_recherche", "Recherche d'articles de codes en vigueur par mots-clés (Légifrance).", { requete: { type: "string" }, code: { type: "string", description: "Nom exact du code, optionnel" } }, ["requete"]),
  fn("judilibre_recherche", "Recherche de décisions de la Cour de cassation et des cours d'appel (Judilibre).", { requete: { type: "string" }, juridiction: { type: "string", enum: ["cc", "ca", "tous"] } }, ["requete"]),
  fn("judilibre_decision", "Motifs et dispositif d'une décision Judilibre.", { id: { type: "string" } }, ["id"]),
  fn("inpi_marques", "Recherche dans le registre INPI (marques FR, UE, internationales), par exemple pour des signes proches.", { signe: { type: "string" }, classes: { type: "array", items: { type: "integer" } } }, ["signe"]),
  fn("entreprise", "Fiche officielle d'une société française (SIREN, dirigeants, siège).", { requete: { type: "string" } }, ["requete"]),
  fn("bodacc", "Annonces légales BODACC d'une société (immatriculation, cession, procédure collective).", { requete: { type: "string", description: "Nom ou SIREN" } }, ["requete"]),
  fn("droit_ue", "Règlements, directives et arrêts de la CJUE (EUR-Lex, CURIA).", { requete: { type: "string" } }, ["requete"]),
  fn("recherche_web", "Dernier recours : recherche web générale. À signaler « (source web) ».", { requete: { type: "string" } }, ["requete"]),
  fn("lancer_recherche_contrefacon", "Lance la recherche d'atteintes en ligne pour une marque du portefeuille (uniquement si demandé).", { marque_id: { type: "string" } }, ["marque_id"]),
  fn("resultats_recherche", "Résultats de la dernière recherche d'atteintes pour une marque du portefeuille.", { marque_id: { type: "string" } }, ["marque_id"]),
];

const SOURCE_LABELS: Record<string, string> = {
  legifrance_article: "Légifrance",
  legifrance_recherche: "Légifrance",
  judilibre_recherche: "Judilibre",
  judilibre_decision: "Judilibre",
  inpi_marques: "INPI",
  entreprise: "Annuaire des entreprises",
  bodacc: "BODACC",
  droit_ue: "EUR-Lex",
  recherche_web: "Web",
  lancer_recherche_contrefacon: "Veille",
  resultats_recherche: "Veille",
};

export type AgentInput = { client: Client | null; marks: Mark[]; messages: { role: "user" | "assistant"; content: string }[]; focusMarkIds: string[]; trigger: "app" | "email" };
export type AgentOutput = { answer: string; sources: { name: string; count: number }[]; runId: string | null };

function portfolioContext(client: Client | null, marks: Mark[], focus: string[]): string {
  const today = new Date().toLocaleDateString("fr-FR", { dateStyle: "long", timeZone: "Europe/Paris" });
  if (!client) return `Date : ${today}. Aucun client sélectionné.`;
  const line = (m: Mark) => {
    const d = deadlineFor(m);
    return [m.id, m.type, m.office, m.name, m.clientOwned === false ? "titulaire : société du groupe" : "", m.favorite ? "favori" : "", m.classes.length ? `cl. ${m.classes.join(",")}` : "", m.filingDate ? `dépôt ${m.filingDate}` : "", d.expiry ? `exp. ${d.expiry.toISOString().slice(0, 10)}` : "", m.status, `${rightCountries(m).length} pays`, m.lastRun ? `dernière recherche ${m.lastRun.startedAt.slice(0, 10)} (${m.lastRun.total} résultats, ${m.lastRun.high} forte confiance)` : ""].filter(Boolean).join(" | ");
  };
  const focused = marks.filter((m) => focus.includes(m.id));
  return [
    `Date : ${today}.`,
    `Client : ${client.name}${client.company ? ` — ${client.company.name}, SIREN ${client.company.siren}, ${client.company.address ?? ""}` : ""}.`,
    focused.length ? `Marques ciblées par l'utilisateur : ${focused.map((m) => `${m.name} [${m.id}]`).join(", ")}. Concentre la réponse sur elles.` : "",
    `Portefeuille (${marks.length} titres) :`,
    ...[...marks].sort((a, b) => Number(a.clientOwned === false) - Number(b.clientOwned === false)).slice(0, 400).map(line),
  ].filter(Boolean).join("\n");
}

function compact(value: unknown): string {
  const s = JSON.stringify(value);
  return s.length > MAX_TOOL_CHARS ? `${s.slice(0, MAX_TOOL_CHARS)}…` : s;
}

// Keeps only links that a tool actually returned; others are turned into plain text.
export function verifyLinks(answer: string, allowed: Set<string>): string {
  const norm = (u: string) => u.replace(/\/$/, "");
  const ok = new Set(Array.from(allowed).map(norm));
  return answer.replace(/\[([^\]]+)\]\((https?:\/\/[^)\s]+)\)/g, (all, text: string, url: string) => (ok.has(norm(url)) ? all : `${text} (lien non vérifié)`));
}

function collectUrls(value: unknown, out: Set<string>) {
  if (typeof value === "string") {
    for (const m of value.match(/https?:\/\/[^\s"'<>)]+/g) ?? []) out.add(m);
  } else if (Array.isArray(value)) value.forEach((v) => collectUrls(v, out));
  else if (value && typeof value === "object") Object.values(value).forEach((v) => collectUrls(v, out));
}

// One readable line per tool call for the déroulé.
function describeCall(name: string, a: Record<string, unknown>, markById: (id: string) => Mark | undefined): string {
  const s = (k: string) => String(a[k] ?? "").trim();
  switch (name) {
    case "legifrance_article":
      return `article ${s("numero")} · ${s("code")}`;
    case "legifrance_recherche":
      return `« ${s("requete")} »${a.code ? ` dans le ${s("code")}` : ""}`;
    case "judilibre_recherche":
      return `« ${s("requete")} »${a.juridiction === "cc" ? " · Cour de cassation" : a.juridiction === "ca" ? " · cours d'appel" : ""}`;
    case "judilibre_decision":
      return `décision ${s("id")}`;
    case "inpi_marques":
      return `signe « ${s("signe")} »${Array.isArray(a.classes) && a.classes.length ? ` · classes ${a.classes.join(", ")}` : ""}`;
    case "lancer_recherche_contrefacon":
    case "resultats_recherche":
      return `marque ${markById(s("marque_id"))?.name ?? s("marque_id")}`;
    default:
      return s("requete") ? `« ${s("requete")} »` : JSON.stringify(a).slice(0, 120);
  }
}

// What a tool sent back, in one sentence, plus the pages it pointed to.
function describeResult(output: unknown): { summary: string; links: { url: string; title?: string }[] } {
  const links: { url: string; title?: string }[] = [];
  const seen = new Set<string>();
  const walk = (v: unknown, depth: number) => {
    if (depth > 4 || links.length >= 8 || !v || typeof v !== "object") return;
    if (Array.isArray(v)) return v.forEach((x) => walk(x, depth + 1));
    const o = v as Record<string, unknown>;
    if (typeof o.url === "string" && !seen.has(o.url)) {
      seen.add(o.url);
      const title = [o.title, o.titre, o.name, o.num ? `art. ${o.num}` : null, o.number, o.company].find((x) => typeof x === "string" && x.trim());
      links.push({ url: o.url, title: typeof title === "string" ? title.slice(0, 120) : undefined });
    }
    Object.values(o).forEach((x) => walk(x, depth + 1));
  };
  walk(output, 0);
  if (!output || typeof output !== "object") return { summary: String(output ?? "aucune réponse"), links };
  const o = output as Record<string, unknown>;
  if (typeof o.erreur === "string") return { summary: `échec : ${o.erreur.slice(0, 200)}`, links };
  if (typeof o.text === "string") return { summary: `texte obtenu${o.via ? ` (${o.via})` : ""} : « ${o.text.replace(/\s+/g, " ").slice(0, 180)}… »`, links };
  if (o.lancee) return { summary: "recherche lancée, résultats sur le tableau de bord", links };
  const list = Array.isArray(output) ? output : Object.values(o).find(Array.isArray);
  if (list) return { summary: `${typeof o.total === "number" ? o.total : list.length} résultat(s)${list.length === 0 ? " : rien trouvé" : ""}`, links };
  return { summary: "réponse reçue", links };
}

export async function runAgent(input: AgentInput): Promise<AgentOutput> {
  const { client, marks } = input;
  const allowed = new Set<string>();
  marks.forEach((m) => m.sourceUrl && allowed.add(m.sourceUrl));
  if (client?.company?.url) allowed.add(client.company.url);
  const used = new Map<string, number>();
  let runId: string | null = null;

  const markById = (id: string) => marks.find((m) => m.id === id) ?? marks.find((m) => m.name.toUpperCase() === id.toUpperCase());

  const exec = async (name: string, args: Record<string, unknown>): Promise<unknown> => {
    const q = String(args.requete ?? args.signe ?? "");
    switch (name) {
      case "legifrance_article":
        if (!pisteConfigured()) return { erreur: "Légifrance non configuré" };
        return (await legifrance.findArticle(String(args.code), String(args.numero))) ?? { erreur: "Article introuvable" };
      case "legifrance_recherche":
        if (!pisteConfigured()) return { erreur: "Légifrance non configuré" };
        return legifrance.searchLaw(q, args.code ? String(args.code) : undefined);
      case "judilibre_recherche":
        if (!pisteConfigured()) return { erreur: "Judilibre non configuré" };
        return judilibre.searchCaseLaw(q, (args.juridiction as "cc" | "ca" | "tous") ?? "tous");
      case "judilibre_decision":
        if (!pisteConfigured()) return { erreur: "Judilibre non configuré" };
        return judilibre.getDecision(String(args.id));
      case "inpi_marques": {
        const classes = Array.isArray(args.classes) && args.classes.length ? ` ET [ClassNumber=(${(args.classes as number[]).join(" OU ")})]` : "";
        const res = await inpi.search("marques", `[Mark_Exp=${q.toUpperCase().replace(/[[\]]/g, "")}]${classes}`, ["FR", "EU", "WO"], 0, 30);
        return res.hits.slice(0, 15).map((h) => inpi.normalize(h, "marque")).filter(Boolean).map((r) => ({ id: r!.id, nom: r!.name, titulaire: r!.owner, classes: r!.classes, depot: r!.filingDate, statut: r!.status, url: r!.sourceUrl }));
      }
      case "entreprise":
        return searchCompanies(q);
      case "bodacc":
        return searchAnnouncements(q);
      case "droit_ue": {
        const r = await webSearch(`${q} site:eur-lex.europa.eu OR site:curia.europa.eu`, "agent-eu");
        return { resultats: r.results.filter((x) => /eur-lex\.europa\.eu|curia\.europa\.eu/.test(x.url)), note: "Sources officielles de l'UE trouvées par recherche web." };
      }
      case "recherche_web": {
        const r = await webSearch(q, "agent-web");
        return { resultats: r.results, note: "Source web : à signaler comme telle." };
      }
      case "lancer_recherche_contrefacon": {
        const mark = markById(String(args.marque_id));
        if (!client || !mark) return { erreur: "Marque introuvable dans le portefeuille" };
        const run = await startWatch(client, mark, input.trigger === "email" ? "email" : "assistant");
        runId = run.id;
        return { lancee: true, marque: mark.name };
      }
      case "resultats_recherche": {
        const mark = markById(String(args.marque_id));
        if (!client || !mark) return { erreur: "Marque introuvable dans le portefeuille" };
        const rows = await store.markFindings(client.id, mark.id);
        const top = rows.map((r) => r.finding).sort((a, b) => (b.confidence ?? 0) - (a.confidence ?? 0)).slice(0, 15);
        return { total: rows.length, resultats: top.map((f) => ({ titre: f.title, url: f.url, pays: f.country, prix: f.price, devise: f.currency, confiance: f.confidence, motif: f.reason, date: f.foundAt.slice(0, 10) })) };
      }
      default:
        return { erreur: `Outil inconnu : ${name}` };
    }
  };

  const messages: ChatMessage[] = [
    { role: "system", content: SYSTEM },
    { role: "system", content: portfolioContext(client, marks, input.focusMarkIds) },
    ...input.messages.slice(-10),
  ];

  for (let round = 0; round < MAX_ROUNDS; round++) {
    const { message } = await chat({ messages, tools: TOOLS, label: `agent-${round}`, reasoning: REASONING_TOKENS > 0 ? REASONING_TOKENS : undefined });
    const calls = message.tool_calls ?? [];
    if (calls.length === 0) {
      const raw = (message.content ?? "").trim();
      const answer = verifyLinks(raw, allowed);
      const flagged = (raw.match(/\]\(https?:/g) ?? []).length - (answer.match(/\]\(https?:/g) ?? []).length;
      traceStep({ kind: "etape", name: "Vérification des liens", detail: flagged ? `${flagged} lien(s) non renvoyé(s) par une source, signalé(s)` : "tous les liens viennent des sources consultées", ok: flagged === 0 });
      void logCall("agent", "answer", { question: input.messages[input.messages.length - 1]?.content, focus: input.focusMarkIds }, answer, 200, 0, client?.slug);
      return { answer: answer || "Je n'ai pas pu formuler de réponse.", sources: Array.from(used.entries()).map(([name, count]) => ({ name, count })), runId };
    }
    messages.push({ role: "assistant", content: message.content ?? null, tool_calls: calls, ...(message.reasoning_details ? { reasoning_details: message.reasoning_details } : {}) });
    for (const call of calls) {
      const started = Date.now();
      let output: unknown;
      let args: Record<string, unknown> = {};
      try {
        args = JSON.parse(call.function.arguments || "{}");
        output = await exec(call.function.name, args);
      } catch (e) {
        output = { erreur: e instanceof Error ? e.message.slice(0, 200) : String(e) };
      }
      collectUrls(output, allowed);
      const label = SOURCE_LABELS[call.function.name] ?? call.function.name;
      used.set(label, (used.get(label) ?? 0) + 1);
      const failed = Boolean((output as { erreur?: string })?.erreur);
      traceStep({ kind: "api", name: `outil · ${call.function.name}`, detail: label, input: args, output, ms: Date.now() - started, ok: !failed, what: describeCall(call.function.name, args, markById), ...describeResult(output) });
      void logCall("tool", call.function.name, args, output, (output as { erreur?: string })?.erreur ? 500 : 200, Date.now() - started, client?.slug);
      messages.push({ role: "tool", tool_call_id: call.id, content: compact(output) });
    }
  }
  return { answer: "La recherche a demandé trop d'étapes ; reformulez la question plus précisément.", sources: Array.from(used.entries()).map(([name, count]) => ({ name, count })), runId };
}
