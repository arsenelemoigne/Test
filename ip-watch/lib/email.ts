import { Resend } from "resend";
import { runAgent } from "./agent";
import { extractJson } from "./llm";
import { logCall } from "./log";
import { toHtml } from "./markdown";
import { addMarks, onboard } from "./onboard";
import * as store from "./store";
import { startWatch } from "./watch";
import { startVisualSearch } from "./visual";
import { LEVEL_LABEL, levelCounts, MODE_LABEL, type VisualSearch } from "./visual-types";
import { runInTrace, startTrace, traceStep } from "./trace";
import type { Client } from "./types";

type Intent = {
  intention?: "nouveau_client" | "mise_a_jour" | "ajout_marque" | "recherche_contrefacon" | "question" | "note";
  client?: string | null;
  marques?: string[];
  question?: string | null;
  resume?: string;
};

export function appUrl(): string {
  return (process.env.APP_URL || "https://pi.getmanda.eu").replace(/\/$/, "");
}

function resend(): Resend {
  const key = process.env.RESEND_API_KEY;
  if (!key) throw new Error("RESEND_API_KEY manquant");
  return new Resend(key);
}

const addressOf = (from: string) => (from.match(/<([^>]+)>/)?.[1] ?? from).trim().toLowerCase();

function allowedSenders(): string[] {
  return [process.env.LAWYER_EMAIL, ...(process.env.ALLOWED_SENDERS ?? "").split(",")].map((s) => (s ?? "").trim().toLowerCase()).filter(Boolean);
}

function stripHtml(html: string): string {
  return html.replace(/<style[\s\S]*?<\/style>/gi, "").replace(/<br\s*\/?>/gi, "\n").replace(/<\/(p|div|tr|li)>/gi, "\n").replace(/<[^>]+>/g, "").replace(/&nbsp;/g, " ").replace(/&amp;/g, "&").replace(/\n{3,}/g, "\n\n").trim();
}

function matchClient(clients: Client[], name: string | null | undefined): Client | null {
  if (clients.length === 1 && !name) return clients[0];
  if (!name) return null;
  const n = name.toLowerCase();
  return clients.find((c) => c.name.toLowerCase() === n || c.slug === n) ?? clients.find((c) => c.name.toLowerCase().includes(n) || n.includes(c.name.toLowerCase())) ?? null;
}

async function reply(to: string, subject: string, markdown: string, messageId: string | null, attachments?: { filename: string; content: Buffer }[], link = appUrl()) {
  const from = process.env.RESEND_FROM || "PI Pilot <assistant@getmanda.eu>";
  const footer = `\n\n[Ouvrir dans PI Pilot](${link})`;
  await resend().emails.send({
    from,
    to,
    subject: subject.startsWith("Re:") ? subject : `Re: ${subject}`,
    html: `<div style="font-family:system-ui,sans-serif;font-size:14px;line-height:1.5;color:#1d1d1b">${toHtml(markdown + footer)}</div>`,
    text: markdown,
    headers: messageId ? { "In-Reply-To": messageId, References: messageId } : undefined,
    ...(attachments?.length ? { attachments } : {}),
  });
}

// The first image attached to the email, if any (photos pasted inline count too).
async function firstImage(emailId: string): Promise<Buffer | null> {
  try {
    const { data } = await resend().emails.receiving.attachments.list({ emailId });
    const image = (data?.data ?? []).find((a) => a.content_type?.startsWith("image/") && a.size > 2000);
    if (!image) return null;
    const res = await fetch(image.download_url, { signal: AbortSignal.timeout(20_000) });
    return res.ok ? Buffer.from(await res.arrayBuffer()) : null;
  } catch (e) {
    void logCall("email", "attachments", { emailId }, e instanceof Error ? e.message : String(e), 500, 0);
    return null;
  }
}

// Summary sent back when a visual search started by email is finished, with the PDF report attached.
function visualSummary(s: VisualSearch): string {
  const counts = levelCounts(s.results);
  const top = s.results.filter((r) => r.level === "tres_proche" || r.level === "proche").slice(0, 5);
  return [
    `**Recherche visuelle terminée** — ${MODE_LABEL[s.mode]}${s.analysis?.titre ? ` : ${s.analysis.titre}` : ""}.`,
    "",
    `${s.results.length} résultat(s) examiné(s) : ${counts.tres_proche} très proche(s), ${counts.proche} proche(s), ${counts.meme_theme} même thème.`,
    ...(top.length ? ["", "Les plus proches :", ...top.map((r) => `- ${LEVEL_LABEL[r.level!]} — [${r.title.slice(0, 90)}](${r.url}) (${r.source}${r.date ? `, ${r.date}` : ""})`)] : []),
    "",
    "Le rapport PDF est joint ; le tableur et les visuels sont rangés dans le dossier du client. Les rapprochements sont proposés par l'IA et restent à vérifier.",
  ].join("\n");
}

// Handles one received email: understand it, act, log a summary in the history and answer by email.
export async function handleInboundEmail(emailId: string): Promise<void> {
  if (await store.emailAlreadyHandled(emailId)) return;
  const { data: email, error } = await resend().emails.receiving.get(emailId);
  if (error || !email) throw new Error(`Resend receiving: ${error?.message ?? "email introuvable"}`);

  const sender = addressOf(email.from);
  const subject = email.subject || "(sans objet)";
  const body = (email.text || stripHtml(email.html ?? "")).slice(0, 12000);
  if (!allowedSenders().includes(sender)) {
    await store.addEvent(null, "email", `Email ignoré — ${subject}`, `Expéditeur non autorisé : ${sender}.`, { emailId, from: sender });
    return;
  }

  // An image in the email starts a visual search; the report is sent back when it is ready.
  const image = await firstImage(emailId);
  if (image) {
    await startVisualFromEmail(emailId, sender, subject, body, email.message_id ?? null, image);
    return;
  }

  const trace = await startTrace("email", subject, null);
  await runInTrace(trace, () => processEmail(emailId, sender, subject, body, email.message_id ?? null, trace.id));
  await trace.finish("done", null);
}

async function startVisualFromEmail(emailId: string, sender: string, subject: string, body: string, messageId: string | null, image: Buffer) {
  const clients = await store.listClients();
  const text = `${subject}\n${body}`;
  const guess = await extractJson<{ client?: string | null; mode?: string; precisions?: string | null }>(
    [
      "Un avocat en propriété intellectuelle envoie une image par email pour une recherche visuelle.",
      `Clients existants : ${clients.map((c) => c.name).join(", ") || "aucun"}.`,
      "mode : « contrefacon » s'il cherche des copies d'un produit ou visuel de son client, sinon « anteriorite » (vérifier avant un lancement ce qui existe déjà).",
      'Réponds en JSON : {"client":"nom ou null","mode":"anteriorite|contrefacon","precisions":"ce qu\'il dit de l\'image, une phrase, ou null"}.',
    ].join("\n"),
    text.slice(0, 4000),
    "email-visual",
  );
  const client = matchClient(clients, guess?.client) ?? (clients.length === 1 ? clients[0] : null);
  const mode = guess?.mode === "contrefacon" || /copie|contrefa|imitation|dupe|counterfeit/i.test(text) ? "contrefacon" : "anteriorite";
  const note = guess?.precisions?.trim() || body.trim().slice(0, 400) || null;
  const search = await startVisualSearch({
    client,
    mode,
    note,
    image,
    origin: "email",
    onDone: async (s, pdf) => {
      const link = `${appUrl()}/recherche-visuelle?id=${s.id}`;
      try {
        await reply(sender, subject, visualSummary(s), messageId, pdf ? [{ filename: "rapport-recherche-visuelle.pdf", content: Buffer.from(pdf) }] : undefined, link);
      } catch (e) {
        void logCall("email", "reply", { to: sender }, e instanceof Error ? e.message : String(e), 500, 0);
      }
    },
  });
  await store.addEvent(client?.id ?? null, "email", subject, `Image reçue par email : recherche visuelle lancée (${MODE_LABEL[mode]}).`, { emailId, from: sender, intent: "recherche_visuelle", visualSearchId: search.id, traceId: search.traceId });
}

async function processEmail(emailId: string, sender: string, subject: string, body: string, messageId: string | null, traceId: string) {
  const clients = await store.listClients();
  const intent =
    (await extractJson<Intent>(
      [
        "Tu tries les emails reçus par l'assistant d'un cabinet d'avocats en propriété intellectuelle.",
        `Clients existants : ${clients.map((c) => c.name).join(", ") || "aucun"}.`,
        "Intentions possibles : nouveau_client (suivre une nouvelle société), mise_a_jour (rafraîchir un client existant), ajout_marque (ajouter des marques précises à un client), recherche_contrefacon (chercher des atteintes pour une marque), question (question juridique ou sur le portefeuille), note (information à garder).",
        'Réponds en JSON : {"intention": "...", "client": "nom ou null", "marques": ["..."], "question": "question reformulée ou null", "resume": "deux phrases maximum, factuelles"}.',
      ].join("\n"),
      `Objet : ${subject}\n\n${body}`,
      "email-intent",
    )) ?? { intention: "note", resume: subject };
  traceStep({ kind: "etape", name: "Intention de l'email", output: intent, ok: true });

  let client = matchClient(clients, intent.client);
  let answer: string;
  try {
    switch (intent.intention) {
      case "nouveau_client": {
        const r = await onboard(`${subject}\n${body}`, "email");
        client = r.client;
        answer = `**${r.client.name}** ajouté au registre. ${r.summary}`;
        break;
      }
      case "mise_a_jour": {
        if (!client) throw new Error("Client introuvable dans le registre.");
        const r = await onboard(client.company?.siren ? `${client.name} ${client.company.siren}` : client.name, "email");
        answer = `Registre de **${client.name}** mis à jour. ${r.summary}`;
        break;
      }
      case "ajout_marque": {
        if (!client) throw new Error("Client introuvable dans le registre.");
        const n = await addMarks(client, intent.marques ?? []);
        answer = `${n} titre(s) ajouté(s) au registre de **${client.name}** (${(intent.marques ?? []).join(", ")}).`;
        break;
      }
      case "recherche_contrefacon": {
        if (!client) throw new Error("Client introuvable dans le registre.");
        const marks = await store.listMarks(client.id);
        const wanted = (intent.marques ?? []).map((m) => m.toUpperCase());
        const mark = marks.find((m) => wanted.some((w) => m.name.toUpperCase() === w)) ?? marks.find((m) => wanted.some((w) => m.name.toUpperCase().includes(w)));
        if (!mark) throw new Error(`Marque introuvable : ${(intent.marques ?? []).join(", ")}.`);
        await startWatch(client, mark, "email");
        answer = `Recherche d'atteintes lancée pour **${mark.name}** [${mark.id}]. Les résultats s'affichent sur le tableau de bord et s'ajoutent au journal de la marque.`;
        break;
      }
      case "question": {
        const marks = client ? await store.listMarks(client.id) : [];
        const r = await runAgent({ client, marks, messages: [{ role: "user", content: intent.question || body }], focusMarkIds: [], trigger: "email" });
        answer = r.answer;
        break;
      }
      default:
        answer = "Noté dans l'historique.";
    }
  } catch (e) {
    answer = `Je n'ai pas pu traiter cette demande : ${e instanceof Error ? e.message : String(e)}`;
  }

  await store.addEvent(client?.id ?? null, "email", subject, intent.resume ?? null, { emailId, from: sender, intent: intent.intention, answer: answer.slice(0, 2000), traceId });
  try {
    await reply(sender, subject, answer, messageId);
  } catch (e) {
    void logCall("email", "reply", { to: sender }, e instanceof Error ? e.message : String(e), 500, 0);
  }
}

export function verifyWebhook(payload: string, headers: Headers) {
  const secret = process.env.RESEND_WEBHOOK_SECRET;
  if (!secret) throw new Error("RESEND_WEBHOOK_SECRET manquant");
  return resend().webhooks.verify({
    payload,
    headers: { id: headers.get("svix-id") ?? "", timestamp: headers.get("svix-timestamp") ?? "", signature: headers.get("svix-signature") ?? "" },
    webhookSecret: secret,
  });
}
