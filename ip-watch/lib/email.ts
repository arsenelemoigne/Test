import { Resend } from "resend";
import { runAgent } from "./agent";
import { extractJson } from "./llm";
import { logCall } from "./log";
import { toHtml } from "./markdown";
import { addMarks, onboard } from "./onboard";
import * as store from "./store";
import { startWatch } from "./watch";
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

async function reply(to: string, subject: string, markdown: string, messageId: string | null) {
  const from = process.env.RESEND_FROM || "Registre PI <assistant@getmanda.eu>";
  const footer = `\n\n[Ouvrir le tableau de bord](${appUrl()})`;
  await resend().emails.send({
    from,
    to,
    subject: subject.startsWith("Re:") ? subject : `Re: ${subject}`,
    html: `<div style="font-family:system-ui,sans-serif;font-size:14px;line-height:1.5;color:#1d1d1b">${toHtml(markdown + footer)}</div>`,
    text: markdown,
    headers: messageId ? { "In-Reply-To": messageId, References: messageId } : undefined,
  });
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

  await store.addEvent(client?.id ?? null, "email", subject, intent.resume ?? null, { emailId, from: sender, intent: intent.intention, answer: answer.slice(0, 2000) });
  try {
    await reply(sender, subject, answer, email.message_id ?? null);
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
