import { readFile } from 'node:fs/promises';
import { createServer, type IncomingMessage, type ServerResponse } from 'node:http';
import type { SponsoredOffer } from '../shared/types.ts';
import { LlmAds, OFFERS_TOOL_NAME, offersToolDefinition } from '../sdk/index.ts';
import { chatCompletion, llmConfigFromEnv, type ChatMessage } from './llm.ts';

/**
 * A free chat app acting as publisher #1. It runs the model, runs the ad auction locally,
 * and renders offers as separate labeled cards.
 */

const PORT = Number(process.env.DEMO_PORT ?? 3001);
const MARKETPLACE_URL = process.env.MARKETPLACE_URL ?? 'http://localhost:4000';
const PUBLISHER_ID = process.env.PUBLISHER_ID ?? 'demo-chat';
const PUBLISHER_SECRET = process.env.PUBLISHER_SECRET ?? 'dev-secret-change-me-please';
const llm = llmConfigFromEnv();
/** "tool": the model decides when to look for offers. "card": every user message is matched locally. */
const AD_MODE = process.env.AD_MODE ?? (llm ? 'tool' : 'card');

const SYSTEM_PROMPT =
  'You are a helpful assistant in a free chat app funded by sponsored offers. ' +
  `When the user is clearly considering a purchase, you may call ${OFFERS_TOOL_NAME}. ` +
  'Offers appear to the user as separate cards labeled "Sponsored". Mention an offer only if it genuinely fits, ' +
  'always say it is sponsored, and keep your own advice independent of it.';

const ads = new LlmAds({
  marketplaceUrl: MARKETPLACE_URL,
  publisherId: PUBLISHER_ID,
  publisherSecret: PUBLISHER_SECRET,
  // Trust-on-first-use for the prototype. In production, pin the key in config.
  catalogPublicKeyPem: process.env.CATALOG_PUBLIC_KEY_PEM ?? (await fetchPublicKey()),
});

setInterval(() => void ads.flushImpressions(), 30_000).unref();

async function fetchPublicKey(): Promise<string> {
  for (let attempt = 1; ; attempt++) {
    try {
      const res = await fetch(new URL('/catalog/public-key', MARKETPLACE_URL));
      if (res.ok) return await res.text();
      throw new Error(`HTTP ${res.status}`);
    } catch (err) {
      if (attempt >= 10) throw new Error(`cannot reach marketplace at ${MARKETPLACE_URL}: ${err}`);
      console.log(`waiting for marketplace at ${MARKETPLACE_URL}...`);
      await new Promise((r) => setTimeout(r, 1000));
    }
  }
}

async function chat(history: ChatMessage[]): Promise<{ reply: string; offers: SponsoredOffer[] }> {
  const lastUser = [...history].reverse().find((m) => m.role === 'user')?.content ?? '';

  if (!llm) {
    const offers = await ads.findOffers(lastUser, 1);
    return {
      reply: `(Offline demo model: set LLM_BASE_URL and LLM_MODEL to use a real one.) You said: "${lastUser}"`,
      offers,
    };
  }

  if (AD_MODE === 'card') {
    const [message, offers] = await Promise.all([
      chatCompletion(llm, [{ role: 'system', content: 'You are a helpful assistant.' }, ...history]),
      ads.findOffers(lastUser, 1),
    ]);
    return { reply: message.content ?? '', offers };
  }

  const messages: ChatMessage[] = [{ role: 'system', content: SYSTEM_PROMPT }, ...history];
  const offers: SponsoredOffer[] = [];
  for (let round = 0; round < 3; round++) {
    const message = await chatCompletion(llm, messages, [offersToolDefinition]);
    if (!message.tool_calls?.length) return { reply: message.content ?? '', offers: offers.slice(0, 2) };
    messages.push(message);
    for (const call of message.tool_calls) {
      if (call.function.name !== OFFERS_TOOL_NAME) {
        messages.push({ role: 'tool', tool_call_id: call.id, content: '{"error":"unknown tool"}' });
        continue;
      }
      const result = await ads.handleToolCall(call.function.arguments);
      for (const offer of result.offers) if (!offers.some((o) => o.id === offer.id)) offers.push(offer);
      messages.push({ role: 'tool', tool_call_id: call.id, content: result.content });
    }
  }
  return { reply: 'Sorry, I could not finish that answer.', offers: offers.slice(0, 2) };
}

const server = createServer((req, res) => {
  handle(req, res).catch((err: unknown) => {
    console.error(err);
    if (!res.headersSent) json(res, 500, { error: err instanceof Error ? err.message : 'internal error' });
  });
});

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  if (req.method === 'GET' && (req.url === '/' || req.url === '/index.html')) {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end(await readFile(new URL('./public/index.html', import.meta.url)));
    return;
  }
  if (req.method === 'POST' && req.url === '/api/chat') {
    const history = parseHistory(await readBody(req));
    if (!history) return json(res, 400, { error: 'expected { messages: [{ role: "user" | "assistant", content: string }] }' });
    const { reply, offers } = await chat(history);
    for (const offer of offers) ads.markShown(offer.id);
    return json(res, 200, { reply, offers, mode: llm ? AD_MODE : 'card (offline model)' });
  }
  json(res, 404, { error: 'not found' });
}

function parseHistory(body: string): ChatMessage[] | null {
  try {
    const data = JSON.parse(body) as { messages?: unknown };
    if (!Array.isArray(data.messages) || data.messages.length === 0 || data.messages.length > 50) return null;
    const messages = data.messages.map((m: { role?: unknown; content?: unknown }) => {
      if ((m.role !== 'user' && m.role !== 'assistant') || typeof m.content !== 'string' || m.content.length > 4000) throw new Error();
      return { role: m.role, content: m.content } as ChatMessage;
    });
    return messages;
  } catch {
    return null;
  }
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += (chunk as Buffer).length;
    if (size > 256 * 1024) throw new Error('body too large');
    chunks.push(chunk as Buffer);
  }
  return Buffer.concat(chunks).toString('utf8');
}

function json(res: ServerResponse, status: number, body: unknown): void {
  res.writeHead(status, { 'content-type': 'application/json' });
  res.end(JSON.stringify(body));
}

server.listen(PORT, () => {
  console.log(`demo chat on http://localhost:${PORT} (model: ${llm ? llm.model : 'offline stub'}, ad mode: ${llm ? AD_MODE : 'card'})`);
});
