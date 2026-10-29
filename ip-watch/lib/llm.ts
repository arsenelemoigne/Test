import { logCall } from "./log";
import { traceStep } from "./trace";

// OpenRouter chat completions (OpenAI-compatible), with tool calls and the web search plugin.
const ENDPOINT = "https://openrouter.ai/api/v1/chat/completions";
const DEFAULT_MODEL = "google/gemini-2.5-flash";

export type ToolCall = { id: string; type: "function"; function: { name: string; arguments: string } };
export type ChatMessage =
  | { role: "system" | "user"; content: string }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[] }
  | { role: "tool"; tool_call_id: string; content: string };
export type ToolSpec = { type: "function"; function: { name: string; description: string; parameters: Record<string, unknown> } };
type Annotation = { type: string; url_citation?: { url: string; title?: string; content?: string } };

export function llmConfigured(): boolean {
  return Boolean(process.env.OPENROUTER_API_KEY);
}

// "strong": legal answers and infringement scoring; "fast": triage, extraction and summaries.
export type Tier = "strong" | "fast";

function modelFor(tier: Tier): string {
  const strong = process.env.OPENROUTER_MODEL || DEFAULT_MODEL;
  return tier === "fast" ? process.env.OPENROUTER_MODEL_FAST || strong : strong;
}

export async function chat(opts: {
  messages: ChatMessage[];
  tier?: Tier;
  tools?: ToolSpec[];
  temperature?: number;
  json?: boolean;
  plugins?: Record<string, unknown>[];
  label?: string;
}): Promise<{ message: { content: string | null; tool_calls?: ToolCall[] }; annotations: Annotation[] }> {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error("OPENROUTER_API_KEY manquant");
  const started = Date.now();
  const res = await fetch(ENDPOINT, {
    method: "POST",
    headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Title": "Registre PI (demo)" },
    body: JSON.stringify({
      model: modelFor(opts.tier ?? "strong"),
      temperature: opts.temperature ?? 0.1,
      messages: opts.messages,
      ...(opts.tools?.length ? { tools: opts.tools, tool_choice: "auto" } : {}),
      ...(opts.json ? { response_format: { type: "json_object" } } : {}),
      ...(opts.plugins ? { plugins: opts.plugins } : {}),
    }),
  });
  const text = await res.text();
  const model = modelFor(opts.tier ?? "strong");
  const last = opts.messages[opts.messages.length - 1];
  const input = { dernier_message: typeof last?.content === "string" ? last.content.slice(0, 600) : null, outils: opts.tools?.length ?? 0 };
  if (!res.ok) {
    await logCall("llm", opts.label ?? "chat", null, text.slice(0, 2000), res.status, Date.now() - started);
    traceStep({ kind: "ia", name: opts.label ?? "chat", detail: model, input, output: text.slice(0, 300), ms: Date.now() - started, ok: false });
    throw new Error(`OpenRouter ${res.status}: ${text.slice(0, 200)}`);
  }
  const json = JSON.parse(text) as { choices?: { message?: { content?: string | null; tool_calls?: ToolCall[]; annotations?: Annotation[] } }[]; usage?: { total_tokens?: number } };
  const message = json.choices?.[0]?.message ?? {};
  traceStep({
    kind: "ia",
    name: opts.label ?? "chat",
    detail: `${model}${json.usage?.total_tokens ? ` · ${json.usage.total_tokens} tokens` : ""}`,
    input,
    output: message.tool_calls?.length ? { decide_d_appeler: message.tool_calls.map((c) => ({ outil: c.function.name, arguments: c.function.arguments })) } : message.content,
    ms: Date.now() - started,
    ok: true,
  });
  return { message: { content: message.content ?? null, tool_calls: message.tool_calls }, annotations: message.annotations ?? [] };
}

// Asks for a JSON object; returns null when the model is unavailable or the answer cannot be parsed.
export async function extractJson<T>(system: string, user: string, label: string, tier: Tier = "fast"): Promise<T | null> {
  if (!llmConfigured()) return null;
  try {
    const { message } = await chat({ messages: [{ role: "system", content: system }, { role: "user", content: user }], json: true, temperature: 0, label, tier });
    const raw = (message.content ?? "").replace(/^```(?:json)?\s*|\s*```$/g, "");
    return JSON.parse(raw) as T;
  } catch {
    return null;
  }
}

export type WebResult = { url: string; title: string; snippet: string };

// Fallback search through OpenRouter's web plugin; only the cited URLs are kept.
export async function webSearch(query: string, label = "web"): Promise<{ summary: string; results: WebResult[] }> {
  const { message, annotations } = await chat({
    messages: [
      { role: "system", content: "Recherche les sources les plus fiables et récentes. Réponds en 3 phrases maximum, en français, sans inventer." },
      { role: "user", content: query },
    ],
    plugins: [{ id: "web", max_results: 6 }],
    label,
    tier: "fast",
  });
  const seen = new Set<string>();
  const results: WebResult[] = [];
  for (const a of annotations) {
    const c = a.url_citation;
    if (!c?.url || seen.has(c.url)) continue;
    seen.add(c.url);
    results.push({ url: c.url, title: c.title ?? c.url, snippet: (c.content ?? "").slice(0, 400) });
  }
  return { summary: message.content ?? "", results };
}
