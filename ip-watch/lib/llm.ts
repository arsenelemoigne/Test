import { logCall } from "./log";

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

export async function chat(opts: {
  messages: ChatMessage[];
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
      model: process.env.OPENROUTER_MODEL || DEFAULT_MODEL,
      temperature: opts.temperature ?? 0.1,
      messages: opts.messages,
      ...(opts.tools?.length ? { tools: opts.tools, tool_choice: "auto" } : {}),
      ...(opts.json ? { response_format: { type: "json_object" } } : {}),
      ...(opts.plugins ? { plugins: opts.plugins } : {}),
    }),
  });
  const text = await res.text();
  if (!res.ok) {
    await logCall("llm", opts.label ?? "chat", null, text.slice(0, 2000), res.status, Date.now() - started);
    throw new Error(`OpenRouter ${res.status}: ${text.slice(0, 200)}`);
  }
  const json = JSON.parse(text) as { choices?: { message?: { content?: string | null; tool_calls?: ToolCall[]; annotations?: Annotation[] } }[] };
  const message = json.choices?.[0]?.message ?? {};
  return { message: { content: message.content ?? null, tool_calls: message.tool_calls }, annotations: message.annotations ?? [] };
}

// Asks for a JSON object; returns null when the model is unavailable or the answer cannot be parsed.
export async function extractJson<T>(system: string, user: string, label: string): Promise<T | null> {
  if (!llmConfigured()) return null;
  try {
    const { message } = await chat({ messages: [{ role: "system", content: system }, { role: "user", content: user }], json: true, temperature: 0, label });
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
