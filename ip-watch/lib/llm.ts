import { logCall } from "./log";
import { traceStep } from "./trace";

// OpenRouter chat completions (OpenAI-compatible), with tool calls and the web search plugin.
// OPENROUTER_BASE_URL only serves local tests against a stub.
const ENDPOINT = `${process.env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1"}/chat/completions`;
const DEFAULT_MODEL = "google/gemini-2.5-flash";

export type ToolCall = { id: string; type: "function"; function: { name: string; arguments: string } };
// Images go to vision-capable models as URLs or data: URIs.
export type ContentPart = { type: "text"; text: string } | { type: "image_url"; image_url: { url: string } };
export type ChatMessage =
  | { role: "system"; content: string }
  | { role: "user"; content: string | ContentPart[] }
  | { role: "assistant"; content: string | null; tool_calls?: ToolCall[]; reasoning_details?: unknown[] }
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
  // Thinking budget in tokens; the model's reasoning is then shown in the déroulé.
  reasoning?: number;
}): Promise<{ message: { content: string | null; tool_calls?: ToolCall[]; reasoning_details?: unknown[] }; annotations: Annotation[]; reasoning: string | null }> {
  const key = process.env.OPENROUTER_API_KEY;
  if (!key) throw new Error("OPENROUTER_API_KEY manquant");
  const started = Date.now();
  const label = opts.label ?? "chat";
  type Attempt = { model: string; reasoning: boolean };
  const post = (a: Attempt) =>
    fetch(ENDPOINT, {
      method: "POST",
      headers: { Authorization: `Bearer ${key}`, "Content-Type": "application/json", "X-Title": "Registre PI (demo)" },
      body: JSON.stringify({
        model: a.model,
        // Extended thinking only accepts the default temperature and needs room above its budget.
        ...(a.reasoning ? { reasoning: { max_tokens: opts.reasoning }, max_tokens: (opts.reasoning ?? 0) + 6000 } : { temperature: opts.temperature ?? 0.1 }),
        messages: a.reasoning ? opts.messages : opts.messages.map((m) => (m.role === "assistant" && m.reasoning_details ? { ...m, reasoning_details: undefined } : m)),
        ...(opts.tools?.length ? { tools: opts.tools, tool_choice: "auto" } : {}),
        ...(opts.json ? { response_format: { type: "json_object" } } : {}),
        ...(opts.plugins ? { plugins: opts.plugins } : {}),
      }),
    });
  const attempts: Attempt[] = [{ model: modelFor(opts.tier ?? "strong"), reasoning: Boolean(opts.reasoning) }];
  let attempt = attempts[0];
  let res: Response | null = null;
  let text = "";
  for (let i = 0; i < attempts.length && i < 3; i++) {
    attempt = attempts[i];
    res = await post(attempt);
    text = await res.text();
    if (res.ok) break;
    // A mistyped or retired model id falls back to the default; a refused thinking option is dropped.
    const aboutReasoning = /reasoning|thinking/i.test(text);
    if ((res.status === 400 || res.status === 404) && attempt.model !== DEFAULT_MODEL && /model/i.test(text) && !aboutReasoning) attempts.push({ ...attempt, model: DEFAULT_MODEL });
    else if (res.status === 400 && attempt.reasoning) attempts.push({ ...attempt, reasoning: false });
    else break;
    const next = attempts[attempts.length - 1];
    traceStep({ kind: "ia", name: label, detail: next.model !== attempt.model ? `${attempt.model} refusé, repli sur ${next.model}` : "réflexion refusée par le modèle, nouvel essai sans", output: text.slice(0, 300), ms: Date.now() - started, ok: false });
  }
  const model = attempt.model;
  const last = opts.messages[opts.messages.length - 1];
  const input = { dernier_message: typeof last?.content === "string" ? last.content.slice(0, 600) : null, outils: opts.tools?.length ?? 0 };
  if (!res?.ok) {
    await logCall("llm", label, null, text.slice(0, 2000), res?.status ?? 0, Date.now() - started);
    traceStep({ kind: "ia", name: label, detail: model, input, output: text.slice(0, 300), ms: Date.now() - started, ok: false });
    throw new Error(`OpenRouter ${res?.status}: ${text.slice(0, 200)}`);
  }
  type Raw = { content?: string | null; tool_calls?: ToolCall[]; annotations?: Annotation[]; reasoning?: string | null; reasoning_details?: unknown[] };
  const json = JSON.parse(text) as { choices?: { message?: Raw }[]; usage?: { total_tokens?: number } };
  const message: Raw = json.choices?.[0]?.message ?? {};
  const reasoning = typeof message.reasoning === "string" && message.reasoning.trim() ? message.reasoning.trim() : null;
  const calls = message.tool_calls ?? [];
  traceStep({
    kind: "ia",
    name: label,
    detail: `${model}${json.usage?.total_tokens ? ` · ${json.usage.total_tokens} tokens` : ""}`,
    input,
    output: calls.length ? { decide_d_appeler: calls.map((c) => ({ outil: c.function.name, arguments: c.function.arguments })) } : message.content,
    // Text written alongside tool calls is the model explaining its next move.
    summary: calls.length && message.content?.trim() ? message.content.trim().slice(0, 600) : undefined,
    thinking: reasoning ?? undefined,
    ms: Date.now() - started,
    ok: true,
  });
  return {
    message: { content: message.content ?? null, tool_calls: message.tool_calls, reasoning_details: attempt.reasoning && Array.isArray(message.reasoning_details) ? message.reasoning_details : undefined },
    annotations: message.annotations ?? [],
    reasoning,
  };
}

// Asks for a JSON object; returns null when the model is unavailable or the answer cannot be parsed.
export async function extractJson<T>(system: string, user: string | ContentPart[], label: string, tier: Tier = "fast"): Promise<T | null> {
  if (!llmConfigured()) return null;
  try {
    const { message } = await chat({ messages: [{ role: "system", content: system }, { role: "user", content: user }], json: true, temperature: 0, label, tier });
    const raw = (message.content ?? "").replace(/^```(?:json)?\s*|\s*```$/g, "");
    try {
      return JSON.parse(raw) as T;
    } catch {
      // Some models wrap the object in a sentence; keep the outermost braces.
      return JSON.parse(raw.slice(raw.indexOf("{"), raw.lastIndexOf("}") + 1)) as T;
    }
  } catch {
    return null;
  }
}

export type WebResult = { url: string; title: string; snippet: string };

// Fallback search through OpenRouter's web plugin; only the cited URLs are kept.
export async function webSearch(query: string, label = "web", opts: { system?: string; maxResults?: number } = {}): Promise<{ summary: string; results: WebResult[] }> {
  const { message, annotations } = await chat({
    messages: [
      { role: "system", content: opts.system ?? "Recherche les sources les plus fiables et récentes. Réponds en 3 phrases maximum, en français, sans inventer." },
      { role: "user", content: query },
    ],
    plugins: [{ id: "web", max_results: opts.maxResults ?? 6 }],
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
