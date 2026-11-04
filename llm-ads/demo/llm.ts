/** Minimal client for any OpenAI-compatible chat endpoint (Ollama, vLLM, OpenRouter, Mistral...). */

export interface ToolCall {
  id: string;
  type: 'function';
  function: { name: string; arguments: string };
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant' | 'tool';
  content: string | null;
  tool_calls?: ToolCall[];
  tool_call_id?: string;
}

export interface LlmConfig {
  baseUrl: string;
  model: string;
  apiKey?: string;
}

/** Set LLM_BASE_URL and LLM_MODEL, e.g. http://localhost:11434/v1 and qwen2.5:3b for Ollama. */
export function llmConfigFromEnv(): LlmConfig | null {
  const baseUrl = process.env.LLM_BASE_URL;
  const model = process.env.LLM_MODEL;
  if (!baseUrl || !model) return null;
  return { baseUrl: baseUrl.replace(/\/$/, ''), model, apiKey: process.env.LLM_API_KEY };
}

export async function chatCompletion(config: LlmConfig, messages: ChatMessage[], tools?: readonly unknown[]): Promise<ChatMessage> {
  const res = await fetch(`${config.baseUrl}/chat/completions`, {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      ...(config.apiKey ? { authorization: `Bearer ${config.apiKey}` } : {}),
    },
    body: JSON.stringify({ model: config.model, messages, ...(tools?.length ? { tools } : {}) }),
  });
  if (!res.ok) throw new Error(`LLM HTTP ${res.status}: ${(await res.text()).slice(0, 300)}`);
  const data = (await res.json()) as { choices?: { message?: ChatMessage }[] };
  const message = data.choices?.[0]?.message;
  if (!message) throw new Error('LLM returned no message');
  return message;
}
