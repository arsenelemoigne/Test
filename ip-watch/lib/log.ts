import { insert, supabaseConfigured } from "./supabase";
import { traceStep } from "./trace";

// Best-effort trace of external calls, read back to debug integrations; never blocks the caller.
export async function logCall(kind: string, name: string, input: unknown, output: unknown, status: number | null, durationMs: number, clientSlug?: string): Promise<void> {
  if (!supabaseConfigured()) return;
  try {
    const trimmed = typeof output === "string" ? output.slice(0, 4000) : JSON.parse(JSON.stringify(output ?? null, (_k, v) => (typeof v === "string" && v.length > 2000 ? `${v.slice(0, 2000)}…` : v)));
    await insert("assistant_logs", [{ kind, name, input: input ?? null, output: trimmed, status, duration_ms: durationMs, client_slug: clientSlug ?? null }]);
  } catch {
    // Logging must never break the request.
  }
}

// The string values of a call's input, for the readable déroulé.
const describe = (input: unknown) =>
  input && typeof input === "object" ? Object.values(input as Record<string, unknown>).filter((v) => typeof v === "string").join(" · ").slice(0, 160) || undefined : undefined;

// Wraps an external call so its input, output summary and status land in the log.
export async function traced<T>(kind: string, name: string, input: unknown, fn: () => Promise<T>, summarize: (v: T) => unknown = (v) => v): Promise<T> {
  const started = Date.now();
  try {
    const value = await fn();
    const summary = summarize(value);
    void logCall(kind, name, input, summary, 200, Date.now() - started);
    traceStep({ kind: "api", name, input, output: summary, ms: Date.now() - started, ok: true, what: describe(input) });
    return value;
  } catch (e) {
    const message = e instanceof Error ? e.message : String(e);
    void logCall(kind, name, input, message, 500, Date.now() - started);
    traceStep({ kind: "api", name, input, output: message, ms: Date.now() - started, ok: false, what: describe(input), summary: `échec : ${message.slice(0, 160)}` });
    throw e;
  }
}
