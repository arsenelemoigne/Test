import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "crypto";
import * as sb from "./supabase";

// Step-by-step record of an operation, shown in the "coulisses" panel.
export type TraceStep = {
  at: string;
  kind: "ia" | "api" | "etape";
  name: string;
  detail?: string;
  input?: unknown;
  output?: unknown;
  ms?: number;
  ok: boolean;
  // Readable layer for the "déroulé": what was asked, what came back, pages consulted, model reasoning.
  what?: string;
  summary?: string;
  links?: { url: string; title?: string }[];
  thinking?: string;
};

export type TraceRecord = { id: string; kind: string; title: string; status: "running" | "done" | "error"; steps: TraceStep[]; result: unknown; createdAt: string; finishedAt: string | null };

const MAX_CHARS = 1500;
// Local development only (no Supabase); shared by every route of the dev server.
const g = globalThis as unknown as { __ipwTraces?: Map<string, TraceRecord> };
const memory = (g.__ipwTraces ??= new Map<string, TraceRecord>());

function trim(value: unknown): unknown {
  if (value === undefined || value === null) return value;
  const s = typeof value === "string" ? value : JSON.stringify(value);
  if (s.length <= MAX_CHARS) return typeof value === "string" ? value : JSON.parse(s);
  return `${s.slice(0, MAX_CHARS)}… (${s.length} caractères)`;
}

export class Trace {
  readonly steps: TraceStep[] = [];
  private flushTimer: ReturnType<typeof setTimeout> | null = null;
  private status: TraceRecord["status"] = "running";
  private result: unknown = null;

  constructor(readonly id: string, readonly kind: string, readonly title: string, private readonly clientId: string | null, private readonly persisted: boolean) {}

  add(step: Omit<TraceStep, "at">) {
    this.steps.push({ ...step, input: trim(step.input), output: trim(step.output), thinking: step.thinking?.slice(0, 6000), at: new Date().toISOString() });
    this.schedule();
  }

  async finish(status: "done" | "error", result: unknown) {
    this.status = status;
    this.result = trim(result);
    if (this.flushTimer) clearTimeout(this.flushTimer);
    await this.flush(true);
  }

  private schedule() {
    if (this.flushTimer) return;
    this.flushTimer = setTimeout(() => {
      this.flushTimer = null;
      void this.flush(false);
    }, 600);
  }

  private async flush(final: boolean) {
    const record = { steps: this.steps, status: this.status, result: this.result, ...(final ? { finished_at: new Date().toISOString() } : {}) };
    if (!this.persisted) {
      const m = memory.get(this.id);
      if (m) Object.assign(m, { steps: [...this.steps], status: this.status, result: this.result, finishedAt: final ? new Date().toISOString() : null });
      return;
    }
    try {
      await sb.update("traces", { id: `eq.${this.id}` }, record);
    } catch {
      // A trace must never break the operation it describes.
    }
  }

  async create() {
    const createdAt = new Date().toISOString();
    if (!this.persisted) {
      memory.set(this.id, { id: this.id, kind: this.kind, title: this.title, status: "running", steps: [], result: null, createdAt, finishedAt: null });
      return;
    }
    try {
      await sb.insert("traces", [{ id: this.id, client_id: this.clientId, kind: this.kind, title: this.title }]);
    } catch {
      // Same as above: tracing is best effort.
    }
  }
}

const als = new AsyncLocalStorage<Trace>();

export function currentTrace(): Trace | null {
  return als.getStore() ?? null;
}

const UUID = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i;

// The browser may pick the id so it can follow the trace live while the request runs.
export async function startTrace(kind: string, title: string, clientId: string | null, id?: string | null): Promise<Trace> {
  const t = new Trace(id && UUID.test(id) ? id : randomUUID(), kind, title, clientId, sb.supabaseConfigured());
  await t.create();
  return t;
}

// Runs fn with the trace as ambient context, so nested AI and API calls record themselves.
export function runInTrace<T>(trace: Trace, fn: () => Promise<T>): Promise<T> {
  return als.run(trace, fn);
}

export function traceStep(step: Omit<TraceStep, "at">) {
  currentTrace()?.add(step);
}

export async function getTrace(id: string): Promise<TraceRecord | null> {
  if (!sb.supabaseConfigured()) return memory.get(id) ?? null;
  const rows = await sb.select<{ id: string; kind: string; title: string; status: TraceRecord["status"]; steps: TraceStep[]; result: unknown; created_at: string; finished_at: string | null }>("traces", { id: `eq.${id}` });
  const r = rows[0];
  return r ? { id: r.id, kind: r.kind, title: r.title, status: r.status, steps: r.steps ?? [], result: r.result, createdAt: r.created_at, finishedAt: r.finished_at } : null;
}
