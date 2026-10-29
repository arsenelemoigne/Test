"use client";

import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import type { ClientData, Finding, Right, Role } from "@/lib/types";

export type FieldEdit = { value: unknown; prev: unknown; by: Role; at: string };
type EditMap = Record<string, Record<string, FieldEdit>>;
type Edits = { rights: EditMap; findings: EditMap };

export type HistoryEntry = { kind: "titre" | "signalement"; id: string; field: string } & FieldEdit;

export type Settings = { lawyerEmail: string | null; assistantEmail: string | null; assistantEnabled: boolean };

type EditsContextValue = {
  client: ClientData;
  role: Role;
  settings: Settings;
  rights: Right[];
  findings: Finding[];
  rightEdits: EditMap;
  findingEdits: EditMap;
  history: HistoryEntry[];
  editRight: (id: string, field: keyof Right, value: unknown) => void;
  editFinding: (id: string, field: keyof Finding, value: unknown) => void;
  resetEdits: () => void;
};

const EditsContext = createContext<EditsContextValue | null>(null);

const EMPTY: Edits = { rights: {}, findings: {} };

function storageKey(slug: string) {
  return `ipw-edits-${slug}`;
}

function load(slug: string): Edits {
  try {
    const raw = window.localStorage.getItem(storageKey(slug));
    return raw ? { ...EMPTY, ...JSON.parse(raw) } : EMPTY;
  } catch {
    return EMPTY;
  }
}

function save(slug: string, edits: Edits) {
  try {
    window.localStorage.setItem(storageKey(slug), JSON.stringify(edits));
  } catch {
    // Storage can be unavailable (private mode); edits then last for the session only.
  }
}

function apply<T extends { id: string }>(items: T[], edits: EditMap): T[] {
  return items.map((item) => {
    const e = edits[item.id];
    if (!e) return item;
    const patch = Object.fromEntries(Object.entries(e).map(([k, v]) => [k, v.value]));
    return { ...item, ...patch };
  });
}

export function EditsProvider({ client, role, settings, children }: { client: ClientData; role: Role; settings: Settings; children: React.ReactNode }) {
  const [edits, setEdits] = useState<Edits>(EMPTY);

  useEffect(() => {
    setEdits(load(client.slug));
  }, [client.slug]);

  const update = useCallback(
    (bucket: keyof Edits, base: { id: string }[], id: string, field: string, value: unknown) => {
      setEdits((current) => {
        const original = (base.find((b) => b.id === id) as Record<string, unknown> | undefined)?.[field];
        const existing = current[bucket][id]?.[field];
        const prev = existing ? existing.value : original;
        if (JSON.stringify(prev) === JSON.stringify(value)) return current;
        const next: Edits = {
          ...current,
          [bucket]: {
            ...current[bucket],
            [id]: { ...current[bucket][id], [field]: { value, prev: existing?.prev ?? original, by: role, at: new Date().toISOString() } },
          },
        };
        save(client.slug, next);
        return next;
      });
    },
    [client.slug, role],
  );

  const value = useMemo<EditsContextValue>(() => {
    const history: HistoryEntry[] = [];
    for (const [id, fields] of Object.entries(edits.rights)) {
      for (const [field, e] of Object.entries(fields)) history.push({ kind: "titre", id, field, ...e });
    }
    for (const [id, fields] of Object.entries(edits.findings)) {
      for (const [field, e] of Object.entries(fields)) history.push({ kind: "signalement", id, field, ...e });
    }
    history.sort((a, b) => b.at.localeCompare(a.at));
    return {
      client,
      role,
      settings,
      rights: apply(client.rights, edits.rights),
      findings: apply(client.findings, edits.findings),
      rightEdits: edits.rights,
      findingEdits: edits.findings,
      history,
      editRight: (id, field, v) => update("rights", client.rights, id, field as string, v),
      editFinding: (id, field, v) => update("findings", client.findings, id, field as string, v),
      resetEdits: () => {
        save(client.slug, EMPTY);
        setEdits(EMPTY);
      },
    };
  }, [client, role, settings, edits, update]);

  return <EditsContext.Provider value={value}>{children}</EditsContext.Provider>;
}

export function useData(): EditsContextValue {
  const ctx = useContext(EditsContext);
  if (!ctx) throw new Error("useData must be used inside EditsProvider");
  return ctx;
}
