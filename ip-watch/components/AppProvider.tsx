"use client";

import { createContext, useContext } from "react";
import type { Client, Role } from "@/lib/types";

type AppValue = { role: Role; client: Client | null; assistantEmail: string | null };

const AppContext = createContext<AppValue | null>(null);

export function AppProvider({ value, children }: { value: AppValue; children: React.ReactNode }) {
  return <AppContext.Provider value={value}>{children}</AppContext.Provider>;
}

export function useApp(): AppValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}
