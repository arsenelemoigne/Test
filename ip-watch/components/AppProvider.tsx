"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";
import type { Client, Role } from "@/lib/types";

type AppValue = {
  role: Role;
  client: Client | null;
  assistantEmail: string | null;
  // "Coulisses": shows what happens under the hood (AI decisions, API calls). Off by default, remembered per browser.
  coulisses: boolean;
  setCoulisses: (on: boolean) => void;
};

const AppContext = createContext<AppValue | null>(null);
const KEY = "ipw-coulisses";

export function AppProvider({ value, children }: { value: Omit<AppValue, "coulisses" | "setCoulisses">; children: React.ReactNode }) {
  const [coulisses, setState] = useState(false);

  useEffect(() => {
    try {
      setState(window.localStorage.getItem(KEY) === "1");
    } catch {
      // Storage can be unavailable; the panel then stays off.
    }
  }, []);

  const setCoulisses = useCallback((on: boolean) => {
    setState(on);
    try {
      window.localStorage.setItem(KEY, on ? "1" : "0");
    } catch {
      // Same as above.
    }
  }, []);

  return <AppContext.Provider value={{ ...value, coulisses, setCoulisses }}>{children}</AppContext.Provider>;
}

export function useApp(): AppValue {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error("useApp must be used inside AppProvider");
  return ctx;
}
